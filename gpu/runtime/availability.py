"""Fail-closed CUDA availability probing.

Detects a CUDA driver/device with REAL probes (ctypes against the CUDA
driver library) and reports typed results.  Absence is a fact to report,
never an error to raise out of ``probe_driver`` and never a state that any
code path may quietly reinterpret as "GPU supported".

Qualification requirements recorded here mirror gpu_region.json:
  1. CUDA-class device (Apple Metal/MPS/MLX are NOT qualification targets),
  2. hardware FP64 (CU_DEVICE_ATTRIBUTE_DOUBLE_PRECISION != 0),
  3. per-cell bit-validation gate passed on that device (tracked by
     gpu.backends.gpu cell-gate state, not by this module).

Only (1)+(2) are probed here; (3) is deliberately out of scope for a
detection module so that "detected" can never be conflated with "qualified".
"""

import ctypes
import ctypes.util
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Tuple

__all__ = ["DriverProbe", "NvccInfo", "probe_driver", "detect_nvcc"]

# Candidate CUDA driver library names, most standard first.  macOS has no
# CUDA driver at all; the list is tried in order and failure is recorded.
_CUDA_DRIVER_CANDIDATES = (
    "libcuda.so.1",   # standard Linux
    "libcuda.so",     # uncommon Linux layout
    "nvcuda.dll",     # Windows (ctypes.util finds it via PATH)
)

# CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR == 75 and ..._MINOR == 76
# (CUDA driver API enum; stable since CUDA 5.0).  Native FP64 support is a
# property of compute capability: every CUDA device with CC >= 1.3 has it,
# and no CC > 1.3 device lacks it (CUDA C Programming Guide, compute
# capabilities table).  Constants recorded here because we load the driver
# by ctypes and cannot include cuda.h.
_CU_DEVICE_ATTRIBUTE_CC_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_CC_MINOR = 76
_MIN_FP64_CAPABILITY = (1, 3)

_CU_SUCCESS = 0
_CU_ERROR_NOT_INITIALIZED = 1
_CU_ERROR_NO_DEVICE = 100


@dataclass(frozen=True)
class DriverProbe:
    """Result of probing this host for a usable CUDA driver/device."""

    available: bool
    reason: str            # machine-readable refusal class when unavailable
    detail: str            # human-readable evidence (paths/errors observed)
    device_count: int = 0
    fp64_capable: bool = False
    device_names: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def unavailable_reason(self) -> str:
        """The typed refusal class for this probe (always meaningful)."""
        return self.reason if not self.available else "available"


def _driver_result(available: bool, reason: str, detail: str,
                   device_count: int = 0, fp64_capable: bool = False,
                   names: Tuple[str, ...] = ()) -> DriverProbe:
    return DriverProbe(available=available, reason=reason, detail=detail,
                       device_count=device_count,
                       fp64_capable=fp64_capable, device_names=names)


def probe_driver() -> DriverProbe:
    """Probe for a CUDA driver and at least one FP64-capable device.

    Never raises: every failure mode becomes a typed, reported refusal.
    """
    if platform.system() == "Darwin":
        # Recorded fact (gpu_region.json hardware audit): macOS ships no
        # CUDA driver; Apple GPUs are explicitly not qualification targets.
        return _driver_result(
            False, "no_cuda_driver",
            "platform=Darwin: no CUDA driver exists for this OS; "
            "Apple GPU present but not a qualification target")

    lib = None
    tried = []
    for name in _CUDA_DRIVER_CANDIDATES:
        path = ctypes.util.find_library(name.replace(".so", "")) or name
        try:
            lib = ctypes.CDLL(path)
            tried.append(f"{name}:loaded({path})")
            break
        except OSError as exc:
            tried.append(f"{name}:{exc}")
            lib = None
    if lib is None:
        return _driver_result(False, "no_cuda_driver", "; ".join(tried))

    try:
        rc = lib.cuInit(0)
        if rc != _CU_SUCCESS:
            return _driver_result(False, "cuInit_failed", f"cuInit rc={rc}")
        count = ctypes.c_int(0)
        rc = lib.cuDeviceGetCount(ctypes.byref(count))
        if rc != _CU_SUCCESS:
            return _driver_result(False, "cuDeviceGetCount_failed",
                                  f"rc={rc}")
        if count.value < 1:
            return _driver_result(False, "no_cuda_device",
                                  "cuDeviceGetCount=0")
        # Enumerate devices; require FP64 on the first device.
        names = []
        fp64_first = False
        for i in range(count.value):
            dev = ctypes.c_int(0)
            if lib.cuDeviceGet(ctypes.byref(dev), ctypes.c_int(i)) != _CU_SUCCESS:
                continue
            # cuDeviceGetName writes narrow char bytes (driver API char*);
            # reading them through a wchar buffer misdecodes the device
            # name — U+4449564E on the real A100 ("NVID" as one UTF-32
            # word).  Read as bytes, decode explicitly.
            name_buf = ctypes.create_string_buffer(256)
            rc = lib.cuDeviceGetName(name_buf, 256, dev)
            if rc == _CU_SUCCESS:
                names.append(name_buf.value.decode("utf-8", "replace"))
            if i == 0:
                major = ctypes.c_int(0)
                minor = ctypes.c_int(0)
                rc_maj = lib.cuDeviceGetAttribute(
                    ctypes.byref(major),
                    ctypes.c_int(_CU_DEVICE_ATTRIBUTE_CC_MAJOR), dev)
                rc_min = lib.cuDeviceGetAttribute(
                    ctypes.byref(minor),
                    ctypes.c_int(_CU_DEVICE_ATTRIBUTE_CC_MINOR), dev)
                fp64_first = (
                    rc_maj == _CU_SUCCESS and rc_min == _CU_SUCCESS
                    and (major.value, minor.value) >= _MIN_FP64_CAPABILITY)
        if not fp64_first:
            return _driver_result(
                False, "no_fp64_device",
                f"devices={names}; compute capability below "
                f"{_MIN_FP64_CAPABILITY[0]}.{_MIN_FP64_CAPABILITY[1]} on "
                "device 0; "
                "FP32 substitution is forbidden (dossier 08)",
                device_count=count.value, fp64_capable=False,
                names=tuple(names))
        return _driver_result(True, "available",
                              f"devices={names}",
                              device_count=count.value, fp64_capable=True,
                              names=tuple(names))
    except AttributeError as exc:
        # Driver library loaded but lacks a probed symbol: still fail closed.
        return _driver_result(False, "driver_symbols_missing", str(exc))


@dataclass(frozen=True)
class NvccInfo:
    on_path: bool
    version: str = ""      # empty when absent
    detail: str = ""


def detect_nvcc() -> NvccInfo:
    """Detect the nvcc toolchain (OPTIONAL for G10: source-level checks only).

    An absent nvcc is recorded, not fatal: device kernels ship as source and
    compile at G11 on the qualification host.
    """
    path = shutil.which("nvcc")
    if path is None:
        return NvccInfo(False, "", "nvcc not on PATH")
    try:
        proc = subprocess.run([path, "--version"], capture_output=True,
                              text=True, timeout=60)
        head = (proc.stdout or "").strip().splitlines()
        version = head[-1].strip() if head else ""
        return NvccInfo(True, version, path)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return NvccInfo(False, "", f"nvcc probe failed: {exc}")
