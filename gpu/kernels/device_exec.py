"""Device execution for the fold-region gate (G11, Delta qualification).

Loads the COMMITTED CUDA source (gpu/kernels/fold_kernels.cu) through nvcc
and the CUDA driver API via ctypes, so the executed artifact is the
committed .cu bytes and the recorded binary fingerprint is the real
(nvcc, arch, flags) triple.  No CUDA-runtime dependency, no numba.cuda:
this module needs only libcuda + nvcc.

FAIL-CLOSED LAW: every failure here is a typed ``DeviceGateError``
(toolchain/context class -> gate exit 3), never a fallback, never a
partial success.  A device that cannot execute the committed kernels
yields no gate verdict at all.

Compile-flag contract (fold_kernels.cu header, mirrored by nvcc_flags()):
  -prec-sqrt=true -prec-div=true -fmad=false   (fp64; no fast-math)

Driver API notes (constants from cuda.h; stable since CUDA 2.x/4.x):
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75, MINOR = 76
  CU_CTX_SCHED_AUTO = 0 (primary context default)
"""

from __future__ import annotations

import ctypes
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

__all__ = ["DeviceGateError", "DeviceIdentity", "DeviceRuntime",
           "compile_ptx", "nvcc_flags"]

_CU_SUCCESS = 0
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
_DRIVER_LIB_CANDIDATES = ("libcuda.so.1", "libcuda.so")

_CU_PARAM_TR = ctypes.c_void_p  # kernel-param slots are pointers to values


class DeviceGateError(Exception):
    """Typed toolchain/context failure (gate exit code 3 class).

    Raised for: driver library present but the launch API surface is
    unusable, nvcc missing/failing, module load failure, launch failure,
    or a context error.  Distinct from HardwareUnavailable (probe class,
    exit 2): a DEVICE EXISTS here but cannot be driven end to end.
    """


@dataclass(frozen=True)
class DeviceIdentity:
    """The recorded identity of the device actually driven."""

    name: str
    compute_capability: Tuple[int, int]
    sm_name: str                 # e.g. "sm_80" (nvcc -arch form: compute_80)
    driver_version: Tuple[int, int]   # (major, minor) from cuDriverGetVersion
    device_index: int


def nvcc_flags(arch: int) -> Tuple[str, ...]:
    """The contractual flag set (fold_kernels.cu header law).

    ``arch`` is the numeric SM (80, 86, 90, ...).  -prec-sqrt/-prec-div on,
    -fmad off, C++17, optimization on; --use_fast_math is FORBIDDEN and its
    absence is part of this contract.
    """
    return ("-O3", "-std=c++17", "-prec-sqrt=true", "-prec-div=true",
            "-fmad=false", f"-arch=compute_{arch}", "-ptx")


def compile_ptx(cu_source: Path, arch: int, nvcc_path: str,
                out_dir: Path) -> Path:
    """Compile the committed .cu to PTX with the contractual flags.

    The PTX lands in ``out_dir`` under a name binding the arch — keep it:
    it is the binary_fingerprint artifact the evidence records.
    """
    flags = nvcc_flags(arch)
    out_ptx = out_dir / f"fold_kernels_compute_{arch}.ptx"
    cmd = [nvcc_path, *flags, "-o", str(out_ptx), str(cu_source)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeviceGateError(f"nvcc invoke failed: {exc}") from exc
    if proc.returncode != 0 or not out_ptx.is_file():
        raise DeviceGateError(
            f"nvcc failed rc={proc.returncode}: {(proc.stderr or '')[-800:]}")
    return out_ptx


def _check(rc: int, what: str) -> None:
    if rc != _CU_SUCCESS:
        raise DeviceGateError(f"{what} failed with CUDA rc={rc}")


class DeviceRuntime:
    """Minimal ctypes CUDA driver-API runtime for the gate kernels.

    Owns: device 0's primary context, module (from PTX), device buffers,
    H2D/D2H copies, kernel launches, synchronization.  Pure ctypes: the
    kernels come from the committed .cu via nvcc, never from a re-derived
    in-Python reimplementation.
    """

    def __init__(self, device_index: int = 0):
        self._lib = None
        self._ctx = ctypes.c_void_p()
        self._module = ctypes.c_void_p()
        self._fns: Dict[str, ctypes.c_void_p] = {}
        self._allocs = []
        for name in _DRIVER_LIB_CANDIDATES:
            try:
                self._lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if self._lib is None:
            raise DeviceGateError("libcuda not loadable (probe said a "
                                  "driver exists; launch API absent)")
        lib = self._lib
        lib.cuInit.argtypes = [ctypes.c_uint]
        _check(lib.cuInit(0), "cuInit")

        lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        ver = ctypes.c_int(0)
        _check(lib.cuDriverGetVersion(ctypes.byref(ver)),
               "cuDriverGetVersion")
        driver_v = (ver.value // 1000, (ver.value % 1000) // 10)

        self._dev = ctypes.c_int(0)
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int),
                                    ctypes.c_int]
        _check(lib.cuDeviceGet(ctypes.byref(self._dev),
                               ctypes.c_int(device_index)), "cuDeviceGet")

        def attr(a: int) -> int:
            v = ctypes.c_int(0)
            lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int),
                                                 ctypes.c_int,
                                                 ctypes.c_int]
            _check(lib.cuDeviceGetAttribute(ctypes.byref(v),
                                            ctypes.c_int(a), self._dev),
                   f"cuDeviceGetAttribute({a})")
            return v.value

        major, minor = attr(_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR), \
            attr(_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)

        name_buf = ctypes.create_string_buffer(256)
        lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_int]
        _check(lib.cuDeviceGetName(name_buf, 256, self._dev),
               "cuDeviceGetName")

        lib.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        _check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(self._ctx),
                                            self._dev),
               "cuDevicePrimaryCtxRetain")
        lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        _check(lib.cuCtxSetCurrent(self._ctx), "cuCtxSetCurrent")

        self.identity = DeviceIdentity(
            name=name_buf.value.decode("utf-8", "replace"),
            compute_capability=(major, minor),
            sm_name=f"sm_{major * 10 + minor}",
            driver_version=driver_v,
            device_index=device_index,
        )

    # -- module -----------------------------------------------------------

    def load_ptx(self, ptx_path: Path) -> None:
        lib = self._lib
        image = ctypes.create_string_buffer(ptx_path.read_bytes())
        lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.c_char_p]
        _check(lib.cuModuleLoadData(ctypes.byref(self._module), image),
               "cuModuleLoadData")
        lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        for kname in ("una_g10_fold_compact",
                      "una_g10_fold_coefficients_exp",
                      "una_g10_fold_coefficients_log",
                      "una_g10_fold_knn_coeffs_exp",
                      "una_g10_fold_knn_coeffs_log",
                      "una_g10_fold_knn_coeffs_linear"):
            fn = ctypes.c_void_p()
            _check(lib.cuModuleGetFunction(ctypes.byref(fn), self._module,
                                           kname.encode()),
                   f"cuModuleGetFunction({kname})")
            self._fns[kname] = fn

    # -- memory ------------------------------------------------------------

    def h2d(self, arr: np.ndarray) -> ctypes.c_void_p:
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        dev = ctypes.c_void_p()
        self._lib.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.c_size_t]
        _check(self._lib.cuMemAlloc(ctypes.byref(dev), nbytes), "cuMemAlloc")
        self._allocs.append(dev)
        self._lib.cuMemcpyHtoD.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_size_t]
        src = arr.ctypes.data_as(ctypes.c_void_p)
        _check(self._lib.cuMemcpyHtoD(dev, src, nbytes), "cuMemcpyHtoD")
        return dev

    def d2h(self, dev: ctypes.c_void_p, like: np.ndarray) -> np.ndarray:
        out = np.empty_like(like)
        self._lib.cuMemcpyDtoH.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_size_t]
        dst = out.ctypes.data_as(ctypes.c_void_p)
        _check(self._lib.cuMemcpyDtoH(dst, dev, like.nbytes), "cuMemcpyDtoH")
        return out

    def free_all(self) -> None:
        self._lib.cuMemFree.argtypes = [ctypes.c_void_p]
        for dev in self._allocs:
            try:
                self._lib.cuMemFree(dev)
            except Exception:  # noqa: BLE001 - free path must not mask
                pass
        self._allocs.clear()

    # -- launch ------------------------------------------------------------

    def launch(self, kname: str, grid: Tuple[int, int, int],
               block: Tuple[int, int, int], params: tuple,
               shared_bytes: int = 0) -> None:
        """Launch a loaded kernel.

        ``params`` holds the kernel arguments IN ORDER; array/device
        arguments pass as ctypes.c_void_p(device pointer), scalars as
        ctypes values (c_longlong / c_double / ...).  The arg list built
        here mirrors fold_kernels.cu signatures exactly; a signature drift
        fails at launch (rc), which is a typed DeviceGateError — never a
        silent misread.
        """
        # One holder per kernel parameter; cuLaunchKernel's params array
        # points AT each holder.  Device pointers sit in the holder BY
        # VALUE; holders stay referenced until the launch returns.
        holders = []
        for p in params:
            if isinstance(p, ctypes._Pointer):
                holders.append((ctypes.c_void_p * 1)(p.value))
            else:
                holders.append((type(p) * 1)(p))
        argv = (_CU_PARAM_TR * len(holders))(
            *[ctypes.cast(ctypes.addressof(h), _CU_PARAM_TR)
              for h in holders])
        self._lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_void_p, ctypes.POINTER(_CU_PARAM_TR), ctypes.c_void_p]
        rc = self._lib.cuLaunchKernel(
            self._fns[kname], grid[0], grid[1], grid[2],
            block[0], block[1], block[2], shared_bytes, None,
            argv if holders else None, None)
        _check(rc, f"cuLaunchKernel({kname})")

    def sync(self) -> None:
        self._lib.cuCtxSynchronize.argtypes = []
        _check(self._lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def close(self) -> None:
        self.free_all()
        if self._lib is not None and self._module.value:
            self._lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
            try:
                self._lib.cuModuleUnload(self._module)
            except Exception:  # noqa: BLE001
                pass
            self._module = ctypes.c_void_p()
        if self._lib is not None and self._ctx.value:
            self._lib.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
            try:
                self._lib.cuDevicePrimaryCtxRelease(self._dev)
            except Exception:  # noqa: BLE001
                pass
            self._ctx = ctypes.c_void_p()
