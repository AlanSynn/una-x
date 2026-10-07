"""G00 hardware/backend audit probe (read-only).

Probes THIS host for GPU accelerator stacks and records presence/absence.
Never claims a device exists because a library imports. Prints one JSON object.
Run with PYTHONDONTWRITEBYTECODE=1.
"""
import json
import platform
import subprocess
import sys


def try_import(name):
    try:
        mod = __import__(name)
        return {"importable": True, "version": getattr(mod, "__version__", None)}
    except Exception as exc:  # absence is a fact to record, not an error to raise
        return {"importable": False, "error_class": type(exc).__name__, "error": str(exc)[:200]}


result = {
    "python": {
        "executable": sys.executable,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    },
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "macos_version": platform.mac_ver()[0],
    },
    "accelerator_stacks": {
        # CUDA-class (qualification class per dossier 08)
        "cupy": try_import("cupy"),
        "torch_cuda": try_import("torch"),
        "pycuda": try_import("pycuda"),
        "jax": try_import("jax"),
        # Apple-class (NOT a qualification target, recorded for completeness)
        "torch_mps_check": None,
        "mlx": try_import("mlx"),
    },
    "compute_stack_in_use": {
        "numpy": try_import("numpy"),
        "numba": try_import("numba"),
        "scipy": try_import("scipy"),
    },
}

# Interrogate torch CUDA/MPS honestly if torch imported.
torch_info = result["accelerator_stacks"]["torch_cuda"]
if torch_info["importable"]:
    import torch

    torch_info["cuda_available"] = torch.cuda.is_available()
    torch_info["cuda_device_count"] = torch.cuda.device_count()
    torch_info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    torch_info["mps_built"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    torch_info["cuda_version_string"] = torch.version.cuda  # None when CPU-only build
    result["accelerator_stacks"]["torch_mps_check"] = "see torch_cuda record"

# Toolchain probes on PATH (absence expected locally).
for tool, args in (("nvcc", ("--version",)), ("nvidia-smi", ()), ("metal_completion", ("metal", "--version"))):
    try:
        proc = subprocess.run([tool, *args], capture_output=True, text=True, timeout=30)
        result["accelerator_stacks"][f"toolchain_{tool}"] = {
            "on_path": proc.returncode == 0,
            "rc": proc.returncode,
            "head": (proc.stdout or proc.stderr).strip()[:200],
        }
    except FileNotFoundError:
        result["accelerator_stacks"][f"toolchain_{tool}"] = {"on_path": False, "error": "FileNotFoundError"}

print(json.dumps(result, indent=2))
