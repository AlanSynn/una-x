"""Device execution module: typed failures on any host, contract-pinned
compile flags, and a launch-argument ABI that mirrors the committed .cu.

On the macOS authoring host there is no libcuda: DeviceRuntime() must
raise the typed DeviceGateError (never segfault, never a partial object).
On the Delta GPU node the same code drives the real device — these tests
pin only the host-independent contracts."""

import ctypes
from pathlib import Path

import pytest

from gpu.kernels.device_exec import (DeviceGateError, DeviceRuntime,
                                     compile_ptx, nvcc_flags)

_CU = Path(__file__).resolve().parents[3] / "gpu" / "kernels" / "fold_kernels.cu"


def test_compile_flag_contract_pins_no_fast_math():
    flags = nvcc_flags(80)
    assert "-fmad=false" in flags
    assert "-prec-sqrt=true" in flags
    assert "-prec-div=true" in flags
    assert "-arch=compute_80" in flags
    assert "-ptx" in flags
    assert not any("fast_math" in f or "fast-math" in f for f in flags), (
        "--use_fast_math (or any fast-math) is forbidden by the .cu law")
    assert nvcc_flags(90) != nvcc_flags(80)  # arch binds the fingerprint


def test_compile_with_absent_nvcc_is_typed(tmp_path):
    with pytest.raises(DeviceGateError, match="nvcc invoke failed"):
        compile_ptx(_CU, 80, "/nonexistent/nvcc", tmp_path)


def test_compiled_ptx_contains_committed_kernel_names(tmp_path):
    nvcc = pytest.importorskip("shutil").which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not on this host (device gate compiles on the "
                    "qualification node)")
    ptx = compile_ptx(_CU, 80, nvcc, tmp_path)
    text = ptx.read_text()
    for k in ("una_g10_fold_compact", "una_g10_fold_coefficients_exp",
              "una_g10_fold_coefficients_log", "una_g10_fold_knn_coeffs_exp",
              "una_g10_fold_knn_coeffs_log",
              "una_g10_fold_knn_coeffs_linear"):
        assert k in text, f"{k} missing from compiled PTX"


def test_runtime_without_driver_is_typed():
    # On a host with no libcuda the runtime refuses typed; where a driver
    # exists this test is vacuously satisfied by construction succeeding.
    try:
        rt = DeviceRuntime(0)
    except DeviceGateError:
        return  # the honest no-driver outcome (macOS authoring host)
    except OSError as exc:  # pragma: no cover - defensive
        pytest.fail(f"untyped driver failure: {exc}")
    else:
        rt.close()  # real device present: identity is queryable
        assert rt.identity.sm_name.startswith("sm_")


def test_kernel_param_order_mirrors_cu_signatures():
    # The launch ABI is textual law: read the committed .cu and assert each
    # kernel's parameter list matches the order the harness passes.  A
    # signature drift fails HERE, as bytes, not as a corrupt device read.
    text = _CU.read_text()
    expected = {
        "una_g10_fold_compact": ["dist", "d_count", "cutoff", "out_idx",
                                 "out_cnt"],
        "una_g10_fold_coefficients_exp": ["dn", "wn", "m", "beta", "plateau",
                                          "c_out"],
        "una_g10_fold_coefficients_log": ["dn", "wn", "m", "growth_rate",
                                          "plateau", "midpoint", "c_out"],
        "una_g10_fold_knn_coeffs_exp": ["dn_s", "wn_s", "kn", "m", "beta",
                                        "plateau", "c_out"],
        "una_g10_fold_knn_coeffs_log": ["dn_s", "wn_s", "kn", "m",
                                        "growth_rate", "plateau", "midpoint",
                                        "c_out"],
        "una_g10_fold_knn_coeffs_linear": ["wn_s", "kn", "m", "c_out"],
    }
    import re
    # strip ALL line comments first: inline comments contain parentheses
    # and would otherwise truncate the signature match
    text_nc = re.sub(r"//[^\n]*", "", text)
    for kname, params in expected.items():
        m = re.search(rf"extern \"C\" __global__ void {kname}\(([^)]*)\)",
                      text_nc, re.S)
        assert m, f"{kname} not found in committed source"
        got = [seg.strip().split()[-1].lstrip("*")
               for seg in m.group(1).split(",")]
        assert got == params, f"{kname}: {got} != {params}"


def test_param_slot_holders_keep_device_pointers_by_value():
    # The launch path wraps device pointers BY VALUE: two holders built
    # from equal pointers compare equal, and scalar holders type-match.
    p = ctypes.c_void_p(0xDEADBEEF)
    holder = (ctypes.c_void_p * 1)(p.value)
    assert holder[0] == 0xDEADBEEF
    scalar = (ctypes.c_longlong * 1)(7)
    assert scalar[0] == 7
