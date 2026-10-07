"""Fail-closed availability: whichever host runs this suite, the probe must
report typed truth — absence where hardware is absent, presence with named
device identity where it exists — and never claim Metal/MPS as a target."""

import inspect
from pathlib import Path

import gpu.backends.gpu as gpub
from gpu.runtime.availability import (detect_nvcc, probe_driver,
                                      _CU_DEVICE_ATTRIBUTE_CC_MAJOR,
                                      _CU_DEVICE_ATTRIBUTE_CC_MINOR,
                                      _MIN_FP64_CAPABILITY)


def test_driver_result_accepts_both_terminal_branch_kwarg_sets():
    # G11 attempt-2 defect (found on the real A100): the available and
    # no_fp64 branches passed fp64_capable=/names= to _driver_result, whose
    # parameter was named fp64= - unreachable on any host without a device,
    # so only real hardware could fire it.  Exercise both kwarg sets here.
    from gpu.runtime.availability import _driver_result
    ok = _driver_result(True, "available", "devices=('A100',)",
                        device_count=1, fp64_capable=True,
                        names=("A100",))
    assert ok.available is True and ok.fp64_capable is True
    assert ok.device_names == ("A100",)
    no64 = _driver_result(False, "no_fp64_device", "detail",
                          device_count=1, fp64_capable=False,
                          names=("x",))
    assert no64.available is False and no64.fp64_capable is False


def test_device_name_read_is_narrow_char_not_wchar():
    # G11 attempt-1 defect (found on the real A100): cuDeviceGetName writes
    # narrow char bytes; reading them through create_unicode_buffer raised
    # ValueError("character U+4449564E ...") — "NVID" as one UTF-32 word.
    # The driver API signature is fixed, so the contract is textual: the
    # probe must read the name as a byte buffer and decode explicitly.
    src = inspect.getsource(probe_driver)
    assert "create_unicode_buffer" not in src
    assert "create_string_buffer" in src
    assert ".decode(" in src
    # No other runtime module may reintroduce a wchar read of driver bytes.
    for mod in (gpub,):
        assert "create_unicode_buffer" not in inspect.getsource(mod)


def test_fp64_criterion_is_documented_compute_capability():
    # The FP64 criterion must be the documented compute-capability rule
    # (CC >= 1.3) read from attributes 75/76 — not an unverified enum value
    # (the prior constant 13 is KERNEL_EXEC_TIMEOUT in the real enum).
    assert _CU_DEVICE_ATTRIBUTE_CC_MAJOR == 75
    assert _CU_DEVICE_ATTRIBUTE_CC_MINOR == 76
    assert _MIN_FP64_CAPABILITY == (1, 3)
    src = inspect.getsource(probe_driver)
    assert "_CU_DEVICE_ATTRIBUTE_CC_MAJOR" in src
    assert "DOUBLE_PRECISION" not in src


def test_probe_reports_typed_truth():
    probe = probe_driver()
    if probe.available:
        # A CUDA host: presence must be named, FP64 asserted.
        assert probe.reason == "available"
        assert probe.device_count >= 1
        assert probe.fp64_capable is True
        assert probe.device_names, "available probe must name devices"
    else:
        assert probe.reason in ("no_cuda_driver", "no_cuda_device",
                                "no_fp64_device", "cuInit_failed",
                                "cuDeviceGetCount_failed",
                                "driver_symbols_missing")
        assert probe.detail, "every refusal carries its evidence detail"


def test_nvcc_state_recorded_not_fatal():
    # Presence and absence are both honest records; neither may raise.
    nvcc = detect_nvcc()
    assert isinstance(nvcc.on_path, bool)
    assert isinstance(nvcc.version, str)


def test_availability_report_shape_and_honesty():
    report = gpub.availability_report()
    assert set(report) >= {"driver", "nvcc", "not_a_qualification_target_note"}
    assert set(report["driver"]) >= {"available", "reason", "detail",
                                     "device_count", "fp64_capable",
                                     "device_names"}
    assert "Metal" in report["not_a_qualification_target_note"]
    assert "CUDA-class FP64" in report["not_a_qualification_target_note"]


def test_requirements_pin_cuda_fp64_and_gate():
    assert gpub.ISA_OR_DEVICE_REQUIREMENTS == (
        "cuda_class_device", "hardware_fp64", "cell_bit_validation_gate")
    assert gpub.SUPPORTED_PROFILES == ("una_legacy",)
