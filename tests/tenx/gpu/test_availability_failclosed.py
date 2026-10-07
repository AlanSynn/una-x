"""Fail-closed availability: whichever host runs this suite, the probe must
report typed truth — absence where hardware is absent, presence with named
device identity where it exists — and never claim Metal/MPS as a target."""

import gpu.backends.gpu as gpub
from gpu.runtime.availability import detect_nvcc, probe_driver


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
