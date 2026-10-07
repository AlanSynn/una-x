"""Fail-closed availability: this host (macOS/Apple Silicon) must probe as
typed-unavailable, and the report must say what exists without ever
claiming capability."""

import gpu.backends.gpu as gpub
from gpu.runtime.availability import detect_nvcc, probe_driver


def test_darwin_probes_typed_unavailable():
    probe = probe_driver()
    assert probe.available is False
    assert probe.reason == "no_cuda_driver"
    assert "not a qualification target" in probe.detail
    assert probe.device_count == 0
    assert probe.fp64_capable is False


def test_nvcc_absence_recorded_not_fatal():
    nvcc = detect_nvcc()
    # Recorded fact from the G00 audit (2026-10-07): nvcc is not on PATH.
    assert nvcc.on_path is False
    assert nvcc.version == ""


def test_availability_report_shape_and_honesty():
    report = gpub.availability_report()
    assert report["driver"]["available"] is False
    assert report["nvcc"]["on_path"] is False
    assert "not_a_qualification_target_note" in report
    assert "Metal" in report["not_a_qualification_target_note"]


def test_requirements_pin_cuda_fp64_and_gate():
    assert gpub.ISA_OR_DEVICE_REQUIREMENTS == (
        "cuda_class_device", "hardware_fp64", "cell_bit_validation_gate")
    assert gpub.SUPPORTED_PROFILES == ("una_legacy",)
