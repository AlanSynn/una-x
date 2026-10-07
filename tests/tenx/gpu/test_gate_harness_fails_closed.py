"""Durable evidence-backed gate store (G10R F2) + gate-harness refusals.

Authority law: a cell is open ONLY through the durable store whose
evidence bytes are re-verified at every read.  There is no in-RAM gate
state to forge.  The harness's no-hardware refusal (exit 2) is exercised
through a faked probe so the contract holds on ANY host, and the
host-battery mode exits 5, never 0 (G10R F4)."""

import json

import pytest

import gpu.backends.gpu as gpub
import gpu.kernels.validate_device_bits as vdb
from gpu.kernels.validate_device_bits import (EXIT_HARDWARE_UNAVAILABLE,
                                              EXIT_HOST_BATTERY, EXIT_PASSED,
                                              main)


class _FakeUnavailable:
    available = False
    reason = "no_cuda_driver"
    detail = "fake probe (test only)"
    device_count = 0
    fp64_capable = False
    device_names = ()


def test_exit_codes_f4_host_battery_is_not_gate_passed():
    assert EXIT_HOST_BATTERY == 5
    assert EXIT_HOST_BATTERY != EXIT_PASSED
    assert EXIT_HARDWARE_UNAVAILABLE == 2


def test_host_battery_mode_returns_5_without_engine(monkeypatch, tmp_path):
    # Dispatch-path check: main() with --emit-host-battery writes the
    # (stubbed) battery and exits 5 — never the gate-passed 0.
    monkeypatch.setattr(vdb, "host_battery_tables",
                        lambda: {"stub": True})
    out = tmp_path / "battery.json"
    rc = main(["--emit-host-battery", str(out)])
    assert rc == EXIT_HOST_BATTERY == 5
    assert json.loads(out.read_text()) == {"stub": True}


def test_no_hardware_refuses_with_designed_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(vdb, "probe_driver", lambda: _FakeUnavailable())
    rc = main([])
    captured = capsys.readouterr()
    assert rc == EXIT_HARDWARE_UNAVAILABLE == 2
    payload = json.loads(captured.out)
    assert payload["status"] == "hardware_unavailable"
    assert "NOT PASSED" in payload["gate"]
    assert payload["driver"]["reason"] == "no_cuda_driver"


def test_default_state_is_closed_everywhere(tmp_path):
    # No store configured, no store file, empty store: all CLOSED.
    assert gpub.cell_is_gated_open("una_legacy.fold.default") is False
    missing = tmp_path / "nope.json"
    assert gpub.cell_is_gated_open("una_legacy.fold.default",
                                   str(missing)) is False
    store = tmp_path / "store.json"
    store.write_text(json.dumps({"cells": {}}))
    assert gpub.cell_is_gated_open("una_legacy.fold.default",
                                   str(store)) is False


def test_record_gate_refuses_unbacked_pass(tmp_path):
    evidence = tmp_path / "evidence.json"
    store = tmp_path / "store.json"
    with pytest.raises(ValueError, match="certified association"):
        gpub.record_gate(str(store), "c", passed=True,
                         association="sum_everything",
                         evidence_ref=str(evidence))
    with pytest.raises(ValueError, match="evidence_ref"):
        gpub.record_gate(str(store), "c", passed=True, association="seq",
                         evidence_ref="")
    with pytest.raises(ValueError, match="does not name an existing file"):
        gpub.record_gate(str(store), "c", passed=True, association="seq",
                         evidence_ref=str(evidence))
    assert gpub.cell_is_gated_open("c", str(store)) is False


def test_passed_gate_opens_only_while_evidence_bytes_hold(tmp_path):
    store = tmp_path / "store.json"
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"device": "a100", "verdict": "passed"}\n')
    entry = gpub.record_gate(str(store), "una_legacy.fold.default",
                             passed=True, association="vec4_seq",
                             evidence_ref=str(evidence))
    assert entry["bit_validated"] is True
    assert entry["evidence_sha256"]
    assert gpub.cell_is_gated_open("una_legacy.fold.default", str(store))
    assert gpub.cell_association("una_legacy.fold.default",
                                 str(store)) == "vec4_seq"

    # Tampering with the evidence bytes CLOSES the cell: the verdict binds
    # to bytes, not to a path.
    evidence.write_text('{"device": "a100", "verdict": "passed", '
                        '"tampered": 1}\n')
    state = gpub.gate_state("una_legacy.fold.default", str(store))
    assert state["open"] is False
    assert "changed" in state["reason"]

    # Deleting the evidence closes it too.
    evidence.unlink()
    assert gpub.cell_is_gated_open("una_legacy.fold.default",
                                   str(store)) is False


def test_refused_entry_records_and_stays_closed(tmp_path):
    store = tmp_path / "store.json"
    evidence = tmp_path / "mismatch.json"
    evidence.write_text('{"status": "battery_mismatch"}\n')
    gpub.record_gate(str(store), "una_legacy.fold.default", passed=False,
                     evidence_ref=str(evidence))
    state = gpub.gate_state("una_legacy.fold.default", str(store))
    assert state["open"] is False
    assert "bit_validated=False" in state["reason"]
    with pytest.raises(ValueError, match="no association"):
        gpub.record_gate(str(store), "c2", passed=False, association="seq")


def test_store_survives_process_like_reload(tmp_path):
    # Durability: a fresh read of the same path re-verifies from bytes.
    store = tmp_path / "store.json"
    evidence = tmp_path / "evidence.json"
    evidence.write_text("evidence-v1\n")
    gpub.record_gate(str(store), "cell-x", passed=True, association="seq",
                     evidence_ref=str(evidence))
    reloaded = json.loads(store.read_text())
    assert reloaded["cells"]["cell-x"]["association"] == "seq"
    assert gpub.gate_state("cell-x", str(store))["open"] is True
