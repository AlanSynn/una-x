"""The gate harness fails closed on this host: rc=2 hardware_unavailable,
gate NOT PASSED, and nothing in the local path marks a cell bit-validated.

(The harness's --emit-host-battery half imports the engine and is run
OUTSIDE pytest with a pinned NUMBA_CACHE_DIR; it is never executed here.)"""

import json

import gpu.backends.gpu as gpub
from gpu.kernels.validate_device_bits import (EXIT_HARDWARE_UNAVAILABLE,
                                              main)


def test_gate_harness_refuses_with_designed_exit_code(capsys):
    rc = main([])
    captured = capsys.readouterr()
    assert rc == EXIT_HARDWARE_UNAVAILABLE == 2
    payload = json.loads(captured.out)
    assert payload["status"] == "hardware_unavailable"
    assert "NOT PASSED" in payload["gate"]
    assert payload["driver"]["reason"] == "no_cuda_driver"


def test_default_cell_is_not_gated_open():
    assert gpub.cell_is_gated_open("una_legacy.fold.default") is False


def test_gate_store_requires_evidence_and_certified_association():
    import pytest
    with pytest.raises(ValueError, match="certified association"):
        gpub.cell_gate("bad-cell", passed=True, association="sum_everything",
                       evidence_ref="x")
    with pytest.raises(ValueError, match="evidence_ref"):
        gpub.cell_gate("bad-cell", passed=True, association="seq",
                       evidence_ref="")
    assert gpub.cell_is_gated_open("bad-cell") is False
