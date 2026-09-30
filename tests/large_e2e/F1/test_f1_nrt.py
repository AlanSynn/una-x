"""F1 NRT census (proof 10): T11 — bounded NUMBA_NRT_STATS children (one
per arm root, each with its own NUMBA_CACHE_DIR) replay ONE selected
observed O2 record and report per-call (alloc, free) deltas. Parent
asserts: no leak (alloc == free per call), arm-equality of the per-call
counts, and the H05 rate of 48 NRT allocations per call."""
from __future__ import annotations

import json
import os
import subprocess

from harness_f1 import b0ns, cns, flush_artifacts, record_census


from _legacy_paths import LEGACY_WS  # portable-path fix (HARNESS 2026-09-30)
HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN_DATA = str(LEGACY_WS) + "/campaign_data"
VENV_PY = str(LEGACY_WS) + "/venvs/campaign/bin/python"
H05_ALLOC_RATE = 48


def _run_child(tag, root, tmp_path):
    out = str(tmp_path / f"nrt_child_{tag}.json")
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = os.path.join(CAMPAIGN_DATA, f"nbc_f1_nrt_{tag}")
    env["NUMBA_NUM_THREADS"] = "2"
    env["NUMBA_NRT_STATS"] = "1"
    env.pop("L1_REUSE_DIR", None)
    proc = subprocess.run(
        [VENV_PY, os.path.join(HERE, "nrt_child_f1.py"), root, out],
        env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, \
        f"{tag} nrt child failed:\n{proc.stdout}\n{proc.stderr}"
    assert "F1_NRT_CHILD_OK" in proc.stdout
    with open(out) as fh:
        return json.load(fh)


def test_t11_nrt_census(tmp_path):
    b0 = _run_child("b0", b0ns().root, tmp_path)
    c = _run_child("cand", cns().root, tmp_path)
    for tag, payload in (("b0", b0), ("cand", c)):
        for i, call in enumerate(payload["per_call"]):
            assert call["alloc"] == call["free"], \
                f"{tag}/call{i}: NRT leak {call}"
    allocs_b = [c_["alloc"] for c_ in b0["per_call"]]
    allocs_c = [c_["alloc"] for c_ in c["per_call"]]
    assert allocs_b == allocs_c, \
        f"per-call alloc counts diverged: {allocs_b} vs {allocs_c}"
    assert allocs_c[0] == H05_ALLOC_RATE, \
        (f"per-call alloc rate {allocs_c[0]} != H05 frozen rate "
         f"{H05_ALLOC_RATE}")
    record_census("nrt_per_call", {
        "b0": b0["per_call"], "cand": c["per_call"],
        "h05_rate": H05_ALLOC_RATE})
    flush_artifacts()
