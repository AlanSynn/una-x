"""Shared invocation/compare helpers for the F1 tests (single B0
singleton via the oracle package; single candidate alias via
cand_load). Also collects per-suite evidence tails (first-divergence
records, warm-up wall times, census results) into
evidence/F1I/test_artifacts.json."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from comparator import OracleMismatch, assert_array_bytes_equal
from support import b0 as _b0
import cand_load

EVIDENCE_DIR = ("/Users/alansynn/orca/workspaces/una-x/wt-large-e2e"
                "/campaigns/una_large_e2e/evidence/F1I")
ARTIFACT_PATH = os.path.join(EVIDENCE_DIR, "test_artifacts.json")

_ARTIFACTS = {"warmup_s": {}, "first_divergences": {}, "census": {},
              "notes": {}}


def b0ns():
    return _b0()


def cns():
    return cand_load.cand()


def record_warmup(key, seconds):
    _ARTIFACTS["warmup_s"][key] = round(float(seconds), 6)


def record_divergence(key, payload):
    _ARTIFACTS["first_divergences"][key] = payload


def record_census(key, payload):
    _ARTIFACTS["census"][key] = payload


def record_note(key, payload):
    _ARTIFACTS["notes"][key] = payload


def flush_artifacts():
    """Write the collected suite tail; safe to call multiple times
    (last writer wins with merged content)."""
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    existing = {}
    if os.path.exists(ARTIFACT_PATH):
        with open(ARTIFACT_PATH) as fh:
            try:
                existing = json.load(fh)
            except ValueError:
                existing = {}
    for section, payload in _ARTIFACTS.items():
        existing.setdefault(section, {}).update(payload)
    tmp = ARTIFACT_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(existing, fh, indent=2, sort_keys=True)
    os.replace(tmp, ARTIFACT_PATH)


def timed_first_call(fn, key, *args, **kwargs):
    """Run fn once, recording its wall time under `key` (warm-up cost
    of a specialization; dossier obligation — no performance claim)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    record_warmup(key, time.perf_counter() - t0)
    return out


def assert_kernel_bytes(out_c, out_b, tag):
    """Byte-compare one kernel call's 4 outputs (T1 form)."""
    assert_array_bytes_equal(np.asarray(out_c[0]), np.asarray(out_b[0]),
                             f"{tag}/out_AB")
    assert_array_bytes_equal(np.asarray(out_c[1]), np.asarray(out_b[1]),
                             f"{tag}/out_BA")
    assert_array_bytes_equal(np.asarray(out_c[2]), np.asarray(out_b[2]),
                             f"{tag}/out_node_flow")
    assert float(out_c[3]).hex() == float(out_b[3]).hex(), \
        f"{tag}/delivered not bit-equal: {out_c[3]!r} vs {out_b[3]!r}"


def assert_engine_bytes(arr_c, arr_b, tag):
    """Byte-compare engine result arrays (T2/T5/T8/T9 form); None on
    both sides is equal (surface disabled in both arms)."""
    for name, a in arr_c.items():
        b = arr_b[name]
        if a is None or b is None:
            assert (a is None) == (b is None), \
                f"{tag}/{name}: enabled in one arm only ({a!r} vs {b!r})"
            continue
        assert_array_bytes_equal(np.asarray(a), np.asarray(b),
                                 f"{tag}/{name}")


def expect_mismatch(fn, tag):
    """Run fn(), expect OracleMismatch; return the recorded
    first-divergence payload. A negative test that fails to fire is
    itself a failure."""
    try:
        fn()
    except OracleMismatch as exc:
        payload = {"tag": tag, "kind": "OracleMismatch",
                   "detail": str(exc)}
        record_divergence(tag, payload)
        return payload
    raise AssertionError(f"negative mutation {tag} was NOT caught by "
                         f"the byte comparator")
