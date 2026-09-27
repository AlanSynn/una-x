"""Shared invocation/compare/evidence helpers for the F3 tests.

Mirrors harness_f1.py: single B0 singleton via the oracle package's
support.b0(), single candidate alias via cand_load.  Collects the
suite's evidence tail into evidence/F3I/test_artifacts.json.

MECHANICAL TIE (proof section 4a / section 10): every flush records
the RUNTIME environment's scipy version and the resolved dijkstra
module path next to the fixture shas, so the pinned-version identity
the proof grounds on is re-verified wherever the tests actually
execute.  Any scipy version drift is a stop-and-report alarm.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys

import numpy as np

from comparator import assert_array_bytes_equal
from support import b0 as _b0
import cand_load

EVIDENCE_DIR = ("/Users/alansynn/orca/workspaces/una-x/wt-large-e2e"
                "/campaigns/una_large_e2e/evidence/F3I")
ARTIFACT_PATH = os.path.join(EVIDENCE_DIR, "test_artifacts.json")

_ARTIFACTS = {"env": {}, "schedules": {}, "call_sequences": {},
              "fixture_shas": {}, "warnings": {}, "notes": {}}

_ENV_RECORDED = False


def b0ns():
    return _b0()


def cns():
    return cand_load.cand()


def record_schedule(key, payload):
    _ARTIFACTS["schedules"][key] = payload


def record_calls(key, payload):
    _ARTIFACTS["call_sequences"][key] = payload


def record_fixture_sha(key, payload):
    _ARTIFACTS["fixture_shas"][key] = payload


def record_warnings(key, payload):
    _ARTIFACTS["warnings"][key] = payload


def record_note(key, payload):
    _ARTIFACTS["notes"][key] = payload


def env_record():
    """The section 4a/10 mechanical tie: runtime scipy identity + the
    resolved dijkstra module path, recorded once per artifact flush."""
    global _ENV_RECORDED
    if _ENV_RECORDED:
        return
    import scipy
    import scipy.sparse.csgraph._shortest_path as _sp
    _ARTIFACTS["env"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "dijkstra_module": _sp.__name__,
        "dijkstra_module_file": _sp.__file__,
        "candidate_root": cns().root,
        "candidate_module_files": dict(cns().module_files),
    }
    _ENV_RECORDED = True


def sha256_of(obj):
    """Deterministic sha256 of a JSON-able fixture description."""
    blob = json.dumps(obj, sort_keys=True, default=lambda o: o.hex()
                      if isinstance(o, bytes) else repr(o)).encode()
    return hashlib.sha256(blob).hexdigest()


def flush_artifacts():
    """Write the collected suite tail; safe to call multiple times
    (last writer wins with merged content)."""
    env_record()
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    existing = {}
    if os.path.exists(ARTIFACT_PATH):
        with open(ARTIFACT_PATH) as fh:
            try:
                existing = json.load(fh)
            except ValueError:
                existing = {}
    for section, payload in _ARTIFACTS.items():
        if section == "env":
            existing.setdefault(section, {}).update(payload)
        else:
            existing.setdefault(section, {}).update(payload)
    tmp = ARTIFACT_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(existing, fh, indent=2, sort_keys=True)
    os.replace(tmp, ARTIFACT_PATH)


def gradient_call(mod, stub, ns):
    """Call one arm's _precompute_dest_gradients on a stub engine."""
    return mod.AggregateFlow._precompute_dest_gradients(stub, ns)


def assert_gradient_bytes(out_c, out_b, tag):
    """Byte-compare one gradient call's 4 outputs (indptr/nodes/dist/pred)."""
    for i, name in enumerate(("indptr", "nodes", "dist", "pred")):
        assert_array_bytes_equal(np.asarray(out_c[i]), np.asarray(out_b[i]),
                                 f"{tag}/{name}")
