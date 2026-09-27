"""F2 test-artifact recorder.

Collects a per-suite evidence tail (case labels, first-divergence
bookkeeping, environment pins) and flushes it to
campaigns/una_large_e2e/evidence/F2I/test_artifacts.json (override with
UNA_F2_ARTIFACTS). Mirrors the F1/F3 harness convention. Written even
on failures via the conftest sessionfinish hook.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os

WT = "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e"
DEFAULT_ARTIFACTS = (f"{WT}/campaigns/una_large_e2e/evidence/F2I/"
                     "test_artifacts.json")

ARTIFACTS = {
    "task": "F2I",
    "suite": "tests/large_e2e/F2",
    "opened_utc": datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "records": [],
}

_first_divergence = None


def record(label, **fields):
    entry = {"label": label, **fields}
    ARTIFACTS["records"].append(entry)


def record_first_divergence(label, detail):
    """Record the FIRST input on which the routes diverge. The suite
    rejects the domain on any divergence, so a call here is fatal to the
    battery — it is recorded (append-only) before the assert fires so
    the evidence tail carries the diverging case even on failure."""
    global _first_divergence
    if _first_divergence is None:
        _first_divergence = {"label": label, "detail": detail}
        ARTIFACTS["first_divergence"] = _first_divergence


def source_sha():
    p = f"{WT}/src/urban_network_analysis/Engines/AggregateFlow.py"
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def flush_artifacts():
    ARTIFACTS["aggregate_flow_sha256_at_flush"] = source_sha()
    ARTIFACTS["flushed_utc"] = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = os.environ.get("UNA_F2_ARTIFACTS", DEFAULT_ARTIFACTS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ARTIFACTS, f, indent=1)
    return path
