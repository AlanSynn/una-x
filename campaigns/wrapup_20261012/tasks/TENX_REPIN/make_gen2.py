"""Generate campaigns/una_10x/identity/metric_dependencies.gen2.json.

Mechanical, not hand-fitted: every changed value is derived by the
identity suite's own probe module (tests/tenx/identity/_engine_facts.py)
run against the live engine bytes; every UNCHANGED region of the
preserved generation-1 table is carried over byte-semantically from the
preserved file and asserted equal modulo the known deltas before the
write happens.  Generation 1 (metric_dependencies.json) is NEVER opened
for writing.

Run from the repo root with the alan interpreter:

  PYTHONHASHSEED=0 python campaigns/wrapup_20261012/tasks/TENX_REPIN/make_gen2.py
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tests" / "tenx" / "identity"))
import _engine_facts as ef  # noqa: E402

GEN1 = REPO / "campaigns" / "una_10x" / "identity" / "metric_dependencies.json"
GEN2 = REPO / "campaigns" / "una_10x" / "identity" / "metric_dependencies.gen2.json"
GEN1_SHA = "sha256:" + hashlib.sha256(GEN1.read_bytes()).hexdigest()

table = json.loads(GEN1.read_text())
facts = ef.fold_facts("reach_gravity_knn_access")
bindings = ef.fold_call_bindings("reach_gravity_knn_access")

# --- assert the delta is exactly the known lazy-import shift ------------
old_sha = table["fold"]["engine_sha256"]
fold1 = dict(table["fold"]); fold2 = dict(facts)
fold1.pop("engine_sha256"); fold2.pop("engine_sha256")
assert fold1 == fold2, "fold facts changed beyond engine_sha256 — STOP"
assert sorted(map(int, bindings)) == \
    [k - 1 for k in table["call_sites"]["linenos"]], \
    "call-site shift is not the known -1 — STOP"
assert table["call_sites"]["per_site"] == \
    {str(k + 1): v for k, v in bindings.items()}, \
    "binding expressions changed — STOP"
assert len({json.dumps(v, sort_keys=True) for v in bindings.values()}) == 1, \
    "bindings not unanimous — STOP"

# --- overlay the live-derived values ------------------------------------
table["fold"] = facts
table["call_sites"] = {
    "bindings_unanimous": True,
    "linenos": sorted(int(k) for k in bindings),
    "per_site": {str(k): v for k, v in sorted(bindings.items())},
}
table["numerical_env"]["engine_sha256"] = facts["engine_sha256"]

# --- generation metadata (append last, original order preserved) --------
table["generation"] = {
    "generation": 2,
    "generated_commit": "48d2c7d9c465ca21c48279961fd18902b9f98127",
    "generated_date": "2026-10-12",
    "generator": ("tests/tenx/identity/_engine_facts.py probe logic, run "
                  "mechanically on the live engine bytes by "
                  "campaigns/wrapup_20261012/tasks/TENX_REPIN/make_gen2.py"),
    "engine_sha256": facts["engine_sha256"],
    "call_site_linenos": table["call_sites"]["linenos"],
    "lineage": (
        "generation 2 — engine evolved by the gated lazy-import change "
        "(exl-20261011T164228Z, 1.10654x; G4RC g4rc-20261012T0512Z, 3.076x); "
        "generation 1 preserved untouched as metric_dependencies.json "
        "(Z00 campaign-close record)"),
    "generation_1_sha256": GEN1_SHA,
}

GEN2.write_text(json.dumps(table, indent=2, sort_keys=True))
print("gen1 sha256:", GEN1_SHA)
print("gen2 sha256: sha256:" + hashlib.sha256(GEN2.read_bytes()).hexdigest())
print("engine sha :", facts["engine_sha256"])
print("call sites :", table["call_sites"]["linenos"])
