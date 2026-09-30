"""Fail-closed structural check on completion claims. Independent review is still required."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path


def check(record: dict, campaign: dict, root: Path) -> list[str]:
    errors = []
    state = record.get("state")
    if state not in {"qualified", "merge_ready", "partial", "blocked_external", "no_safe_merge", "no_performance_gain"}:
        errors.append("unknown terminal state")
    strict = state in {"qualified", "merge_ready"}
    if not strict:
        return errors
    if not re.fullmatch(r"[0-9a-f]{40}", record.get("source_sha") or ""):
        errors.append("qualified record needs full source SHA")
    if not record.get("api_census_complete") or record.get("unimplemented_supported_apis"):
        errors.append("API census incomplete or supported API missing")
    if record.get("bitwise_failures"):
        errors.append("bitwise mismatches cannot qualify")
    for name in campaign["requirements"]:
        feature = record.get("features", {}).get(name, {})
        if feature.get("status") != "passed" or not isinstance(feature.get("tests"), int) or feature.get("tests", 0) < 1:
            errors.append(f"mandatory feature not tested/passed: {name}")
        entries = feature.get("evidence", [])
        if not entries:
            errors.append(f"missing evidence: {name}")
        for entry in entries:
            if not isinstance(entry, dict) or not {"path", "sha256"}.issubset(entry):
                errors.append(f"malformed evidence: {name}")
                continue
            path = (root / entry["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                errors.append(f"missing/unsafe evidence: {name}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                errors.append(f"evidence hash mismatch: {name}")
    parity = record.get("features", {}).get("bitwise_parity", {})
    if parity.get("compared_arrays", 0) < 1 or parity.get("compared_bytes", 0) < 1:
        errors.append("non-vacuous bitwise parity evidence required")
    if not set(campaign.get("semantic_profiles", [])).issubset(set(parity.get("profiles_tested", []))):
        errors.append("declared numerical profiles not covered")
    for name in ["native_backend", "gpu_backend"]:
        f = record.get("features", {}).get(name, {})
        if f.get("substantive_kernel_calls", 0) < 1 or f.get("fallback_only", True):
            errors.append(f"no substantive backend engagement: {name}")
    gpu = record.get("features", {}).get("gpu_backend", {})
    if not gpu.get("real_hardware") or gpu.get("simulator", True):
        errors.append("GPU qualification needs actual hardware, not simulation")
    if record.get("review") != "passed":
        errors.append("independent review not passed")
    if record.get("ci") != "passed":
        errors.append("required CI not passed")
    if state == "merge_ready" and not record.get("merge_rehearsal_passed"):
        errors.append("merge-ready without rehearsal")
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("record", type=Path)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    errors = check(json.loads(a.record.read_text()), json.loads((a.root / "campaign.json").read_text()), a.root)
    print(json.dumps({"valid_claim_structure": not errors, "errors": errors,
        "note": "Structure/hashes only; not an independent audit of evidence truth."}, indent=2))
    raise SystemExit(1 if errors else 0)

if __name__ == "__main__":
    main()
