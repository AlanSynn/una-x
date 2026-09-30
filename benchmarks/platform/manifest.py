"""Frozen workload manifests (HARNESS.md "One real job"; BENCHMARKS.md
"Freeze workload content ... before candidate measurements").

A manifest pins: profile/model, requested backend, workload class, input
fingerprints, settings patch, output obligations and the resource policy.
`load_manifest` recomputes every input fingerprint — a drifted workload is a
ManifestError *before any timing* (negative control 5).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .errors import ManifestError

MANIFEST_VERSION = 1

_REQUIRED_KEYS = ("manifest_version", "campaign", "profile", "model",
                  "backend_requested", "workload", "settings_patch",
                  "output_obligations", "resource_policy")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_workload(workload: dict, base_dir: str | None = None) -> dict:
    """Recompute sha256 of every declared input; return {path: digest}."""
    digests = {}
    for name, rel in workload.get("inputs", {}).items():
        p = Path(rel)
        if not p.is_absolute() and base_dir:
            p = Path(base_dir) / p
        if not p.is_file():
            raise ManifestError(f"workload input {name} missing: {p}")
        digests[name] = {"path": str(p), "sha256": sha256_file(p)}
    return digests


def load_manifest(path: str | Path, base_dir: str | None = None,
                  verify_inputs: bool = True) -> dict:
    """Load and validate a frozen manifest.  With verify_inputs, recompute
    all workload fingerprints and compare with the frozen ones."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ManifestError(f"manifest missing keys: {missing}")
    if raw["manifest_version"] != MANIFEST_VERSION:
        raise ManifestError(
            f"manifest_version {raw['manifest_version']} unsupported "
            f"(expected {MANIFEST_VERSION})")
    model = raw["model"]
    for key in ("analysis", "engine"):
        if key not in model:
            raise ManifestError(f"model.{key} not pinned in manifest")
    if raw["backend_requested"] not in ("auto", "reference", "native", "gpu"):
        raise ManifestError(
            f"backend_requested {raw['backend_requested']!r} invalid")
    if verify_inputs:
        frozen = raw["workload"].get("input_hashes") or {}
        actual = fingerprint_workload(raw["workload"], base_dir)
        if set(frozen) != set(actual):
            raise ManifestError(
                f"workload fingerprint set drifted: frozen={sorted(frozen)} "
                f"actual={sorted(actual)}")
        for name, rec in actual.items():
            if frozen[name]["sha256"] != rec["sha256"]:
                raise ManifestError(
                    f"workload input {name} changed since freeze "
                    f"({frozen[name]['sha256'][:12]} -> "
                    f"{rec['sha256'][:12]}); refusing to time drifted input")
        raw["_verified_input_hashes"] = actual
    return raw


def freeze_manifest(out_path: str | Path, *, campaign: str, profile: str,
                    model: dict, backend_requested: str, workload: dict,
                    settings_patch: dict, output_obligations: dict,
                    resource_policy: dict, expected_identity: dict | None,
                    base_dir: str | None = None, extra: dict | None = None):
    """Write a manifest with freshly computed input fingerprints."""
    doc = {
        "manifest_version": MANIFEST_VERSION,
        "campaign": campaign,
        "profile": profile,
        "model": dict(model),
        "backend_requested": backend_requested,
        "workload": dict(workload),
        "settings_patch": dict(settings_patch),
        "output_obligations": output_obligations,
        "resource_policy": resource_policy,
        "expected_identity": expected_identity or {},
    }
    doc["workload"]["input_hashes"] = fingerprint_workload(workload, base_dir)
    if extra:
        doc.update(extra)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    return doc


def resource_policy_of(manifest: dict) -> dict:
    """Normalized declared policy used by the runner and ledger."""
    pol = manifest.get("resource_policy") or {}
    return {
        "max_workers": int(pol.get("max_workers", 1)),
        "queue_depth": int(pol.get("queue_depth", 2)),
        "memory_limit_bytes": pol.get("memory_limit_bytes"),
        "threads_per_worker": int(pol.get("threads_per_worker", 1)),
    }
