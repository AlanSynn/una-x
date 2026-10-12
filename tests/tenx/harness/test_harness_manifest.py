"""H00 manifest conformance: the recorded harness state is the real tree.

The deliverable harness_manifest.json (run root, via
UNA_10X_RUN_ROOT when present) must byte-match a fresh build; every
recorded sha256 recomputes; the law list, window classes and timer
definitions are present.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.una_10x import HARNESS_VERSION
from benchmarks.una_10x.manifest import LAWS, build_manifest
from benchmarks.una_10x.timing import PublicCallTimer
from benchmarks.una_10x.window import WINDOW_CLASSES

REPO = Path(__file__).resolve().parents[3]


def test_manifest_hashes_recompute():
    manifest = build_manifest(REPO)
    assert manifest["harness_version"] == HARNESS_VERSION
    assert manifest["task"] == "H00"
    for group in ("modules", "negative_tests", "borrowed_platform_modules"):
        assert manifest[group], group
    from benchmarks.una_10x.manifest import _sha256
    for group in ("modules", "negative_tests", "borrowed_platform_modules"):
        for rel, entry in manifest[group].items():
            sha = entry["sha256"] if isinstance(entry, dict) else entry
            assert sha == _sha256(REPO / rel), rel


def test_laws_and_regimes_present():
    manifest = build_manifest(REPO)
    ids = [law["id"] for law in manifest["laws"]]
    assert ids == [law[0] for law in LAWS] and len(ids) >= 10
    assert manifest["window_classes"] == list(WINDOW_CLASSES)
    assert set(PublicCallTimer.definitions) <= set(manifest["timer_definitions"])


def test_deliverable_matches_live_tree_when_present():
    root = os.environ.get("UNA_10X_RUN_ROOT")
    if not root:
        pytest.skip("UNA_10X_RUN_ROOT not set (deliverable written by the "
                    "H00 close-out)")
    deliverable = Path(root) / "harness_manifest.json"
    if not deliverable.is_file():
        pytest.skip(f"harness_manifest.json not yet written in {root}")
    recorded = json.loads(deliverable.read_text(encoding="utf-8"))
    assert recorded == build_manifest(REPO)
