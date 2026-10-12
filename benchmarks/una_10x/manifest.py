"""harness_manifest.json builder (H00 deliverable).

Records, mechanically from the tree:
  * sha256 of every una_10x harness module and harness negative-test file;
  * the borrowed platform modules with their sha256 and the platform
    harness version string (audit verdict: REUSE - primary measurement
    vehicle);
  * the enforced law list (each negative test cites the law it proves);
  * the public-boundary definitions.

`build_manifest` is pure: it hashes files and never mutates the tree.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from benchmarks.platform.runner import HARNESS_VERSION as PLATFORM_VERSION

from . import HARNESS_VERSION
from .timing import PublicCallTimer
from .window import WINDOW_CLASSES

_UNA_10X_MODULES = ("__init__.py", "errors.py", "timing.py", "window.py",
                    "dispatch_probe.py", "manifest.py")
_HARNESS_TESTS = ("test_timer_laws.py", "test_window_laws.py",
                  "test_bounded_queue.py", "test_harness_manifest.py",
                  "test_dispatch_probe.py")
_BORROWED = ("runner.py", "identity.py", "guards.py", "records.py",
             "errors.py", "sampler.py", "validation.py", "child.py")

LAWS = (
    ("true_api_dispatch", "a window is valid only when exactly the "
     "expected public entry (UNA.RunAccessibility/RunODM/RunFlow/RunBatch) "
     "was traced; wrong-analysis success is forbidden"),
    ("import_identity", "the imported module file must be an installed "
     "site-packages tree with recorded installed_tree_sha256; a source "
     "shadow invalidates the window"),
    ("counts_close", "completed+failed+cancelled == submitted; any failed "
     "or cancelled job invalidates the window; zero validated jobs is "
     "never a window"),
    ("byte_bounded_queues", "pending-result descriptors are byte-capped; "
     "exceeding the cap fails the window instead of growing it"),
    ("queue_ledger", "peak in-flight <= min(max_workers, queue_depth); "
     "the enforced policy must match the declared one"),
    ("memory_tree_sum", "memory peaks are simultaneous whole-process-tree "
     "sums with retained samples; per-worker maxima are inadmissible"),
    ("timer_span_law", "diagnostic spans are exclusive or explicitly "
     "nested, never summed into the primary wall; spans leaking across "
     "the primary boundary are violations, recomputed from retained "
     "records"),
    ("post_hash_wall", "comparison hashes are timed, retained, outside "
     "service timing, and start only after the primary boundary"),
    ("primary_wall_boundary", "the primary wall covers "
     "public_call_complete: plan+validate+dispatch+compute+commit+"
     "composite+exports+checkpoints; outer process/import/pool/JIT "
     "overhead is recorded separately"),
    ("artifacts_in_boundary", "every required output is listed with "
     "bytes>0 and a sha256; empty or unhashed artifacts invalidate the "
     "window"),
    ("fault_status_none", "a window with a recorded fault is never a "
     "valid measurement"),
    ("window_class_regime", "every window names one BENCHMARKS.md cache "
     "regime"),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(repo: str | os.PathLike = ".") -> dict:
    repo = Path(repo).resolve()
    base = repo / "benchmarks" / "una_10x"
    modules = {}
    for name in _UNA_10X_MODULES:
        p = base / name
        modules[f"benchmarks/una_10x/{name}"] = {
            "sha256": _sha256(p), "bytes": p.stat().st_size}
    tests = {}
    tdir = repo / "tests" / "tenx" / "harness"
    for name in _HARNESS_TESTS:
        p = tdir / name
        tests[f"tests/tenx/harness/{name}"] = {
            "sha256": _sha256(p), "bytes": p.stat().st_size}
    borrowed = {}
    pbase = repo / "benchmarks" / "platform"
    for name in _BORROWED:
        p = pbase / name
        borrowed[f"benchmarks/platform/{name}"] = _sha256(p)
    return {
        "task": "H00",
        "harness_version": HARNESS_VERSION,
        "campaign": "una_10x",
        "borrowed_platform_harness_version": PLATFORM_VERSION,
        "reuse_verdict_basis": "benchmarks/platform is the primary "
                               "measurement vehicle (h00_harness_audit "
                               "benchmark_reuse_verdicts)",
        "modules": modules,
        "negative_tests": tests,
        "borrowed_platform_modules": borrowed,
        "laws": [{"id": lid, "law": text} for lid, text in LAWS],
        "window_classes": list(WINDOW_CLASSES),
        "timer_definitions": dict(PublicCallTimer.definitions),
        "public_boundary": "primary wall = public_call_complete "
                           "(plan+validate+dispatch+compute+commit+"
                           "composite+exports+checkpoints); post-hash "
                           "wall timed separately; diagnostic spans "
                           "outside timing",
    }
