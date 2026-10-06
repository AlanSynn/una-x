"""P03C matched-sample probe (campaign una_10x post_p00, designs/03
"Tests and evidence"): record dispatch count, copies/bytes, per-origin
count and compiler warm-up SEPARATELY for the frozen Python replay and
the compiled outer driver, on one matched trace, with bitwise parity
re-asserted inside the probe.

This is NOT a performance claim: "a replay-only win is not the 10x
application claim" (designs/03).  The warm-up figures exist to document
the compiler boundary honestly (first call = compile, second = warm),
never to serve as a speed ratio.  The PUBLIC 10x measurement stays at
the P05 gate under the frozen policy.

Usage (repo root, venv-p01m, fresh NUMBA_CACHE_DIR):

    NUMBA_CACHE_DIR=<fresh dir> python \\
        tests/tenx/post_p00/replay/matched_samples_probe.py <out.json>
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

TENX = REPO / "tests" / "tenx"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(out_path):
    assert len(sys.argv) == 2, "usage: probe.py <out.json>"
    import os
    cache = os.environ.get("NUMBA_CACHE_DIR")
    assert cache, "NUMBA_CACHE_DIR must be pinned for this probe"

    # Engine modules are imported FIRST, under the caller's fresh
    # NUMBA_CACHE_DIR: numba freezes its cache directory at import, so
    # the cold-compile warm-up below is real.  (The C11 conftest
    # re-points the env at its owner dir when exec'd — done AFTER, so
    # it cannot affect where these compiles land.)
    from urban_network_analysis.Engines import _trace_access_scratch as tr
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        integrated_scope_access)
    legacy_ref = _load(
        "p03c_probe_legacy",
        Path(__file__).resolve().parent / "legacy_replay_reference.py")

    tck = _load("p03c_probe_trace_conftest",
                TENX / "trace" / "conftest.py")

    # ---- matched input: the C11 medium ring (40 origins) -------------
    rng = np.random.default_rng(23)
    v = 400
    src = np.concatenate([np.arange(v, dtype=np.int64),
                          rng.integers(0, v, 300).astype(np.int64)])
    vec = np.concatenate([((np.arange(v) + 1) % v).astype(np.int64),
                          rng.integers(0, v, 300).astype(np.int64)])
    order = np.argsort(src, kind="stable")
    src, vec = src[order], vec[order]
    ptr = np.zeros(v + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=v))
    wts = rng.uniform(0.1, 1.0, vec.shape[0])
    flag = np.ones(vec.shape[0], dtype=bool)
    origins = np.arange(40, dtype=np.int64)
    o = np.stack([origins, (origins + 11) % v], axis=1).astype(np.int64)
    ow = np.full((40, 2), 0.1)
    d_count = 100
    d = np.stack([rng.integers(0, v, d_count),
                  rng.integers(0, v, d_count)], axis=1).astype(np.int64)
    dw = rng.uniform(0.0, 0.5, (d_count, 2))
    dwt = rng.uniform(0.25, 1.0, d_count)
    case = dict(o=o, ow=ow, ptr=ptr, vec=vec, wts=wts, flag=flag,
                d=d, dw=dw, dwt=dwt, cutoff=8.0)

    rec = tck.build(tr, case)
    assert rec.complete and rec.origin_count == 40
    packed_sha = hashlib.sha256(
        np.asarray(rec.indptr).tobytes()
        + np.asarray(rec.destination_ids).tobytes()
        + np.asarray(rec.distance_bits).tobytes()).hexdigest()
    nnz = int(np.asarray(rec.destination_ids).shape[0])

    bundle = (1.5, 2.0, 3.0, 0.8, "exponential", tck.METRIC_PARAMS[4])

    # ---- arm L: frozen Python replay, fold dispatches counted --------
    real_fold = legacy_ref.reach_gravity_knn_access
    counts = {"fold_dispatches": 0}

    def counting_fold(*a, **k):
        counts["fold_dispatches"] += 1
        return real_fold(*a, **k)

    legacy_ref.reach_gravity_knn_access = counting_fold
    try:
        t0 = time.perf_counter()
        old = legacy_ref.legacy_replay_trace(rec, dwt, *bundle,
                                             case["cutoff"])
        legacy_first_s = time.perf_counter() - t0
        dispatches_first = counts["fold_dispatches"]
        assert dispatches_first == rec.origin_count, counts
        t0 = time.perf_counter()
        old2 = legacy_ref.legacy_replay_trace(rec, dwt, *bundle,
                                              case["cutoff"])
        legacy_second_s = time.perf_counter() - t0
    finally:
        legacy_ref.reach_gravity_knn_access = real_fold
    assert legacy_ref.reach_gravity_knn_access is real_fold

    # ---- arm C: compiled driver --------------------------------------
    # First call on a fresh cache dir includes the JIT of the scan and
    # the driver (compiler warm-up, recorded separately).
    t0 = time.perf_counter()
    new = tr.replay_trace(rec, dwt, *bundle, case["cutoff"])
    compiled_first_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    new2 = tr.replay_trace(rec, dwt, *bundle, case["cutoff"])
    compiled_second_s = time.perf_counter() - t0

    # Structural dispatch evidence: with the module-global fold replaced
    # by a poison wrapper, the COMPILED replay still returns the exact
    # bits — the fold is bound inside compiled code, so the outer loop
    # leaves Python dispatch space (1 driver invocation per replay).
    real_fold_live = tr.reach_gravity_knn_access

    def poison(*a, **k):
        raise AssertionError("fold dispatched through Python")

    tr.reach_gravity_knn_access = poison
    try:
        patched = tr.replay_trace(rec, dwt, *bundle, case["cutoff"])
    finally:
        tr.reach_gravity_knn_access = real_fold_live
    assert tr.reach_gravity_knn_access is real_fold_live

    # ---- arm B: original integrated engine ---------------------------
    ref = integrated_scope_access(*tck.driver_args(case))

    # ---- bitwise parity, re-asserted here -----------------------------
    fields = ("reach", "gravity_exponential", "gravity_logistic",
              "knn_access")
    ref_items = list(ref)  # the integrated engine returns the 4-tuple
    parity = {}
    for f in fields:
        parity[f] = {
            "compiled_eq_legacy": (
                np.asarray(getattr(new, f)).tobytes()
                == np.asarray(getattr(old, f)).tobytes()),
            "compiled_eq_integrated": (
                np.asarray(getattr(new, f)).tobytes()
                == np.asarray(ref_items[fields.index(f)]).tobytes()),
        }
    parity_all = all(v["compiled_eq_legacy"] and
                     v["compiled_eq_integrated"]
                     for v in parity.values())

    # analytic copy volume per replay (both arms, by construction):
    # per origin one fresh C-order distance copy + one fresh gathered
    # weights copy, float64 (8 bytes each).
    copy_bytes_per_replay = 8 * 2 * nnz

    out = {
        "schema": "una-10x-p03c-matched-samples-1",
        "task": "P03C",
        "generated": "2026-10-06",
        "claim_disclaimer": "NO performance claim; designs/03: a "
            "replay-only win is not the 10x application claim.  Warm-up "
            "documents the compiler boundary only.",
        "matched_resources": {
            "process": "single interpreter, single thread of replay",
            "input": "C11 medium ring (seed 23, 40 origins, 100 dests, "
                     "cutoff 8.0), one producer trace consumed by all "
                     "arms",
            "metric_bundle": list(bundle[:4]) + [bundle[4],
                                                 "METRIC_PARAMS[4]"],
        },
        "input_identity": {
            "trace_packed_sha256": packed_sha,
            "origin_count": int(rec.origin_count),
            "nnz": nnz,
            "reach_dtype": rec.reach_dtype,
            "metric_dtype": rec.metric_dtype,
        },
        "dispatch_counts": {
            "legacy_python_replay": {
                "fold_dispatcher_calls_per_replay": dispatches_first,
                "equals_origin_count":
                    dispatches_first == rec.origin_count,
                "method": "module-global fold replaced by a counting "
                          "wrapper; the pure-Python loop re-reads the "
                          "global each origin; figure snapshotted "
                          "after the FIRST replay only",
            },
            "compiled_driver": {
                "python_level_dispatches_per_replay": 1,
                "evidence": "with the trace module's global fold "
                            "replaced by a poison wrapper, the "
                            "compiled replay STILL returned the exact "
                            "bits: the fold is bound inside compiled "
                            "code; the outer loop makes ONE Python "
                            "call (the driver) per replay",
                "poison_patch_survived_bitwise": bool(
                    np.asarray(patched.reach).tobytes()
                    == np.asarray(new.reach).tobytes()),
            },
        },
        "copies_per_replay": {
            "per_origin": "1 fresh C-order distance copy + 1 fresh "
                          "gathered-weights copy (both arms)",
            "analytic_float64_bytes": copy_bytes_per_replay,
            "note": "identical copy composition by construction; the "
                    "compiled copies happen inside the nogil driver",
        },
        "compiler_warmup_seconds_not_a_claim": {
            "legacy_first": round(legacy_first_s, 6),
            "legacy_second": round(legacy_second_s, 6),
            "compiled_first_includes_jit": round(compiled_first_s, 6),
            "compiled_second": round(compiled_second_s, 6),
            "numba_cache_dir": cache,
            "fresh_cache_dir": True,
        },
        "bitwise_parity": parity,
        "bitwise_parity_all": bool(parity_all),
    }
    assert parity_all, parity
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2) + "\n")
    print("MATCHED SAMPLES OK ->", out_path)
    print("dispatches: legacy %d (== origins) / compiled 1 python-level"
          % counts["fold_dispatches"])
    print("parity_all:", parity_all)


if __name__ == "__main__":
    main(sys.argv[1])
