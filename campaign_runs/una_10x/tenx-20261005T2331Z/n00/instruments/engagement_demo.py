"""N00 evidence instrument: live engagement-counter demonstration.

Runs the native twin over a deterministic CSR case and prints, from
the module's own counters: the pre-call state, the post-call state,
and the record the dispatch produced.  This is the evidence that
engagement counters record REAL dispatches (they advance exactly by
the dispatch's own numbers) — run AFTER import, never at import.

Run (venv-n00):
  NUMBA_CACHE_DIR=<owner scratch> PYTHONDONTWRITEBYTECODE=1 \
    venv-n00/bin/python instruments/engagement_demo.py > engagement_demo.txt
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/alansynn/orca/workspaces/una-x/main/src")
sys.path.insert(0, "/Users/alansynn/orca/workspaces/una-x/main/native")

import trace_build_native as tbn


def ring_case(v=400, extra=300, origins=40, d_count=100, cutoff=8.0,
              seed=23):
    rng = np.random.default_rng(seed)
    src = np.concatenate([np.arange(v, dtype=np.int64),
                          rng.integers(0, v, extra).astype(np.int64)])
    vec = np.concatenate([((np.arange(v) + 1) % v).astype(np.int64),
                          rng.integers(0, v, extra).astype(np.int64)])
    order = np.argsort(src, kind="stable")
    src, vec = src[order], vec[order]
    ptr = np.zeros(v + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=v))
    wts = rng.uniform(0.1, 1.0, vec.shape[0])
    flag = np.ones(vec.shape[0], dtype=bool)
    oi = np.arange(origins, dtype=np.int64)
    o = np.stack([oi, (oi + 11) % v], axis=1).astype(np.int64)
    ow = np.full((origins, 2), 0.1)
    d = np.stack([rng.integers(0, v, d_count),
                  rng.integers(0, v, d_count)], axis=1).astype(np.int64)
    dw = rng.uniform(0.0, 0.5, (d_count, 2))
    dwt = rng.uniform(0.25, 1.0, d_count)
    return dict(o=o, ow=ow, ptr=ptr, vec=vec, wts=wts, flag=flag,
                d=d, dw=dw, dwt=dwt, cutoff=cutoff)


def main():
    case = ring_case()
    args = (case["o"], case["ow"], case["ptr"], case["vec"], case["wts"],
            case["flag"], case["d"], case["dw"], case["dwt"], case["cutoff"])

    print("== import-time engagement (must be the module defaults; the")
    print("   import itself engages nothing beyond loading bytes) ==")
    print(json.dumps(tbn.engagement(), indent=1, sort_keys=True))

    before = tbn.engagement()
    rec = tbn.native_build_trace(*args)
    after = tbn.engagement()

    print("== after ONE real native dispatch ==")
    print(json.dumps(after, indent=1, sort_keys=True))
    print("== record ==")
    print("route=%s complete=%s refusal=%r" %
          (rec.route, rec.complete, rec.refusal))
    print("origin_count=%d dest_count=%d" % (rec.origin_count,
                                             rec.dest_count))
    print("content_digest=%s" % rec.content_digest)
    print(json.dumps({k: getattr(rec.counters, k) for k in
                      rec.counters.__dataclass_fields__},
                     indent=1, sort_keys=True))
    print("== counter deltas vs the record's own numbers ==")
    print("dispatches_delta=%d (expect 1)" %
          (after["dispatches"] - before["dispatches"]))
    print("origins_delta=%d (expect %d)" %
          (after["origins"] - before["origins"], rec.origin_count))
    print("tiles_delta=%d (expect %d)" %
          (after["tiles"] - before["tiles"], rec.counters.tiles))
    print("candidates_delta=%d (expect %d)" %
          (after["candidates"] - before["candidates"],
           rec.counters.candidates_total))
    print("== numba route leaves engagement untouched ==")
    from urban_network_analysis.Engines import _trace_access_scratch as trm
    snap = tbn.engagement()
    trm.build_trace(*args)
    assert tbn.engagement() == snap, "numba route moved engagement"
    print("numba dispatch left every counter unchanged: OK")
    print("== build identity ==")
    info = tbn.build_info()
    print("compiler=%s" % info.get("compiler"))
    print("compile_flags=%s" % info.get("compile_flags"))
    print("engagement_module_state_final=%s" %
          json.dumps(tbn.engagement(), sort_keys=True))


if __name__ == "__main__":
    main()
