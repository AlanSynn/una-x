"""Per-arm compiled-B0 battery runner for the H03 oracles.

Executed as a SUBPROCESS with exactly one import root bound (via env
UNA_ORACLE_ARM_A / UNA_ORACLE_ARM_B or PYTHONPATH), it builds the L0
corpus, runs the real compiled B0 kernels and engines, and records:

* one NPZ with every input and output array (stable key naming),
* one JSON with dtypes/shapes, refusal-probe exception TYPES, warning
  counts, input-immutability hashes, aliasing checks, the compiler
  profile and the in-process double-run determinism verdict.

The battery is run TWICE in-process; any non-bit-identical array fails
the runner. Cross-process determinism is established by comparing the
NPZ keys/content of two runner invocations (see
benchmarks/large_e2e/oracle/compare_arms.py with both roots = B0).

Usage:
  python runner_arm.py --out-npz /p/arm.npz --out-json /p/arm.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ORACLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ORACLE_DIR))

import conftest  # noqa: F401,E402 - pins env BEFORE numba/B0 imports
import fixtures  # noqa: E402
import stub_topology  # noqa: E402
from b0_import import arm_roots, load_arm  # noqa: E402
from compiler_profile import capture_profile  # noqa: E402


def _hash(a) -> str:
    arr = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(arr.dtype.str.encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def run_battery(b0, store, meta):
    """Run the compiled corpus once, filling `store` and `meta`."""
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

    # ------------------------------------------------------------------
    # Engine battery (accessibility)
    # ------------------------------------------------------------------
    for case in fixtures.engine_cases():
        name = case.name
        topo = stub_topology.StubTopology(case)
        eng = b0.Accessibility(topo)
        ge = eng.graph_engine

        # Replicated row order vs live engine arrays (construction path
        # may be the ordered CSR fast path; row order must still match).
        replica = case.kernel_arrays()
        live = dict(
            adjacency_pointer=ge.adjacency_pointer,
            adjacency_vector=ge.adjacency_vector,
            adjacency_vector_weights=ge.adjacency_vector_weights,
            adjacynct_vector_network_node=ge.adjacynct_vector_network_node,
        )
        for key in replica:
            store[f"{name}/engine/{key}"] = np.asarray(live[key])
            store[f"{name}/replica/{key}"] = np.asarray(replica[key])
            meta["checks"].setdefault("engine_vs_replica_row_order", {})[
                f"{name}/{key}"] = (_hash(live[key]) == _hash(replica[key]))

        term = case.terminal_arrays()
        for key, arr in term.items():
            store[f"{name}/in/{key}"] = np.asarray(arr)
        store[f"{name}/in/adjacency_pointer"] = np.asarray(live["adjacency_pointer"])
        store[f"{name}/in/adjacency_vector"] = np.asarray(live["adjacency_vector"])
        store[f"{name}/in/adjacency_vector_weights"] = np.asarray(
            live["adjacency_vector_weights"])
        store[f"{name}/in/adjacynct_vector_network_node"] = np.asarray(
            live["adjacynct_vector_network_node"])
        store[f"{name}/in/cutoff"] = np.array(case.cutoff, dtype=np.float64)
        store[f"{name}/in/knn_weights"] = np.asarray(case.knn_weights)

        ptr = np.asarray(live["adjacency_pointer"])
        nbr = np.asarray(live["adjacency_vector"])
        wgt = np.asarray(live["adjacency_vector_weights"])
        flg = np.asarray(live["adjacynct_vector_network_node"])

        immut_before = {k: _hash(v) for k, v in live.items()}
        meta["checks"].setdefault("input_immutability", {})[name] = True

        n_origins = term["o_terminal_idxs"].shape[0]
        for oi in range(n_origins):
            o_idx = term["o_terminal_idxs"][oi]
            o_w = term["o_terminal_weights"][oi]
            scope, pred = b0.compact_vector_node_view_scope(
                o_idx, o_w, ptr, nbr, wgt, flg, case.cutoff,
                term["d_count"])
            store[f"{name}/scope/{oi}"] = np.asarray(scope)
            store[f"{name}/scope_pred/{oi}"] = np.asarray(pred)

            d_dist = b0.adjust_destination_distances(
                scope, term["d_terminal_idxs"], term["d_terminal_weights"],
                term["d_count"])
            store[f"{name}/adjust/{oi}"] = np.asarray(d_dist)

            filt = np.where(d_dist <= case.cutoff)[0]
            store[f"{name}/retain/idx/{oi}"] = filt.astype(np.int64)
            store[f"{name}/retain/dist/{oi}"] = np.asarray(
                d_dist[filt], dtype=np.float64)
            store[f"{name}/retain/w/{oi}"] = np.asarray(
                term["d_weights"][filt], dtype=np.float64)

            # ownership/aliasing: scope must not alias any input
            aliased = any(np.shares_memory(scope, a) for a in (ptr, nbr, wgt, flg, o_idx, o_w))
            meta["checks"].setdefault("scope_output_ownership", {})[
                f"{name}/{oi}"] = bool(aliased)

        integrated = b0.integrated_scope_access(
            term["o_terminal_idxs"], term["o_terminal_weights"],
            ptr, nbr, wgt, flg,
            term["d_terminal_idxs"], term["d_terminal_weights"],
            term["d_weights"],
            case.gravity_beta, case.metric_plateau,
            case.metric_midpoint,
            float(np.log(99.0)) / case.metric_midpoint,
            case.knn_decay, np.ascontiguousarray(case.knn_weights),
            case.cutoff,
        )
        for key, arr in zip(
                ("reach", "gravity_exponential", "gravity_logistic", "knn_access"),
                integrated):
            store[f"{name}/integrated/{key}"] = np.asarray(arr)

        od = b0.od_compact_vector_node_view_scope(
            term["o_terminal_idxs"], term["o_terminal_weights"],
            ptr, nbr, wgt, flg, case.cutoff, term["d_count"],
            term["d_terminal_idxs"], term["d_terminal_weights"])
        store[f"{name}/od_kernel"] = np.asarray(od)

        # Public engine path (chronology-relevant outputs).
        settings = stub_topology.make_settings(
            b0.Settings, accessibility=True,
            search_radius=case.cutoff,
            knn_decay=case.knn_decay,
            knn_weights=tuple(float(x) for x in case.knn_weights),
            gravity_beta=case.gravity_beta,
            gravity_plateau=case.metric_plateau,
            gravity_logistic_midpoint=case.metric_midpoint,
            gravity_decay_constant=float(np.log(99.0)),
        )
        eng.Centrality(settings)
        for key in ("reach", "gravity_exponential", "gravity_logistic", "knn_access"):
            value = getattr(eng, key, None)
            store[f"{name}/centrality/{key}"] = (
                np.asarray(value) if value is not None else np.zeros(0, np.float64))
        od_public = eng.OD_Matrix(search_radius=case.cutoff)
        store[f"{name}/od_public"] = np.asarray(od_public)

        if any(_hash(v) != immut_before[k] for k, v in live.items()):
            meta["checks"]["input_immutability"][name] = False

    # ------------------------------------------------------------------
    # Kernel-only scope battery
    # ------------------------------------------------------------------
    for kc in fixtures.kernel_scope_cases():
        scope, pred = b0.compact_vector_node_view_scope(
            kc.o_terminal_idxs[0], kc.o_terminal_weights[0],
            kc.adjacency_pointer, kc.adjacency_vector,
            kc.adjacency_vector_weights, kc.adjacynct_vector_network_node,
            kc.cutoff, kc.d_count)
        store[f"k/{kc.name}/scope"] = np.asarray(scope)
        store[f"k/{kc.name}/pred"] = np.asarray(pred)
        store[f"k/{kc.name}/in/pointer"] = kc.adjacency_pointer
        store[f"k/{kc.name}/in/nbrs"] = kc.adjacency_vector
        store[f"k/{kc.name}/in/weights"] = kc.adjacency_vector_weights
        store[f"k/{kc.name}/in/flags"] = kc.adjacynct_vector_network_node
        store[f"k/{kc.name}/in/o_idx"] = kc.o_terminal_idxs
        store[f"k/{kc.name}/in/o_w"] = kc.o_terminal_weights
        store[f"k/{kc.name}/in/cutoff"] = np.array(kc.cutoff, dtype=np.float64)

    # ------------------------------------------------------------------
    # Flow battery: fixed-K engine runs + compiled gradient assembly
    # ------------------------------------------------------------------
    spec = fixtures.flow_network_spec()
    settings_flow = stub_topology.make_settings(
        b0.Settings, accessibility=False,
        **fixtures.flow_settings_overrides(node_flow=True))
    settings_flow_nonode = stub_topology.make_settings(
        b0.Settings, accessibility=False,
        **fixtures.flow_settings_overrides(node_flow=False))

    eng = b0.AggregateFlow(stub_topology.StubFlowTopology(spec))
    ns = eng._prepare_params(settings_flow)
    meta["flow_params"] = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                           for k, v in ns.items()}
    eng.Centrality(settings_flow)  # builds CSR; K=default first (not stored)

    csr = dict(
        indptr=eng._csr_indptr, indices=eng._csr_indices,
        weights=eng._csr_weights, edge_id=eng._csr_edge_id,
        direction=eng._csr_direction,
        csr_fwd_data=eng._csr_fwd.data, csr_fwd_indices=eng._csr_fwd.indices,
        csr_fwd_indptr=eng._csr_fwd.indptr,
        csr_rev_data=eng._csr_rev.data, csr_rev_indices=eng._csr_rev.indices,
        csr_rev_indptr=eng._csr_rev.indptr,
    )
    for key, arr in csr.items():
        store[f"flow/csr/{key}"] = np.asarray(arr)
    store["flow/meta/n_net"] = np.array(eng._n_network_nodes, dtype=np.int64)
    store["flow/meta/n_dest"] = np.array(eng._n_destinations, dtype=np.int64)
    store["flow/meta/n_origins"] = np.array(eng._n_origins, dtype=np.int64)
    store["flow/meta/first_origin_node"] = np.array(
        eng._first_origin_node, dtype=np.int64)

    grads = eng._precompute_dest_gradients(ns)
    store["flow/grads/indptr"] = np.asarray(grads[0])
    store["flow/grads/nodes"] = np.asarray(grads[1])
    store["flow/grads/dist"] = np.asarray(grads[2])
    store["flow/grads/pred"] = np.asarray(grads[3])

    for K in (1, 2, 3):
        eng.Centrality(settings_flow)
        eng.num_threads = K
        eng.Centrality(settings_flow)
        store[f"flow/K{K}/AB"] = np.asarray(eng.edge_flow_AB)
        store[f"flow/K{K}/BA"] = np.asarray(eng.edge_flow_BA)
        store[f"flow/K{K}/node"] = np.asarray(eng.node_flow)
        store[f"flow/K{K}/undirected"] = np.asarray(eng.edge_flow)

    eng.Centrality(settings_flow_nonode)
    eng.num_threads = 2
    eng.Centrality(settings_flow_nonode)
    store["flow/K2_nonode/AB"] = np.asarray(eng.edge_flow_AB)
    store["flow/K2_nonode/BA"] = np.asarray(eng.edge_flow_BA)
    store["flow/K2_nonode/node"] = np.asarray(
        eng.node_flow if eng.node_flow is not None else np.zeros(0, np.float64))

    # Gradient equivalence across chunk sizes on the SAME compiled CSR
    # (F3: numerical behavior must be independent of the chunk choice).
    n_dest = eng._n_destinations
    n_total = eng._csr_indptr.shape[0] - 1
    dest_nodes = np.arange(eng._n_network_nodes,
                           eng._n_network_nodes + n_dest, dtype=np.int64)
    limit = float(b0._cutoff_for_shortest(
        float(ns["search_radius"]), ns["mode"], float(ns["ratio"]),
        float(ns["buffer"])))
    for chunk in (1, 2, 3, max(1, int(1e8 // max(n_total, 1)))):
        indptr = np.zeros(n_dest + 1, dtype=np.int64)
        idx_parts, dist_parts, pred_parts = [], [], []
        counts = np.zeros(n_dest, dtype=np.int64)
        n_chunks = 0
        for s in range(0, n_dest, chunk):
            e = min(s + chunk, n_dest)
            dist, preds = scipy_dijkstra(
                eng._csr_rev, directed=True, indices=dest_nodes[s:e],
                limit=limit, return_predecessors=True)
            if dist.ndim == 1:
                dist, preds = dist[None, :], preds[None, :]
            finite = np.isfinite(dist)
            n_chunks += 1
            for k in range(e - s):
                cols = np.where(finite[k])[0]
                counts[s + k] = cols.shape[0]
                idx_parts.append(cols.astype(np.int64))
                dist_parts.append(dist[k, cols].astype(np.float64))
                pred_parts.append(preds[k, cols].astype(np.int32))
        np.cumsum(counts, out=indptr[1:])
        store[f"flow/grads_chunk{chunk}/indptr"] = indptr
        store[f"flow/grads_chunk{chunk}/nodes"] = (
            np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int64))
        store[f"flow/grads_chunk{chunk}/dist"] = (
            np.concatenate(dist_parts) if dist_parts else np.zeros(0, np.float64))
        store[f"flow/grads_chunk{chunk}/pred"] = (
            np.concatenate(pred_parts) if pred_parts else np.zeros(0, np.int32))
        meta["flow_grad_chunks"] = meta.get("flow_grad_chunks", {})
        meta["flow_grad_chunks"][str(chunk)] = {"n_chunks": n_chunks}

    # ------------------------------------------------------------------
    # Refusal probes (typing-level; NO new specializations compiled)
    # ------------------------------------------------------------------
    probes = {}

    def probe(label, fn, expected_type):
        try:
            fn()
            probes[label] = {"raised": False, "exception_type": None,
                             "expected_type": expected_type.__name__}
        except Exception as exc:  # noqa: BLE001 - probe records type only
            probes[label] = {"raised": True,
                             "exception_type": type(exc).__name__,
                             "expected_type": expected_type.__name__}

    kc0 = fixtures.kernel_scope_cases()[0]
    probe("scope_cutoff_str",
          lambda: b0.compact_vector_node_view_scope(
              kc0.o_terminal_idxs[0], kc0.o_terminal_weights[0],
              kc0.adjacency_pointer, kc0.adjacency_vector,
              kc0.adjacency_vector_weights,
              kc0.adjacynct_vector_network_node,
              "10.0", kc0.d_count),
          Exception)
    probe("scope_object_weights",
          lambda: b0.compact_vector_node_view_scope(
              kc0.o_terminal_idxs[0], kc0.o_terminal_weights[0],
              kc0.adjacency_pointer, kc0.adjacency_vector,
              kc0.adjacency_vector_weights.astype(object),
              kc0.adjacynct_vector_network_node,
              kc0.cutoff, kc0.d_count),
          Exception)
    probe("scope_cutoff_none",
          lambda: b0.compact_vector_node_view_scope(
              kc0.o_terminal_idxs[0], kc0.o_terminal_weights[0],
              kc0.adjacency_pointer, kc0.adjacency_vector,
              kc0.adjacency_vector_weights,
              kc0.adjacynct_vector_network_node,
              None, kc0.d_count),
          Exception)

    case0 = fixtures.engine_cases()[0]
    topo_bad = stub_topology.StubTopology(case0)
    topo_bad.origins.weight_to_start = np.zeros(
        len(case0.o_start) + 1, dtype=np.float64)
    probe("engine_mismatched_origin_weights",
          lambda: b0.Accessibility(topo_bad), Exception)
    meta["refusal_probes"] = probes

    # Kernel signature inventory (all specializations observed here).
    meta["signatures"] = {
        kernel: [str(sig) for sig in getattr(b0, kernel).signatures]
        for kernel in ("compact_vector_node_view_scope",
                       "adjust_destination_distances",
                       "reach_gravity_knn_access",
                       "od_compact_vector_node_view_scope",
                       "integrated_scope_access",
                       "_accumulate_od_flow", "_decay")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--runs", type=int, default=2,
                    help="in-process battery repetitions (>=2 asserts "
                         "bit-identical self-determinism)")
    args = ap.parse_args()

    b0 = load_arm(arm_roots()[0])
    meta = {
        "import_root": b0.root,
        "module_files": dict(b0.module_files),
        "numba_cache_dir": __import__("os").environ.get("NUMBA_CACHE_DIR"),
        "numba_num_threads": __import__("os").environ.get("NUMBA_NUM_THREADS"),
        "l1_reuse_dir": __import__("os").environ.get("L1_REUSE_DIR"),
        "checks": {},
        "warnings": [],
    }

    store_first = {}
    run_results = []
    for run_index in range(max(2, args.runs)):
        store = {}
        meta["checks"] = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_battery(b0, store, meta)
        meta["warnings"] = [
            f"{w.category.__name__}: {w.message}" for w in caught]
        if run_index == 0:
            store_first = store
        identical, first_bad = True, None
        if run_index > 0:
            if set(store.keys()) != set(store_first.keys()):
                identical = False
                first_bad = "key set differs"
            else:
                for key in store_first:
                    a, b = store_first[key], store[key]
                    if (np.asarray(a).dtype != np.asarray(b).dtype
                            or np.asarray(a).shape != np.asarray(b).shape
                            or np.ascontiguousarray(np.asarray(a)).tobytes()
                            != np.ascontiguousarray(np.asarray(b)).tobytes()):
                        identical = False
                        first_bad = key
                        break
        run_results.append({"run": run_index, "n_keys": len(store),
                            "identical_to_first": identical,
                            "first_mismatch": first_bad})

    meta["self_determinism"] = {
        "runs": run_results,
        "all_identical": all(r["identical_to_first"] for r in run_results),
    }
    meta["profile"] = capture_profile(
        b0, kernels_parallel_warmed=True)
    meta["fixture_seed"] = fixtures.FIXTURE_SEED

    np.savez(args.out_npz, **store_first)
    with open(args.out_json, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)

    ok = meta["self_determinism"]["all_identical"]
    bad_construction = [k for k, v in
                        meta["checks"].get("engine_vs_replica_row_order", {}).items()
                        if not v]
    bad_immut = [k for k, v in
                 meta["checks"].get("input_immutability", {}).items() if not v]
    bad_alias = [k for k, v in
                 meta["checks"].get("scope_output_ownership", {}).items() if v]
    if bad_construction or bad_immut or bad_alias or not ok:
        print(f"RUNNER FAILED: determinism={ok} "
              f"row_order_mismatch={bad_construction} "
              f"immutable_violations={bad_immut} "
              f"aliased_outputs={bad_alias}")
        return 1
    print(f"RUNNER OK: {len(store_first)} arrays, "
          f"{len(run_results)} identical in-process runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
