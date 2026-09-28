"""W00 installed engagement probe (run under each ARM venv's python).

Proves, against the INSTALLED wheel in this interpreter (never a source
tree): identity guards hold in-process; the composed dispatch routes
actually engage (A1 scratch + A3 tail + F2 local overlap + F3 chunking)
on the selected arm; the B0 arm shows the original surfaces (no dispatch
symbols, no _f2_stats) and byte-identical kernel outputs.  Cross-arm
byte equality is adjudicated by the W00 supervisor from the emitted
hash lists, not here.

Fixture stack: reuses the composition suite's builders (A-side guard
matrix, F-side flow specs, F3 gradient stub) plus the oracle
flow/stub-topology modules — all numpy/scipy-only, no package imports,
no B0 source loading.  The package under test resolves ONLY from this
venv's site-packages (PYTHONPATH must be absent; the supervisor launches
the probe with a clean environment).

CLI:
    <arm-python> engagement_probe.py --arm <label> --kind selected|b0
        --identity <identity.json> --out <result.json>

Exit 0 iff every expectation for this arm kind held; rc 2 otherwise
(failed evidence is retained, never retried in place).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path("/Users/alansynn/orca/workspaces/una-x/wt-large-e2e")

# numpy-only fixture stack (oracle + composition builders)
sys.path.insert(0, str(REPO / "tests/large_e2e/oracle"))
sys.path.insert(0, str(REPO / "tests/large_e2e/composition"))
# harness package (identity guards) — imported BEFORE the engine package
sys.path.insert(0, str(REPO / "benchmarks/large_e2e"))

import numpy as np  # noqa: E402
import numba as nb  # noqa: E402

RESULT: dict = {"records": []}


def rec(label, **fields):
    RESULT["records"].append({"label": label, **fields})


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def fail(msg):
    RESULT["failure"] = msg
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(RESULT, fh, indent=1, default=str)
    print(f"[w00-probe] FAIL: {msg}")
    sys.exit(2)


# ======================================================================
# Stage 1: identity guards, in-process (HARNESS.md behavior 2)
# ======================================================================
def identity_leg(idmod, identity_path: Path):
    if os.environ.get("PYTHONPATH"):
        fail(f"probe launched with PYTHONPATH={os.environ['PYTHONPATH']!r}")
    data = json.loads(identity_path.read_text())
    identity = idmod.Identity(data, identity_path)

    package = __import__("urban_network_analysis")
    root = Path(package.__file__).resolve().parent
    sp = idmod.check_under_site_packages(root, identity.site_packages)
    pth = idmod.scan_pth_files(identity.site_packages)
    editable = idmod.scan_editable_install(identity.site_packages)
    dist = idmod.require_dist_info(identity.site_packages)
    tree = idmod.verify_package_tree(root, identity.module_hashes)
    rec("identity_guards", arm=identity.arm, kind=identity.kind,
        package_root=str(root), site_packages=str(sp),
        dist_info=dist, pth_findings=pth, editable_findings=editable,
        verified_files=tree["verified_files"],
        tree_sha256=tree["tree_sha256"],
        identity_file_sha256=hashlib.sha256(
            identity_path.read_bytes()).hexdigest())
    if pth or editable:
        fail("identity guard findings present")
    if tree["tree_sha256"] != identity.package_tree_sha256:
        fail("package tree digest != identity digest")
    return identity, idmod.verify_imported_modules(
        dict(sys.modules), identity)


# ======================================================================
# Stage 2: A-side — dispatch guards + cross-arm kernel bytes
# ======================================================================
def a_side_leg(kind, arm):
    import fixtures_composition as fc
    # C1 (W00 resume-fire adjudication): the package __init__ re-exports
    # only UNA/Settings/Topology — the Engines surfaces are NOT namespace
    # attributes.  Bind every surface explicitly from its own module as a
    # LOCAL name; never attribute-probe the package namespace.
    from urban_network_analysis.Engines.Accessibility import (
        integrated_scope_access,
        od_compact_vector_node_view_scope,
    )
    try:
        from urban_network_analysis.Engines import \
            _large_access_scratch as scratch
        scratch_error = None
    except ImportError as exc:  # b0 negative: this module must be absent
        scratch = None
        scratch_error = f"{type(exc).__name__}: {exc}"
    rec("scratch_import", present=scratch is not None, error=scratch_error)

    if kind == "selected":
        if scratch is None:
            fail(f"selected arm lacks the A1/A3 dispatch module: "
                 f"{scratch_error}")

        # closure-captured guards: the @overload guards resolve ONLY under
        # nopython, and attribute calls are resolved at compile time —
        # mirror the proven composition-suite probe idiom exactly
        a1_admits = scratch._a1_scope_admits
        a3_admits = scratch._a3_tail_admits

        @nb.njit(cache=False)
        def probe(adjacency_pointer, adjacency_vector,
                  adjacency_vector_weights, adjacynct_vector_network_node,
                  o_terminal_idxs, o_terminal_weights, cutoff,
                  d_terminal_idxs, d_count, node_count):
            a1_ok, a1_max = a1_admits(
                adjacency_pointer, adjacency_vector,
                adjacency_vector_weights, adjacynct_vector_network_node,
                o_terminal_idxs, o_terminal_weights, cutoff)
            a3_ok = a3_admits(d_terminal_idxs, d_count, node_count)
            return a1_ok, a1_max, a3_ok

        guard_rows = {}
        for variant in fc.VARIANTS:
            arr = getattr(fc, variant)()
            a1_ok, a1_max, a3_ok = probe(
                arr["adjacency_pointer"], arr["adjacency_vector"],
                arr["adjacency_vector_weights"],
                arr["adjacynct_vector_network_node"],
                arr["o_terminal_idxs"], arr["o_terminal_weights"],
                arr["cutoff"], arr["d_terminal_idxs"], arr["d_count"],
                arr["adjacency_pointer"].shape[0] - 1)
            got = (bool(a1_ok), bool(a3_ok))
            exp = fc.GUARD_EXPECTATIONS[variant]
            if got != (exp[0], exp[1]):
                fail(f"guard cell {variant}: got {got} expected {exp}")
            guard_rows[variant] = {"a1_admitted": got[0],
                                   "a1_max_degree": int(a1_max),
                                   "a3_admitted": got[1]}
        rec("guard_matrix", cells=guard_rows)

    elif scratch is not None:
        fail("b0 arm unexpectedly imports Engines._large_access_scratch")

    # cross-arm kernel bytes (both arms run the SAME public kernels)
    byte_rows = {}
    for variant in ("both_admit", "a3_refuse_oob", "a1_refusal"):
        arr = getattr(fc, variant)()
        od_plain = np.asarray(od_compact_vector_node_view_scope(
            arr["o_terminal_idxs"], arr["o_terminal_weights"],
            arr["adjacency_pointer"], arr["adjacency_vector"],
            arr["adjacency_vector_weights"],
            arr["adjacynct_vector_network_node"], arr["cutoff"],
            arr["d_count"], arr["d_terminal_idxs"],
            arr["d_terminal_weights"]))
        outs = integrated_scope_access(
            arr["o_terminal_idxs"], arr["o_terminal_weights"],
            arr["adjacency_pointer"], arr["adjacency_vector"],
            arr["adjacency_vector_weights"],
            arr["adjacynct_vector_network_node"],
            arr["d_terminal_idxs"], arr["d_terminal_weights"],
            arr["d_weights"], arr["gravity_beta"], arr["metric_plateau"],
            arr["metric_midpoint"],
            float(np.log(99.0) / arr["metric_midpoint"]),
            arr["knn_decay"], arr["knn_weights"], arr["cutoff"])
        byte_rows[variant] = {
            "od_scope": sha(od_plain),
            "integrated": [sha(np.asarray(o)) for o in outs],
        }
    rec("a_side_bytes", variants=byte_rows)
    return byte_rows


# ======================================================================
# Stage 3: F-side — flow engines, F2 route observability
# ======================================================================
def f_side_leg(kind, arm):
    import fixtures as oracle_fixtures
    import stub_topology
    import fixtures_composition as fc
    from urban_network_analysis.Settings import Settings
    from urban_network_analysis.Engines.AggregateFlow import AggregateFlow

    def run_spec(spec):
        overrides = oracle_fixtures.flow_settings_overrides(node_flow=True)
        topo = stub_topology.StubFlowTopology(spec)
        settings = stub_topology.make_settings(Settings,
                                               accessibility=False,
                                               **overrides)
        eng = AggregateFlow(topo)
        eng.num_threads = 2
        eng.Centrality(settings)
        outs = {"edge_flow_AB": eng.edge_flow_AB,
                "edge_flow_BA": eng.edge_flow_BA,
                "edge_flow": eng.edge_flow,
                "node_flow": eng.node_flow}
        return eng, {k: (None if v is None else sha(v))
                     for k, v in outs.items()}

    rows = {}
    for label, spec in (("long_admitting", fc.flow_spec_long()),
                        ("oracle_refusing", fc.flow_spec_oracle())):
        eng, hashes = run_spec(spec)
        stats = getattr(eng, "_f2_stats", None)
        rows[label] = {"hashes": hashes,
                       "f2_stats": None if stats is None else {
                           k: (bool(v) if isinstance(v, bool) else int(v))
                           for k, v in stats.items()}}
        if kind == "selected":
            if stats is None:
                fail(f"selected arm flow leg {label}: no _f2_stats")
            if label == "long_admitting" and not (
                    stats["local_route_ok"] and stats["fast_calls"] >= 1):
                fail(f"selected {label}: F2 local route did not engage: {stats}")
            if label == "oracle_refusing" and not (
                    stats["local_route_ok"] and stats["fallback_calls"] >= 1):
                fail(f"selected {label}: F2 fallback did not engage: {stats}")
        elif stats is not None:
            fail(f"b0 arm flow leg {label}: unexpected _f2_stats {stats}")
    rec("f_side", legs=rows)
    return {k: v["hashes"] for k, v in rows.items()}


# ======================================================================
# Stage 4: F3-side — gradient precompute, chunk schedules
# ======================================================================
def f3_side_leg(kind, arm):
    import fixtures_composition as fc
    from urban_network_analysis.Engines.AggregateFlow import AggregateFlow

    stub = fc.make_gradient_stub(fc.grad_arcs_chain(), n_net=8, n_dest=2,
                                 n_extra=1)
    rows = {}
    for label, limit in (("production_cap", np.inf),
                         ("forced_chunks", 6.0)):
        stub._gradient_limit = lambda ns_arg, _lim=limit: _lim
        out = AggregateFlow._precompute_dest_gradients(stub, {})
        rows[label] = [sha(np.asarray(o)) for o in out]
        stub.logger.calls.clear()
    nofit = None
    if kind == "selected":
        stub._gradient_limit = lambda ns_arg: 2.0   # forced NO_FIT
        AggregateFlow._precompute_dest_gradients(stub, {})
        lines = stub.logger.v2_lines()
        nofit = {"v2_line_count": len(lines),
                 "has_nofit_token": any(
                     "NO_FIT" in ln or "fallback" in ln for ln in lines),
                 "lines": lines[:6]}
        if not nofit["v2_line_count"]:
            fail("selected NO_FIT leg produced no schedule log lines")
    else:
        # _lfws lives in the AggregateFlow MODULE namespace, not on the
        # class — check the module (imported when the class was resolved)
        flow_mod = sys.modules.get(
            "urban_network_analysis.Engines.AggregateFlow")
        if flow_mod is not None and hasattr(flow_mod, "_lfws"):
            fail("b0 arm unexpectedly carries the _lfws workspace module hook")
    rec("f3_side", schedules=rows, no_fit=nofit)
    return rows


# ======================================================================
def main(argv=None):
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--kind", required=True, choices=["selected", "b0"])
    ap.add_argument("--identity", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    RESULT.update({"task": "W00", "record": "installed_engagement_probe",
                   "arm": args.arm, "kind": args.kind,
                   "utc_env": {"NUMBA_NUM_THREADS":
                               os.environ.get("NUMBA_NUM_THREADS"),
                               "NUMBA_CACHE_DIR":
                               os.environ.get("NUMBA_CACHE_DIR")}})

    from harness import identity as idmod
    identity, imported = identity_leg(idmod, args.identity)
    # C2 (W00 resume-fire adjudication): package_root is a PosixPath —
    # stringify at the record site; the dump sites below also pass
    # default=str so no Path-typed value can crash a receipt write again.
    RESULT["identity"] = {"package_root": str(identity.package_root),
                          "tree_sha256": identity.package_tree_sha256,
                          "verified_modules": imported["verified_modules"]}

    a_rows = a_side_leg(args.kind, args.arm)
    f_rows = f_side_leg(args.kind, args.arm)
    f3_rows = f3_side_leg(args.kind, args.arm)
    RESULT["byte_hashes"] = {"a_side": a_rows, "f_side": f_rows,
                             "f3_side": f3_rows}
    RESULT["status"] = "ok"

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(RESULT, fh, indent=1, default=str)
    print(f"[w00-probe] OK arm={args.arm} records={len(RESULT['records'])}")
    return 0


if __name__ == "__main__":
    args = None
    sys.exit(main())
