"""F1 fixed-stripe tests (proof 10): T3 per-stripe partials (replica),
T4 repeated-concurrency determinism (in-process + bounded children),
T5 concurrent independent engines, T13 stripe-buffer distinctness, and
the M1/M2 first-divergence negatives (fold order, contiguous
partition)."""
from __future__ import annotations

import os
import subprocess
import threading

import numpy as np
import pytest

import fixtures
import fixtures_f1
from comparator import assert_array_bytes_equal
from harness_f1 import assert_engine_bytes, b0ns, cns, record_note


from _legacy_paths import LEGACY_WS  # portable-path fix (HARNESS 2026-09-30)
HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN_DATA = str(LEGACY_WS) + "/campaign_data"
VENV_PY = str(LEGACY_WS) + "/venvs/campaign/bin/python"


def _replica(ns, engine, settings, k, **knobs):
    return fixtures_f1.striped_replica(ns, engine, settings, k, **knobs)


def _build_and_run(ns, case, k):
    eng, settings = fixtures_f1.build_engine(ns, case, k)
    eng.Centrality(settings)
    return eng, settings


@pytest.mark.parametrize("k", [1, 2, 3, 9])
def test_t3_replica_matches_engine_and_partials_arm_invariant(k):
    """(a) The replica's slot-order reduction byte-equals the SAME
    arm's engine output at the same K; (b) per-stripe partials before
    the reduction are arm-invariant (B0 replica == candidate replica,
    byte-exact). Each K is a separate K-matched pair."""
    eng_b, settings_b = _build_and_run(b0ns(), "base", k)
    eng_c, settings_c = _build_and_run(cns(), "base", k)

    rep_b = _replica(b0ns(), eng_b, settings_b, k)
    rep_c = _replica(cns(), eng_c, settings_c, k)

    assert rep_b["n_kernel_calls"] == rep_c["n_kernel_calls"] > 0
    assert rep_b["stripe_members"] == rep_c["stripe_members"]
    assert rep_b["fold_sequence"] == list(range(k))

    # (a) replica reduction == engine output, same arm, same K
    assert_array_bytes_equal(rep_b["final_AB"], eng_b.edge_flow_AB,
                             f"t3/k{k}/b0/final_AB_vs_engine")
    assert_array_bytes_equal(rep_b["final_BA"], eng_b.edge_flow_BA,
                             f"t3/k{k}/b0/final_BA_vs_engine")
    assert_array_bytes_equal(rep_c["final_AB"], eng_c.edge_flow_AB,
                             f"t3/k{k}/cand/final_AB_vs_engine")
    assert_array_bytes_equal(rep_c["final_BA"], eng_c.edge_flow_BA,
                             f"t3/k{k}/cand/final_BA_vs_engine")
    if eng_b.node_flow is not None:
        assert_array_bytes_equal(rep_b["final_node"], eng_b.node_flow,
                                 f"t3/k{k}/b0/final_node_vs_engine")
        assert_array_bytes_equal(rep_c["final_node"], eng_c.node_flow,
                                 f"t3/k{k}/cand/final_node_vs_engine")

    # (b) per-stripe partials are arm-invariant
    for slot in range(k):
        assert_array_bytes_equal(rep_c["local_AB"][slot],
                                 rep_b["local_AB"][slot],
                                 f"t3/k{k}/slot{slot}/local_AB")
        assert_array_bytes_equal(rep_c["local_BA"][slot],
                                 rep_b["local_BA"][slot],
                                 f"t3/k{k}/slot{slot}/local_BA")
        if eng_b.node_flow is not None:
            assert_array_bytes_equal(rep_c["local_node"][slot],
                                     rep_b["local_node"][slot],
                                     f"t3/k{k}/slot{slot}/local_node")


def test_t4_inprocess_repeats_k3():
    """R independent K=3 engine runs (fresh objects, both arms) are all
    byte-identical to each other and arm-matched."""
    runs = {"b0": [], "cand": []}
    for _ in range(3):
        for ns, tag in ((b0ns(), "b0"), (cns(), "cand")):
            eng, _ = _build_and_run(ns, "base", 3)
            runs[tag].append(fixtures_f1.engine_output_arrays(eng))
    for tag, arrs in runs.items():
        for i, arrs_i in enumerate(arrs[1:], start=1):
            assert_engine_bytes(arrs_i, arrs[0], f"t4/{tag}/run{i}")
    assert_engine_bytes(runs["cand"][0], runs["b0"][0], "t4/k3")


def test_t4_bounded_children_k3(tmp_path):
    """Two bounded subprocess children (one per import root, each with
    its own NUMBA_CACHE_DIR) run the K=3 engine twice and save outputs;
    parent byte-compares cross-arm and against the in-process runs."""
    child_npz = {}
    for tag, root in (("b0", b0ns().root), ("cand", cns().root)):
        out = str(tmp_path / f"mt_child_{tag}.npz")
        env = dict(os.environ)
        env["NUMBA_CACHE_DIR"] = os.path.join(
            CAMPAIGN_DATA, f"nbc_f1_child_{tag}")
        env["NUMBA_NUM_THREADS"] = "4"
        env.pop("L1_REUSE_DIR", None)
        proc = subprocess.run(
            [VENV_PY, os.path.join(HERE, "mt_child_f1.py"), root, out, "3"],
            env=env, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, \
            f"{tag} child failed:\n{proc.stdout}\n{proc.stderr}"
        assert "F1_MT_CHILD_OK" in proc.stdout
        with np.load(out) as data:
            child_npz[tag] = {k: data[k] for k in data.files}
    for key in child_npz["b0"]:
        assert child_npz["cand"][key].tobytes() == \
            child_npz["b0"][key].tobytes(), f"t4_children/{key} diverged"
    # children agree with the in-process same-arm runs
    for ns, tag in ((b0ns(), "b0"), (cns(), "cand")):
        eng, _ = _build_and_run(ns, "base", 3)
        assert eng.edge_flow_AB.tobytes() == \
            child_npz[tag]["edge_flow_AB"].tobytes(), \
            f"t4_children/{tag}/inprocess-vs-child"


def test_t5_concurrent_independent_engines():
    """Two engines on INDEPENDENT topologies running simultaneously in
    one process (per arm): each concurrent output is byte-equal to its
    solo run (no shared mutable state; nogil overlap does not perturb
    results)."""
    spec_a = fixtures.flow_network_spec()
    spec_b = {**spec_a,
              "origins": [list(o) for o in spec_a["origins"]]}
    spec_b["origins"][1][5] = 1.5   # deterministic variant of topology A
    spec_b["origins"][3][5] = 0.0

    solo = {}
    for ns_tag, ns in (("b0", b0ns()), ("cand", cns())):
        for spec_tag, spec in (("a", spec_a), ("b", spec_b)):
            eng, settings = fixtures_f1.build_engine(
                ns, "base", 3, spec=spec)
            eng.Centrality(settings)
            solo[(ns_tag, spec_tag)] = fixtures_f1.engine_output_arrays(
                eng)

    def run_pair(ns, specs, out):
        for spec in specs:
            eng, settings = fixtures_f1.build_engine(
                ns, "base", 3, spec=spec)
            eng.Centrality(settings)
            out.append(fixtures_f1.engine_output_arrays(eng))

    for ns_tag, ns in (("b0", b0ns()), ("cand", cns())):
        out_a, out_b = [], []
        t1 = threading.Thread(target=run_pair, args=(ns, [spec_a], out_a))
        t2 = threading.Thread(target=run_pair, args=(ns, [spec_b], out_b))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(out_a) == len(out_b) == 1
        assert_engine_bytes(out_a[0], solo[(ns_tag, "a")],
                            f"t5/{ns_tag}/concurrent-a")
        assert_engine_bytes(out_b[0], solo[(ns_tag, "b")],
                            f"t5/{ns_tag}/concurrent-b")


def test_t13_stripe_buffers_distinct_objects():
    """At K>1 the per-stripe AB/BA/node buffers are distinct objects
    (replica mirrors the driver's allocation); with node flow disabled
    every slot shares the SAME shape-0 sentinel, mirroring the driver's
    _empty_node aliasing — and outputs still byte-match."""
    eng, settings = _build_and_run(b0ns(), "base", 3)
    rep = _replica(b0ns(), eng, settings, 3)
    assert len(set(id(a) for a in rep["local_AB"])) == 3
    for i in range(3):
        for j in range(i + 1, 3):
            assert not np.shares_memory(rep["local_AB"][i],
                                        rep["local_AB"][j])
            assert not np.shares_memory(rep["local_BA"][i],
                                        rep["local_BA"][j])
            assert not np.shares_memory(rep["local_node"][i],
                                        rep["local_node"][j])

    eng_nn, settings_nn = _build_and_run(b0ns(), "no_node_flow", 3)
    rep_nn = _replica(b0ns(), eng_nn, settings_nn, 3)
    assert len(set(id(a) for a in rep_nn["local_node"])) == 1
    assert rep_nn["local_node"][0].shape == (0,)
    # engine outputs byte-equal across arms in the node-off case too
    eng_c, _ = _build_and_run(cns(), "no_node_flow", 3)
    assert_engine_bytes(fixtures_f1.engine_output_arrays(eng_c),
                        fixtures_f1.engine_output_arrays(eng_nn),
                        "t13/nodeoff/engine")


def _first_byte_diff(a, b):
    ab = np.asarray(a).tobytes()
    bb = np.asarray(b).tobytes()
    assert ab != bb
    return next(i for i, (x, y) in enumerate(zip(ab, bb)) if x != y)


@pytest.mark.parametrize("ns_tag", ["b0", "cand"])
def test_m1_fold_descending_diverges(ns_tag):
    """M1 negative: folding slots K-1..0 (instead of the production
    slot order) must produce byte-different finals on the adversarial
    fixture — the comparator catches the reassociation."""
    ns = b0ns() if ns_tag == "b0" else cns()
    eng, settings = _build_and_run(ns, "decay_equal", 3)
    prod = _replica(ns, eng, settings, 3)
    mut = _replica(ns, eng, settings, 3, fold="descending")
    assert mut["fold_sequence"] == [2, 1, 0]
    diverged = []
    for name in ("final_AB", "final_BA", "final_node"):
        a, b = prod[name], mut[name]
        if a.shape[0] and a.tobytes() != b.tobytes():
            diverged.append({"array": name,
                             "first_diff_byte": _first_byte_diff(a, b)})
    assert diverged, (f"M1 fold-order mutant NOT caught on arm "
                      f"{ns_tag} — comparator sensitivity failure")
    record_note(f"m1_{ns_tag}", {"mutations_caught": diverged})
    # and the production replica still equals the engine output
    assert_array_bytes_equal(prod["final_AB"], eng.edge_flow_AB,
                             f"m1/{ns_tag}/prod_final_AB_vs_engine")


@pytest.mark.parametrize("ns_tag", ["b0", "cand"])
def test_m2_contiguous_partition_diverges(ns_tag):
    """M2 negative: a contiguous-block stripe decomposition (instead of
    the production range(slot, n_origins, K) striding) must be caught."""
    ns = b0ns() if ns_tag == "b0" else cns()
    eng, settings = _build_and_run(ns, "decay_equal", 3)
    prod = _replica(ns, eng, settings, 3)
    mut = _replica(ns, eng, settings, 3, partition="contiguous")
    assert mut["stripe_members"] != prod["stripe_members"]
    diverged = []
    for name in ("final_AB", "final_BA", "final_node"):
        a, b = prod[name], mut[name]
        if a.shape[0] and a.tobytes() != b.tobytes():
            diverged.append({"array": name,
                             "first_diff_byte": _first_byte_diff(a, b)})
    assert diverged, (f"M2 contiguous-partition mutant NOT caught on "
                      f"arm {ns_tag} — comparator sensitivity failure")
    record_note(f"m2_{ns_tag}", {"mutations_caught": diverged})
