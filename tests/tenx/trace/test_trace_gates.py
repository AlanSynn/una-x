"""C10 obligations discharged as C11 runtime gates (campaign una_10x).

C10-MINOR-1: trace slice validation is ENFORCED BEHAVIORALLY —
structurally impossible records (broken indptr, non-ascending or
out-of-range destination ids, wrong dtypes) are refused at replay
before any fold, never replayed as a cache hit.  The recording-only
text checks stay recording-only.

C10-NOTE-3: a SECOND, permanently independent derivation of the fold's
metric dependency facts, cross-checked against the C10 table
(metric_dependencies.gen2.json, generation 2; generation 1 is
preserved untouched as metric_dependencies.json): the engine fold's
own signature is
introspected at runtime, the engine file's bytes are re-hashed, the
replay binding is compared to the C10 unanimous binding record, and
each metric-only input is poison-tested behaviorally (a changed value
changes outputs; the trace identity and packed bytes never move).
"""
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from conftest import (RESULT_FIELDS, admitted_case, bits, build)

REPO = Path(__file__).resolve().parents[3]
ENGINE_PATH = (REPO / "src" / "urban_network_analysis" / "Engines" /
               "AccessibilityWElevation.py")
# Generation 2: the engine evolved by the gated lazy-import change
# (exl-20261011T164228Z); generation 1 stays preserved untouched as
# metric_dependencies.json (Z00 campaign-close record).
DEPS_PATH = (REPO / "tests" / "tenx" / "identity" / "tables" /
             "metric_dependencies.gen2.json")
KNN = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
METRIC_ONLY = ("gravity_beta", "gravity_plateau",
               "gravity_logistic_midpoint", "gravity_growth_rate",
               "knn_decay", "knn_weights")


def forged(tr, rec, **overrides):
    """A record carrying tampered packed arrays (fresh writable
    copies; the frozen original untouched).

    The content digest is RECOMPUTED over the final forged bytes: the
    structural gates are exercised against a digest-consistent forger
    (the stronger adversary).  Tampering that leaves a STALE digest is
    the weaker adversary — pinned separately by the C11-NOTE-4 test.
    """
    fields = {f.name: getattr(rec, f.name)
              for f in dataclasses.fields(rec)}
    fields.update(overrides)
    fields["content_digest"] = hashlib.sha256(
        np.asarray(fields["indptr"]).tobytes()
        + np.asarray(fields["destination_ids"]).tobytes()
        + np.asarray(fields["distance_bits"]).tobytes()).hexdigest()
    return tr.TraceRecord(**fields)


def packed_of(rec):
    return (np.asarray(rec.indptr).tobytes(),
            np.asarray(rec.destination_ids).tobytes(),
            np.asarray(rec.distance_bits).tobytes())


def test_c10_minor_1_broken_indptr_refused(tr):
    case = admitted_case("base")
    rec = build(tr, case)
    bad = np.array([1, 2], dtype=np.int64)          # indptr[0] != 0
    r = tr.replay_trace(forged(tr, rec, indptr=bad), case["dwt"],
                        1.5, 2.0, 3.0, 0.8, "exponential", KNN,
                        case["cutoff"])
    assert r.route == "trace_replay_refused"
    assert r.refusal == "corrupt_trace_structure"
    bad = np.array([0, 9], dtype=np.int64)          # nnz beyond arrays
    r = tr.replay_trace(forged(tr, rec, indptr=bad), case["dwt"],
                        1.5, 2.0, 3.0, 0.8, "exponential", KNN,
                        case["cutoff"])
    assert r.refusal == "corrupt_trace_structure"
    bad = np.array([0, 2, 1], dtype=np.int64)       # non-monotone
    r = tr.replay_trace(forged(tr, rec, indptr=bad, origin_count=2),
                        case["dwt"], 1.5, 2.0, 3.0, 0.8, "exponential",
                        KNN, case["cutoff"])
    assert r.refusal == "corrupt_trace_structure"
    bad = np.zeros(2, dtype=np.int32)               # dtype plan broken
    r = tr.replay_trace(forged(tr, rec, indptr=bad), case["dwt"],
                        1.5, 2.0, 3.0, 0.8, "exponential", KNN,
                        case["cutoff"])
    assert r.refusal == "corrupt_trace_structure"


def test_c10_minor_1_non_ascending_ids_refused(tr):
    case = admitted_case("base")
    rec = build(tr, case)
    ids = np.asarray(rec.destination_ids)[::-1].copy()
    r = tr.replay_trace(forged(tr, rec, destination_ids=ids),
                        case["dwt"], 1.5, 2.0, 3.0, 0.8, "exponential",
                        KNN, case["cutoff"])
    assert r.route == "trace_replay_refused"
    assert r.refusal == "corrupt_trace_slice_order"
    ids = np.asarray(rec.destination_ids).copy()
    ids[len(ids) // 2] = ids[0]                     # duplicate
    r = tr.replay_trace(forged(tr, rec, destination_ids=ids),
                        case["dwt"], 1.5, 2.0, 3.0, 0.8, "exponential",
                        KNN, case["cutoff"])
    assert r.refusal == "corrupt_trace_slice_order"


def test_c10_minor_1_out_of_range_ids_refused(tr):
    case = admitted_case("base")
    rec = build(tr, case)
    ids = np.asarray(rec.destination_ids).copy()
    ids[0] = rec.dest_count
    r = tr.replay_trace(forged(tr, rec, destination_ids=ids),
                        case["dwt"], 1.5, 2.0, 3.0, 0.8, "exponential",
                        KNN, case["cutoff"])
    assert r.route == "trace_replay_refused"
    assert r.refusal == "corrupt_trace_destination_ids"
    ids = np.asarray(rec.destination_ids).copy()
    ids[-1] = -1
    r = tr.replay_trace(forged(tr, rec, destination_ids=ids),
                        case["dwt"], 1.5, 2.0, 3.0, 0.8, "exponential",
                        KNN, case["cutoff"])
    assert r.refusal == "corrupt_trace_destination_ids"
    dist = np.zeros(int(rec.indptr[-1]), dtype=np.float32)
    r = tr.replay_trace(forged(tr, rec, distance_bits=dist),
                        case["dwt"], 1.5, 2.0, 3.0, 0.8, "exponential",
                        KNN, case["cutoff"])
    assert r.refusal == "corrupt_trace_structure"


def test_c10_note_3_engine_bytes_and_signature_vs_table(tr):
    """The engine fold's runtime signature and the engine file's actual
    bytes are re-derived and compared against the C10 table."""
    deps = json.loads(DEPS_PATH.read_text())
    engine_sha = hashlib.sha256(ENGINE_PATH.read_bytes()).hexdigest()
    assert engine_sha == deps["numerical_env"]["engine_sha256"]
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        reach_gravity_knn_access)
    assert deps["fold"]["fold_name"] == "reach_gravity_knn_access"
    params = list(inspect.signature(
        reach_gravity_knn_access.py_func).parameters)
    assert deps["fold"]["outputs"] == list(RESULT_FIELDS)
    for alias, rec in deps["alias_bindings"].items():
        assert alias in params, alias
        assert rec["bound_to"] in params, alias
    assert deps["call_sites"]["bindings_unanimous"] is True
    site = deps["call_sites"]["per_site"]
    binding = next(iter(site.values()))
    assert set(binding) == set(params)
    replay_params = list(inspect.signature(
        tr.replay_trace).parameters)
    legal_sources = set(replay_params) | {"d_distance", "d_weights"}
    for fold_param, source in binding.items():
        assert fold_param in params
        assert source in legal_sources, (fold_param, source)
    build_params = list(inspect.signature(tr.build_trace).parameters)
    id_params = list(inspect.signature(
        tr.compute_trace_id).parameters)
    # build_trace carries d_weights for ADMISSION only (the S11 gate's
    # 10-arity); the identity carries NONE of the metric-only inputs.
    for name in METRIC_ONLY:
        assert name not in build_params, name
    for name in METRIC_ONLY + ("d_weights",):
        assert name not in id_params, name


def test_c10_note_3_d_weights_admission_only_behavior(tr):
    """Behavioral proof that build-time d_weights is admission-only:
    two builds with different destination METRIC weights yield the
    same identity and bytewise-identical packed arrays, and both
    replays match their own full runs."""
    case = admitted_case("base")
    other = np.array([1.7, 0.2, 0.9, 1.1, 0.05, 0.6])
    a = build(tr, case)
    case_b = dict(case)
    case_b["dwt"] = other
    b = build(tr, case_b)
    assert a.trace_id == b.trace_id
    assert (np.asarray(a.indptr).tobytes()
            == np.asarray(b.indptr).tobytes())
    assert (np.asarray(a.destination_ids).tobytes()
            == np.asarray(b.destination_ids).tobytes())
    assert (np.asarray(a.distance_bits).tobytes()
            == np.asarray(b.distance_bits).tobytes())


def test_c10_note_3_poison_sensitivity(tr):
    """Behavioral poison tests: every metric-only input changes at
    least one output bit (the dependency table is not vacuous) while
    the trace identity and packed bytes never move."""
    case = admitted_case("base")
    rec = build(tr, case)
    packed0 = packed_of(rec)

    def rep(**kw):
        return tr.replay_trace(
            rec, case["dwt"], kw.get("gravity_beta", 1.5),
            kw.get("gravity_plateau", 2.0),
            kw.get("gravity_logistic_midpoint", 3.0),
            kw.get("gravity_growth_rate", 0.8),
            kw.get("knn_decay", "exponential"),
            kw.get("knn_weights", KNN), case["cutoff"])

    base_rp = rep()
    poisons = [dict(gravity_beta=2.5), dict(gravity_plateau=0.5),
               dict(gravity_logistic_midpoint=1.25),
               dict(gravity_growth_rate=0.4), dict(knn_decay="logistic"),
               dict(knn_decay="other"),
               dict(knn_weights=np.full(10, 0.3))]
    for kw in poisons:
        rp = rep(**kw)
        changed = any(bits(getattr(rp, f)) != bits(getattr(base_rp, f))
                      for f in RESULT_FIELDS)
        assert changed, kw
        assert build(tr, case).trace_id == rec.trace_id
        assert packed_of(rec) == packed0
