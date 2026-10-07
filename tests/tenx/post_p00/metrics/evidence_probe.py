"""P03M evidence probe (campaign una_10x post_p00, designs/04).

Recomputes the dependency matrix and the MetricID v2 identity laws
LIVE (the same checks the suites assert) and writes the two campaign
evidence objects:

  metric_dependency_matrix.json — one row per (dependency, KNN
  branch): predicted changed fields (from the fold source lines) vs
  EMPIRICAL changed fields (bitwise, on a real admitted C11 case),
  plus the replay-refusal classes (typed cutoff, byte order) and the
  0/-0 record.

  metric_identity_v2.json — the schema field inventory and the
  collision/refusal verdicts (v1 witnesses computed live).

Usage (repo root, venv-p01m, fresh NUMBA_CACHE_DIR):

    NUMBA_CACHE_DIR=<fresh dir> python \\
        tests/tenx/post_p00/metrics/evidence_probe.py <out_dir>
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import conftest as mc  # the suite's own helpers (single source)


def main(out_dir):
    assert len(sys.argv) == 2, "usage: evidence_probe.py <out_dir>"
    import os
    assert os.environ.get("NUMBA_CACHE_DIR"), \
        "NUMBA_CACHE_DIR must be pinned for this probe"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from urban_network_analysis.Engines import _metric_identity as mid
    from urban_network_analysis.Engines import _trace_access_scratch as trm

    case = mc.admitted_case("base")
    rec = mc.tck.build(trm, case)

    # ------------------------------------------------------------------
    # 1. Dependency matrix (empirical bits vs source predictions)
    # ------------------------------------------------------------------
    rows = []

    def bundle(beta, plateau, mid_, growth):
        return (beta, plateau, mid_, growth)

    B = mc.BUNDLE
    branch_knn_reads = {
        "exponential": {"gravity_beta", "gravity_plateau"},
        "logistic": {"gravity_growth_rate", "gravity_plateau",
                     "gravity_logistic_midpoint"},
        "other": set(),
    }
    scalar_effects = {
        "gravity_beta": ({"gravity_exponential"},
                         bundle(2.5, B[1], B[2], B[3])),
        "gravity_plateau": ({"gravity_exponential", "gravity_logistic"},
                            bundle(B[0], 0.5, B[2], B[3])),
        "gravity_logistic_midpoint": ({"gravity_logistic"},
                                      bundle(B[0], B[1], 1.25, B[3])),
        "gravity_growth_rate": ({"gravity_logistic"},
                                bundle(B[0], B[1], B[2], 0.4)),
    }
    for decay, knn_reads in branch_knn_reads.items():
        for scalar, (base_effects, pert) in scalar_effects.items():
            base_b = mc.fields_bits(mc.replay_with(
                trm, rec, case, knn_decay=decay))
            pert_b = mc.fields_bits(mc.replay_with(
                trm, rec, case, knn_decay=decay, bundle=pert))
            empirical = sorted(mc.changed_fields(base_b, pert_b))
            predicted = sorted(base_effects
                               | ({"knn_access"} if scalar in knn_reads
                                  else set()))
            rows.append({
                "dependency": scalar,
                "knn_branch": decay,
                "predicted_changed": predicted,
                "empirical_changed": empirical,
                "match": predicted == empirical,
                "fold_source_lines": {
                    "gravity_beta": "AWE:83 (ge), AWE:104 (knn-exp)",
                    "gravity_plateau": "AWE:83/85/104/106",
                    "gravity_logistic_midpoint": "AWE:85/106",
                    "gravity_growth_rate": "AWE:85/106",
                }[scalar],
            })
    # d_weights values (shift +2.0: the int64 reach cast absorbs sub-
    # quantum bumps), knn_weights values, branch switch
    base_b = mc.fields_bits(mc.replay_with(trm, rec, case))
    dwt2 = np.asarray(case["dwt"], np.float64) + 2.0
    rows.append({
        "dependency": "d_weights.values(+2.0 all)", "knn_branch": "exponential",
        "predicted_changed": sorted(mc.RESULT_FIELDS),
        "empirical_changed": sorted(mc.changed_fields(
            base_b, mc.fields_bits(mc.replay_with(trm, rec, case,
                                                  d_weights=dwt2)))),
        "match": True, "fold_source_lines": "AWE:75-108 (all fields)",
    })
    rows[-1]["match"] = (rows[-1]["predicted_changed"]
                         == rows[-1]["empirical_changed"])
    kw2 = mc.METRIC_PARAMS[4] * 2.0
    emp = mc.changed_fields(base_b, mc.fields_bits(
        mc.replay_with(trm, rec, case, knn_weights=kw2)))
    rows.append({"dependency": "knn_weights.values(x2)",
                 "knn_branch": "exponential",
                 "predicted_changed": ["knn_access"],
                 "empirical_changed": emp, "match": emp == ["knn_access"],
                 "fold_source_lines": "AWE:90-108"})
    # cutoff membership across traces (derived from retained distances)
    db = np.asarray(rec.distance_bits)
    cutoff_b = float((db.min() + db.max()) / 2.0)
    case_b = dict(case)
    case_b["cutoff"] = cutoff_b
    rec_b = mc.tck.build(trm, case_b)
    emp = sorted(mc.changed_fields(base_b, mc.fields_bits(
        mc.replay_with(trm, rec_b, case_b))))
    nnz_shrank = int(rec_b.indptr[-1]) < int(rec.indptr[-1])
    rows.append({"dependency": "cutoff.membership(midpoint)",
                 "knn_branch": "exponential",
                 "predicted_changed": sorted(mc.RESULT_FIELDS),
                 "empirical_changed": emp,
                 "match": emp == sorted(mc.RESULT_FIELDS)
                 and nnz_shrank,
                 "fold_source_lines": "AWE:75 (membership filter)",
                 "trace_nnz_a": int(rec.indptr[-1]),
                 "trace_nnz_b": int(rec_b.indptr[-1])})
    # refusal classes
    res = mc.replay_with(trm, rec, case, cutoff=25)
    rows.append({"dependency": "cutoff.dtype_only(int 25 vs float 25.0)",
                 "knn_branch": "exponential",
                 "predicted_changed": "replay_refusal",
                 "empirical_changed": res.refusal,
                 "match": res.route == "trace_replay_refused"
                 and res.refusal == "cutoff_mismatch_with_trace_identity",
                 "fold_source_lines": "C11 typed cutoff identity"})
    dbe = np.asarray(case["dwt"], np.float64)
    dbe = dbe.astype(dbe.dtype.newbyteorder(">"))
    try:
        mc.replay_with(trm, rec, case, d_weights=dbe)
        be_refusal = "NO_REFUSAL"
    except Exception as exc:
        be_refusal = f"{type(exc).__module__}.{type(exc).__name__}"
    rows.append({"dependency": "d_weights.byteorder_only(>f8)",
                 "knn_branch": "exponential",
                 "predicted_changed": "numba_TypingError_refusal",
                 "empirical_changed": be_refusal,
                 "match": be_refusal.endswith("TypingError"),
                 "fold_source_lines": "numba dispatcher domain"})
    # 0/-0 pure sign-bit record
    retained = np.unique(rec.destination_ids)
    d_plus = np.asarray(case["dwt"], np.float64).copy()
    d_plus[retained[0]] = 0.0
    d_minus = d_plus.copy()
    d_minus[retained[0]] = -0.0
    emp = mc.changed_fields(
        mc.fields_bits(mc.replay_with(trm, rec, case, d_weights=d_plus)),
        mc.fields_bits(mc.replay_with(trm, rec, case,
                                      d_weights=d_minus)))
    rows.append({"dependency": "d_weights.zero_sign_bit(+0.0 vs -0.0)",
                 "knn_branch": "exponential",
                 "predicted_changed": "(case-empirical record)",
                 "empirical_changed": emp,
                 "match": emp == [],
                 "fold_source_lines": "IEEE-754 sum semantics; MetricID "
                                     "v2 distinguishes the inputs",
                 "note": "identity differs; output bits coincide on "
                         "this case (zero term absorbed by nonzero "
                         "partials) — conservative direction"})

    matrix = {
        "schema": "una-10x-p03m-dependency-matrix-1",
        "task": "P03M",
        "generated": "2026-10-06",
        "law": "designs/04: admission is a dependency proof; ONE "
               "dependency perturbed at a time; predictions derived "
               "from the fold source, outcomes bitwise",
        "case": "C11 battery admitted 'base' case",
        "rows": rows,
        "all_match": all(r["match"] for r in rows),
    }
    (out_dir / "metric_dependency_matrix.json").write_text(
        json.dumps(matrix, indent=1) + "\n")

    # ------------------------------------------------------------------
    # 2. MetricID v2 identity evidence (collisions + refusals, live)
    # ------------------------------------------------------------------
    from urban_network_analysis.cache import identity as _ckid
    rep = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 0.25, 0.75, 1.25, 1.75,
                    2.25], dtype=np.float64)

    def v1_key(knn_w, d_w, cutoff):
        return _ckid.stage_key("una:cohort:o_access:v1", {
            "trace_id": "witness", "gravity_beta": 1.5,
            "gravity_plateau": 2.0, "gravity_logistic_midpoint": 3.0,
            "gravity_decay_constant": 2.4, "knn_decay": "exponential",
            "knn_weights": np.asarray(knn_w, dtype=np.float64),
            "search_radius": float(cutoff),
            "d_weights": np.asarray(d_w, dtype=np.float64)})

    def v2_id(knn_w=None, d_w=None, cutoff=None, beta=None):
        return mid.compute_metric_id(rec,
                                     d_w if d_w is not None else case["dwt"],
                                     beta if beta is not None else B[0],
                                     B[1], B[2], B[3],
                                     "exponential",
                                     knn_w if knn_w is not None
                                     else mc.METRIC_PARAMS[4],
                                     cutoff if cutoff is not None
                                     else case["cutoff"])

    ident = {
        "schema": "una-10x-p03m-identity-v2-1",
        "task": "P03M",
        "generated": "2026-10-06",
        "metric_id_schema_version": mid.METRIC_ID_SCHEMA_VERSION,
        "metric_closure_version": mid.METRIC_CLOSURE_VERSION,
        "trace_schema_unchanged": trm.TRACE_SCHEMA_VERSION,
        "binding_fields": [name for name, _ in mid.metric_binding_fields(
            rec, *mc.replay_binding(case))],
        "e8_resolution": {
            "v1_status": "REPORTING ONLY — float()-/float64-coercing "
                         "(dtype-erasing); never consulted for reuse; "
                         "disclosure wired at cohort.py construction",
            "witnesses": [
                {"law": "int cutoff 800 vs float 800.0",
                 "v1_collides": True, "v2_distinguishes": True},
                {"law": "float32-exact knn_weights values, f32 vs f64",
                 "v1_collides": None, "v2_distinguishes": None},
            ],
        },
        "refusal_enumeration": [
            "complex dtypes", "object arrays", "non-ndarray weights",
            "string scalars", "python ints beyond signed 64-bit",
        ],
        "live_checks": {},
    }
    ident["e8_resolution"]["witnesses"][0].update({
        "v1_collides": v1_key(mc.METRIC_PARAMS[4], case["dwt"], 25)
        == v1_key(mc.METRIC_PARAMS[4], case["dwt"], 25.0),
        "v2_distinguishes": v2_id(cutoff=25) != v2_id(cutoff=25.0)})
    ident["e8_resolution"]["witnesses"][1].update({
        "v1_collides": v1_key(rep, case["dwt"], case["cutoff"])
        == v1_key(rep.astype(np.float32), case["dwt"], case["cutoff"]),
        "v2_distinguishes": v2_id(knn_w=rep)
        != v2_id(knn_w=rep.astype(np.float32))})
    qnan = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000))[0]
    qnan1 = struct.unpack("<d", struct.pack("<Q",
                                            0x7FF8000000000001))[0]
    checks = {
        "stable_within_process": v2_id() == v2_id(),
        "negative_zero_beta_distinct":
            v2_id(beta=-0.0) != v2_id(beta=0.0),
        "nan_payloads_distinct": v2_id(beta=qnan) != v2_id(beta=qnan1),
        "one_ulp_beta_distinct":
            v2_id(beta=float(np.nextafter(B[0], 2.0))) != v2_id(),
        "dtype_only_dweights_distinct":
            v2_id(d_w=np.asarray(case["dwt"], np.float64).astype(
                np.float32)) != v2_id(),
        "byteorder_only_distinct":
            v2_id(d_w=np.asarray(case["dwt"], np.float64).astype(
                np.asarray(case["dwt"], np.float64).dtype.newbyteorder(
                    ">"))) != v2_id(),
        "layout_only_distinct": None,
        "writability_distinct": None,
        "float16_knn_encodable_distinct":
            v2_id(knn_w=mc.METRIC_PARAMS[4].astype(np.float16))
            != v2_id(knn_w=mc.METRIC_PARAMS[4].astype(np.float32)),
        "fold_closure_binds_module_bytes": None,
    }
    doubled = np.concatenate([case["dwt"], case["dwt"]])
    checks["layout_only_distinct"] = (
        v2_id(d_w=doubled[::2]) != v2_id(
            d_w=np.ascontiguousarray(doubled[::2])))
    d_rw = np.asarray(case["dwt"], np.float64).copy()
    d_ro = d_rw.copy()
    d_ro.flags.writeable = False
    checks["writability_distinct"] = (
        v2_id(d_w=d_rw) != v2_id(d_w=d_ro))
    checks["fold_closure_binds_module_bytes"] = (
        dict(mid.metric_binding_fields(
            rec, *mc.replay_binding(case)))["fold_module_sha256"]
        == hashlib.sha256(
            open(__import__("urban_network_analysis.Engines."
                            "AccessibilityWElevation",
                            fromlist=["x"]).__file__, "rb").read()
        ).hexdigest())
    ident["live_checks"] = checks
    ident["all_checks_pass"] = (
        all(v is True for k, v in checks.items())
        and all(w["v1_collides"] and w["v2_distinguishes"]
                for w in ident["e8_resolution"]["witnesses"]))
    (out_dir / "metric_identity_v2.json").write_text(
        json.dumps(ident, indent=1) + "\n")

    print(f"EVIDENCE OK matrix_all_match={matrix['all_match']} "
          f"identity_all_pass={ident['all_checks_pass']} -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
