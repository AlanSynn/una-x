"""Private fine-grained metric reuse qualification (campaign una_10x,
C12; dossier campaigns/una_10x/dossiers/05_metric_reuse.md).

The split-fold variants in the generated region below are AST surgery
on the engine's own ``reach_gravity_knn_access`` source (the splice
tool keeps the exact signature and the retention/gather prelude
verbatim and returns exactly one output), so a split fold is the same
ordered plan by construction.  Whether the COMPILER keeps the same
bits for a single-output extraction is an empirical question: the C12
characterization (tests/tenx/metrics) measures every variant against
the combined fold across lengths, tie permutations, extremes and
branches.  One 1-ULP divergence rejects the fine-grained profile —
never a tolerance.

The output-dependency map is the dossier-05 law per branch.  The
replay planner groups metric jobs by their typed MetricID and, only
when the characterization accepted splits, reuses an output across
jobs exactly when the map proves that output cannot read any changed
parameter.  A changed destination-weight vector recomputes EVERY
weighted output (all of them read it).  Reused values are copied into
FRESH correctly typed arrays per public job; shared storage is
private.  ``allow_split=False`` selects the safe trace-only
continuation (whole combined bundle per distinct metric input).
"""
import hashlib
from pathlib import Path

import numba as nb
import numpy as np

from ._trace_access_scratch import _frame, _frame_fields, _identity

# The split folds decorate with the ENGINE'S OWN flags: a different
# decorator would be a different compiler environment and the
# characterization would not be about extraction alone.
from .AccessibilityWElevation import (NUMBA_CACHE, NUMBA_FASTMATH,
                                      NUMBA_NOGIL, NUMBA_PARALLEL)

METRIC_REUSE_SCHEMA = "una-10x-c12-metric-1"


def _engine_sha256():
    import urban_network_analysis.Engines.AccessibilityWElevation \
        as eng
    return hashlib.sha256(Path(eng.__file__).read_bytes()).hexdigest()


_split_njit = nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_CACHE,
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
)

# ---------------------------------------------------------------------------
# Output-dependency map (dossier 05, per branch).  Every output lists
# EVERYTHING it may read: membership (cutoff + distances), the kept
# distances and weights in ORIGINAL order, its branch parameters, and
# (for knn) the argsort/crop order of the kept distances.  A parameter
# absent from an output's list is characterization-only for that
# output; the poison tests must show no bit change.
# ---------------------------------------------------------------------------
OUTPUT_DEPENDENCIES = {
    # reach reads the retention filter and the kept WEIGHTS only — it
    # never touches the kept distances (fold bytes + the C10
    # plan_fields_per_output table agree); kept_distances is not a
    # reach dependency
    "reach": ["membership_cutoff", "kept_weights", "output_cast"],
    "gravity_exponential": ["membership_cutoff", "kept_distances",
                            "kept_weights", "gravity_beta",
                            "gravity_plateau"],
    "gravity_logistic": ["membership_cutoff", "kept_distances",
                         "kept_weights", "gravity_growth_rate",
                         "gravity_plateau",
                         "gravity_logistic_midpoint"],
    "knn_access": {
        "exponential": ["membership_cutoff", "kept_distances",
                        "kept_weights", "knn_weights", "argsort_order",
                        "gravity_beta", "gravity_plateau"],
        "logistic": ["membership_cutoff", "kept_distances",
                     "kept_weights", "knn_weights", "argsort_order",
                     "gravity_growth_rate", "gravity_plateau",
                     "gravity_logistic_midpoint"],
        "other": ["membership_cutoff", "kept_distances",
                  "kept_weights", "knn_weights", "argsort_order"],
    },
}

# knn_decay is deliberately ABSENT: decay-branch agreement is gated
# explicitly by the knn_branch equality in the knn_access source
# search below (a branch change must recompute knn even when every
# scalar agrees); a dead "knn_decay_branch" token here would silently
# drop that parameter from every dependency list.
_PARAM_TO_DEP = {
    "cutoff": "membership_cutoff",
    "d_weights": "kept_weights",
    "gravity_beta": "gravity_beta",
    "gravity_plateau": "gravity_plateau",
    "gravity_logistic_midpoint": "gravity_logistic_midpoint",
    "gravity_growth_rate": "gravity_growth_rate",
    "knn_weights": "knn_weights",
}


def knn_branch(knn_decay):
    if knn_decay == "exponential":
        return "exponential"
    if knn_decay == "logistic":
        return "logistic"
    return "other"


def output_dependencies(knn_decay):
    """The effective dependency set per output for a decay branch."""
    b = knn_branch(knn_decay)
    deps = {k: list(v) for k, v in OUTPUT_DEPENDENCIES.items()
            if k != "knn_access"}
    deps["knn_access"] = list(OUTPUT_DEPENDENCIES["knn_access"][b])
    return deps


def metric_identity(d_weights, gravity_beta, gravity_plateau,
                    gravity_logistic_midpoint, gravity_growth_rate,
                    knn_decay, knn_weights, cutoff,
                    profile_id="legacy_default_profile"):
    """Typed MetricID over exactly the fold's real dependencies.

    Every float enters by its IEEE-754 bits (1-ULP sensitivity), the
    decay branch as a string, the coefficient vector as an ordered
    list of bit values, and the dtype plan plus the engine fold's
    bytes as compiler identity.  Provenance has no field to enter.
    """
    # dtype identity is the SOURCE arrays' dtype: the f64 working copies
    # below are the fold's numerical convention, not the admission
    # identity
    w_src = np.asarray(d_weights)
    k_src = np.asarray(knn_weights)
    w = np.ascontiguousarray(d_weights, dtype=np.float64)
    k = np.ascontiguousarray(knn_weights, dtype=np.float64)
    return _identity([
        ("schema_version", METRIC_REUSE_SCHEMA),
        ("profile_id", str(profile_id)),
        ("dest_metric_weights_bits", [float(v) for v in w.ravel()]),
        ("gravity_beta_bits", float(gravity_beta)),
        ("gravity_plateau_bits", float(gravity_plateau)),
        ("gravity_logistic_midpoint_bits",
         float(gravity_logistic_midpoint)),
        ("gravity_growth_rate_bits", float(gravity_growth_rate)),
        ("knn_decay_branch", str(knn_decay)),
        ("knn_weights_bits", [float(v) for v in k.ravel()]),
        ("cutoff_bits", float(cutoff)),
        ("dest_weights_dtype", str(np.dtype(w_src.dtype))),
        ("knn_weights_dtype", str(np.dtype(k_src.dtype))),
        ("engine_fold_sha256", _engine_sha256()),
    ])


def plan_replay(jobs, allow_split):
    """Private replay plan for metric jobs sharing one trace.

    ``jobs``: list of dicts with the eight metric inputs (the
    ``metric_identity`` keys).  Returns per-job plans:

    - identical MetricID -> mode "bundle", every output reuses the
      first job that produced it
    - distinct MetricID with ``allow_split`` and an ACCEPTED
      characterization: each output reuses an earlier canonical job
      exactly when that job agrees on EVERY dependency in the output's
      branch map; the outputs without a source run their split folds
      (all four without a source -> one full combined fold)
    - distinct MetricID with ``allow_split=False`` (or a rejected
      characterization): one full combined fold per distinct MetricID
      — the safe trace-only continuation

    Every distinct-MetricID job becomes canonical for its outputs
    (later jobs may reuse them).
    """
    plans = []
    canonical = []
    for job in jobs:
        mid = metric_identity(
            job["d_weights"], job["gravity_beta"],
            job["gravity_plateau"], job["gravity_logistic_midpoint"],
            job["gravity_growth_rate"], job["knn_decay"],
            job["knn_weights"], job["cutoff"])
        bundle = next((k for k, c in enumerate(canonical)
                       if c["metric_id"] == mid), None)
        if bundle is not None:
            plans.append({"metric_id": mid, "mode": "bundle",
                          "source": bundle,
                          "outputs": {out: bundle for out in
                                      ("reach", "gravity_exponential",
                                       "gravity_logistic",
                                       "knn_access")},
                          "split_folds": []})
            continue
        entry = {"metric_id": mid, "mode": "full_fold", "source": None,
                 "outputs": {}, "split_folds": []}
        if allow_split:
            deps = output_dependencies(job["knn_decay"])
            plan_outs = {}
            for out in ("reach", "gravity_exponential",
                        "gravity_logistic", "knn_access"):
                need = deps[out]
                src = next((k for k, c in enumerate(canonical)
                            if (out != "knn_access"
                                or knn_branch(c["job"]["knn_decay"])
                                == knn_branch(job["knn_decay"]))
                            and all(_job_agrees(job, c["job"], p)
                                    for p in _PARAM_TO_DEP
                                    if _PARAM_TO_DEP[p] in need)),
                           None)
                if src is None:
                    entry["split_folds"].append(
                        "_%s_only" % out)
                else:
                    plan_outs[out] = src
            if len(entry["split_folds"]) == 4:
                # nothing to reuse: one full combined fold, not four
                # separate extractions
                entry["split_folds"] = []
            else:
                entry["mode"] = "split"
                entry["outputs"] = plan_outs
        canonical.append({"metric_id": mid, "job": job})
        plans.append(entry)
    return plans


def _job_agrees(a, b, param):
    """Typed agreement of one metric parameter between two jobs.

    Scalars agree by IEEE-754 bits, not by value equality (C12-NOTE-2
    repair): -0.0 does not agree with +0.0, consistent with the
    MetricID's signed-zero law; distinct NaN payloads disagree (no
    reuse); identical-bit inputs agree.  Vectors agree by shape and
    bytes.
    """
    va, vb = a[param], b[param]
    if hasattr(va, "__len__") or hasattr(vb, "__len__"):
        ca, cb = np.ascontiguousarray(va), np.ascontiguousarray(vb)
        return ca.shape == cb.shape and ca.tobytes() == cb.tobytes()
    return (np.float64(va).tobytes() == np.float64(vb).tobytes())


# ---- C12 generated split-fold variants (do not edit) ----
@_split_njit
def _reach_only(d_distance, d_weights, cutoff, gravity_beta, gravity_plateau, gravity_logistic_midpoint, knn_gravity_plateau, knn_weights, knn_gravity_beta, knn_decay, knn_gravity_logistic_midpoint, gravity_growth_rate, knn_gravity_growth_rate):
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]
    reach = n_weights.sum()
    return reach
@_split_njit
def _gravity_exponential_only(d_distance, d_weights, cutoff, gravity_beta, gravity_plateau, gravity_logistic_midpoint, knn_gravity_plateau, knn_weights, knn_gravity_beta, knn_decay, knn_gravity_logistic_midpoint, gravity_growth_rate, knn_gravity_growth_rate):
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]
    gravity_exponential = (n_weights / np.exp(gravity_beta * np.maximum(0, n_distances - gravity_plateau))).sum()
    return gravity_exponential
@_split_njit
def _gravity_logistic_only(d_distance, d_weights, cutoff, gravity_beta, gravity_plateau, gravity_logistic_midpoint, knn_gravity_plateau, knn_weights, knn_gravity_beta, knn_decay, knn_gravity_logistic_midpoint, gravity_growth_rate, knn_gravity_growth_rate):
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]
    gravity_logistic = (n_weights * (1 - 1 / (1 + np.exp(-gravity_growth_rate * (n_distances - gravity_plateau - gravity_logistic_midpoint))))).sum()
    return gravity_logistic
@_split_njit
def _knn_access_only(d_distance, d_weights, cutoff, gravity_beta, gravity_plateau, gravity_logistic_midpoint, knn_gravity_plateau, knn_weights, knn_gravity_beta, knn_decay, knn_gravity_logistic_midpoint, gravity_growth_rate, knn_gravity_growth_rate):
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]
    knn = min(n_distances.shape[0], knn_weights.shape[0])
    if knn == 0:
        knn_access = 0.0
    else:
        idx                         = np.argsort(n_distances) # we sort the distances to find the nearest neighbors. We can then apply the knn_weights to the sorted weights and distances.
        n_distances_sorted          = n_distances[idx]
        n_weights_sorted            = n_weights[idx]

        n_distances_cropped         = n_distances_sorted[:knn]
        n_weights_cropped           = n_weights_sorted[:knn]
        knn_coefficients_cropped    = knn_weights[:knn]

        if knn_decay == "exponential":
            knn_access      = ((knn_coefficients_cropped * n_weights_cropped) * np.exp(-gravity_beta * np.maximum(0, n_distances_cropped-gravity_plateau))).sum()
        elif knn_decay == "logistic":
             knn_access      = ((knn_coefficients_cropped * n_weights_cropped) * (1 - 1 / (1 + np.exp(-gravity_growth_rate * (n_distances_cropped - gravity_plateau - gravity_logistic_midpoint))))).sum()
        else:
            knn_access = (knn_coefficients_cropped * n_weights_cropped).sum()
    return knn_access
# ---- end C12 generated region ----
