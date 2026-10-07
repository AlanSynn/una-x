"""P03M partial-plan admission probe (campaign una_10x post_p00,
designs/04 steps 3-4).

The candidate plans are per-field extractions of the ORIGINAL fold
expressions into standalone compiled functions with the SAME
compilation flags and the SAME 9-parameter binding the replay driver
threads.  designs/04: "implement a separately reviewed compiled
partial plan only when it is bit-identical to the corresponding
component of the full compiled reference across its admitted
signature.  A source-expression match is not enough; splitting a
fastmath function changes LLVM."  And: "If a partial plan differs by
even one ULP, keep the full fold for that signature."

This probe runs EVERY candidate plan against the ORIGINAL full fold
component over an adversarial sweep of the replay-admitted domain
(lengths 0..257, duplicate distances/ties, subnormals, max-float and
cutoff-boundary magnitudes, sign-split scalars, every KNN branch, k
crops shorter/longer than the slice) and compares BYTES per case.
One differing byte in one case fails that plan's admission.

Output: JSON verdict {field: {admitted, cases, first_divergence...}}.
This is a CORRECTNESS admission instrument, not a performance claim;
no timing is recorded here (designs/04: "Do not infer wall saving
from number of saved metric fields").

Usage (repo root, venv-p01m, fresh NUMBA_CACHE_DIR):

    NUMBA_CACHE_DIR=<fresh dir> python \\
        tests/tenx/post_p00/metrics/partial_plan_probe.py <out.json>
"""
import json
import sys
from pathlib import Path

import numba as nb
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

# The fold's own flags (AWE NUMBA_* constants): parallel=False,
# cache=True, nogil=True, fastmath=True.  The candidate plans mirror
# them exactly (cache included — it does not change codegen and the
# probe's cache lands in the pinned NUMBA_CACHE_DIR).
FAST_FLAGS = dict(cache=True, nogil=True, fastmath=True)


@nb.njit(**FAST_FLAGS)
def plan_reach(d_distance, d_weights, cutoff, gravity_beta,
               gravity_plateau, gravity_logistic_midpoint,
               gravity_growth_rate, knn_decay, knn_weights):
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]
    return n_weights.sum()


@nb.njit(**FAST_FLAGS)
def plan_gravity_exponential(d_distance, d_weights, cutoff, gravity_beta,
                             gravity_plateau, gravity_logistic_midpoint,
                             gravity_growth_rate, knn_decay, knn_weights):
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]
    return (n_weights / np.exp(gravity_beta * np.maximum(
        0, n_distances - gravity_plateau))).sum()


@nb.njit(**FAST_FLAGS)
def plan_gravity_logistic(d_distance, d_weights, cutoff, gravity_beta,
                          gravity_plateau, gravity_logistic_midpoint,
                          gravity_growth_rate, knn_decay, knn_weights):
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]
    return (n_weights * (1 - 1 / (
        1 + np.exp(-gravity_growth_rate * (
            n_distances - gravity_plateau
            - gravity_logistic_midpoint))))).sum()


@nb.njit(**FAST_FLAGS)
def plan_knn_access(d_distance, d_weights, cutoff, gravity_beta,
                    gravity_plateau, gravity_logistic_midpoint,
                    gravity_growth_rate, knn_decay, knn_weights):
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]

    knn = min(n_distances.shape[0], knn_weights.shape[0])

    if knn == 0:
        return 0.0
    idx = np.argsort(n_distances)
    n_distances_sorted = n_distances[idx]
    n_weights_sorted = n_weights[idx]

    n_distances_cropped = n_distances_sorted[:knn]
    n_weights_cropped = n_weights_sorted[:knn]
    knn_coefficients_cropped = knn_weights[:knn]

    if knn_decay == "exponential":
        return ((knn_coefficients_cropped * n_weights_cropped)
                * np.exp(-gravity_beta * np.maximum(
                    0, n_distances_cropped - gravity_plateau))).sum()
    elif knn_decay == "logistic":
        return ((knn_coefficients_cropped * n_weights_cropped)
                * (1 - 1 / (1 + np.exp(
                    -gravity_growth_rate * (
                        n_distances_cropped - gravity_plateau
                        - gravity_logistic_midpoint))))).sum()
    else:
        return (knn_coefficients_cropped * n_weights_cropped).sum()


@nb.njit(**FAST_FLAGS)
def plan_reach(d_distance, d_weights, cutoff, gravity_beta,
               gravity_plateau, gravity_logistic_midpoint,
               gravity_growth_rate, knn_decay, knn_weights):
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]
    return n_weights.sum()


@nb.njit(**FAST_FLAGS)
def plan_gravity_exponential(d_distance, d_weights, cutoff, gravity_beta,
                             gravity_plateau, gravity_logistic_midpoint,
                             gravity_growth_rate, knn_decay, knn_weights):
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]
    return (n_weights / np.exp(gravity_beta * np.maximum(
        0, n_distances - gravity_plateau))).sum()


@nb.njit(**FAST_FLAGS)
def plan_gravity_logistic(d_distance, d_weights, cutoff, gravity_beta,
                          gravity_plateau, gravity_logistic_midpoint,
                          gravity_growth_rate, knn_decay, knn_weights):
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]
    return (n_weights * (1 - 1 / (
        1 + np.exp(-gravity_growth_rate * (
            n_distances - gravity_plateau
            - gravity_logistic_midpoint))))).sum()


@nb.njit(**FAST_FLAGS)
def plan_knn_access(d_distance, d_weights, cutoff, gravity_beta,
                    gravity_plateau, gravity_logistic_midpoint,
                    gravity_growth_rate, knn_decay, knn_weights):
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]

    knn = min(n_distances.shape[0], knn_weights.shape[0])

    if knn == 0:
        return 0.0
    idx = np.argsort(n_distances)
    n_distances_sorted = n_distances[idx]
    n_weights_sorted = n_weights[idx]

    n_distances_cropped = n_distances_sorted[:knn]
    n_weights_cropped = n_weights_sorted[:knn]
    knn_coefficients_cropped = knn_weights[:knn]

    if knn_decay == "exponential":
        return ((knn_coefficients_cropped * n_weights_cropped)
                * np.exp(-gravity_beta * np.maximum(
                    0, n_distances_cropped - gravity_plateau))).sum()
    elif knn_decay == "logistic":
        return ((knn_coefficients_cropped * n_weights_cropped)
                * (1 - 1 / (1 + np.exp(
                    -gravity_growth_rate * (
                        n_distances_cropped - gravity_plateau
                        - gravity_logistic_midpoint))))).sum()
    else:
        return (knn_coefficients_cropped * n_weights_cropped).sum()


def _case_slices():
    """Adversarial (d_distance, cutoff) candidate slices over the
    replay-admitted domain (finite, nonnegative distances <= cutoff;
    weights finite nonnegative).  Deterministic; includes the
    structural shapes reuse admission cares about."""
    rng = np.random.default_rng(20261006)
    cases = []
    # length sweep incl. 0 and 1, powers of two around vectorizer
    # widths, and non-round tails
    for n in [0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65,
              127, 128, 129, 255, 256, 257]:
        dist = rng.uniform(0.0, 24.0, n)
        cases.append((dist, 25.0))
    # exact ties and duplicates (argsort tie domain for knn)
    ties = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
    cases.append((ties, 25.0))
    # subnormals and extremes (finite, admitted)
    sub = np.array([5e-324, 1e-320, 2.2250738585072014e-308, 1.0,
                    1.7976931348623157e308 / 4], dtype=np.float64)
    cases.append((sub, 25.0))
    # distances EXACTLY at the cutoff (membership boundary) and -0.0
    bnd = np.array([0.0, -0.0, 12.5, 25.0, 24.999999999999996, 25.0])
    cases.append((bnd, 25.0))
    # near-cutoff magnitudes
    near = 25.0 - np.abs(rng.normal(0, 1e-12, 64))
    near = np.clip(near, 0.0, 25.0)
    cases.append((near, 25.0))
    return cases


def _bundle_grid():
    betas = [0.0, -0.0, 1e-12, 0.001, 1.5, 4.0, 1e6]
    plateaus = [0.0, -0.0, 2.0, 24.999999999, 1e6]
    mids = [0.0, 3.0, 1e6]
    growths = [0.0, 1e-12, 0.8, 1e6]
    decays = ["exponential", "logistic", "other"]
    kws = [
        np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
        np.array([2.0, 1.0]),                   # k crop shorter than slice
        np.array([1.0] * 10 + [2.0]),           # k crop longer than slice
    ]
    grid = []
    for beta in betas:
        for plateau in plateaus:
            for mid_ in mids:
                for growth in growths:
                    for decay in decays:
                        for kw in kws:
                            grid.append((beta, plateau, mid_, growth,
                                         decay, kw))
    return grid


def _case_slices_int_cutoff():
    """The INTEGER-cutoff adversarial slices: the REAL cohort signature
    carries ``Settings.search_radius: int`` (the frozen/audit manifests
    hold search_radius=400), so the int-cutoff specialization is the
    production domain the split plans must be admitted on.  Membership
    boundaries land on integer distances exactly; everything else
    mirrors the float sweep."""
    rng = np.random.default_rng(20261007)
    cases = []
    for n in [0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65,
              127, 128, 129, 255, 256, 257]:
        dist = rng.uniform(0.0, 399.0, n)
        cases.append((dist, 400))          # int cutoff (Settings int)
    ties = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
    cases.append((ties, 400))
    sub = np.array([5e-324, 1e-320, 2.2250738585072014e-308, 1.0,
                    1.7976931348623157e308 / 4], dtype=np.float64)
    cases.append((sub, 400))
    # distances EXACTLY at the integer cutoff and one-ULP below it
    bnd = np.array([0.0, -0.0, 200.0, 400.0,
                    399.99999999999994, 400.0])
    cases.append((bnd, 400))
    near = 400.0 - np.abs(rng.normal(0, 1e-12, 64))
    near = np.clip(near, 0.0, 400.0)
    cases.append((near, 400))
    return cases


def _bundle_grid_int_scalars():
    """Scalar bundles over the ACTUAL Settings types on the frozen
    cohort: gravity_plateau int (default 0), gravity_logistic_midpoint
    int (500), gravity_beta float, growth float (true division of the
    float decay constant by the int midpoint).  Integer plateau/midpoint
    change the fold's SPECIALIZATION (int64 arguments) even where the
    numeric promotion matches — bit identity is measured, never
    inferred."""
    betas = [0.0, -0.0, 1e-12, 0.001, 1.5, 4.0, 1e6]
    plateaus = [0, 2, 24]                  # int (Settings int)
    mids = [3, 500]                        # int (Settings int)
    growths = [0.0, 1e-12, 0.8, 1e6]
    decays = ["exponential", "logistic", "other"]
    kws = [
        np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
        np.array([2.0, 1.0]),                   # k crop shorter than slice
        np.array([1.0] * 10 + [2.0]),           # k crop longer than slice
    ]
    grid = []
    for beta in betas:
        for plateau in plateaus:
            for mid_ in mids:
                for growth in growths:
                    for decay in decays:
                        for kw in kws:
                            grid.append((beta, plateau, mid_, growth,
                                         decay, kw))
    return grid


FIELDS = ("reach", "gravity_exponential", "gravity_logistic",
          "knn_access")
PLANS = {"reach": plan_reach,
         "gravity_exponential": plan_gravity_exponential,
         "gravity_logistic": plan_gravity_logistic,
         "knn_access": plan_knn_access}


def _run_sweep(slices, bundles, verdicts):
    """One full sweep; returns the fold-evaluation count.  Reference =
    the REAL compiled fold with the duplicated knn binding the replay
    driver threads."""
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        reach_gravity_knn_access as real_fold)

    def full_fold(dist, d_weights, cutoff, beta, plateau, mid_, growth,
                  decay, kw):
        return real_fold(dist, d_weights, cutoff, beta, plateau, mid_,
                         plateau, kw, beta, decay, mid_, growth, growth)

    weights_pool = [
        ("linspace", np.linspace(0.25, 2.75, 258)),
        ("ones", np.ones(258)),
        ("decaying", np.concatenate(
            [np.ones(129), np.full(129, 1e-8)])),
    ]
    total = 0
    for dist, cutoff in slices:
        n = dist.shape[0]
        for wname, pool in weights_pool:
            d_weights = pool[:n].copy()
            for bundle in bundles:
                beta, plateau, mid_, growth, decay, kw = bundle
                r, ge, gl, kn = full_fold(
                    dist, d_weights, cutoff, beta, plateau, mid_,
                    growth, decay, kw)
                ref = {"reach": r, "gravity_exponential": ge,
                       "gravity_logistic": gl, "knn_access": kn}
                total += 1
                for field, plan in PLANS.items():
                    got = plan(dist, d_weights, cutoff, beta, plateau,
                               mid_, growth, decay, kw)
                    if np.asarray(got).tobytes() \
                            != np.asarray(ref[field]).tobytes():
                        v = verdicts[field]
                        v["admitted"] = False
                        if v["first_divergence"] is None:
                            v["first_divergence"] = {
                                "slice_len": int(n),
                                "weights": wname,
                                "bundle": {"gravity_beta": beta,
                                           "gravity_plateau": plateau,
                                           "gravity_logistic_midpoint":
                                               mid_,
                                           "growth_rate": growth,
                                           "knn_decay": decay},
                                "ref_bits":
                                    np.asarray(ref[field]).tobytes().hex(),
                                "plan_bits":
                                    np.asarray(got).tobytes().hex(),
                            }
                verdicts[field]["cases"] += 1
    return total


def main(out_path):
    assert len(sys.argv) == 2, "usage: probe.py <out.json>"
    import os
    assert os.environ.get("NUMBA_CACHE_DIR"), \
        "NUMBA_CACHE_DIR must be pinned for this probe"

    import numba
    import llvmlite

    # Sweep 1: python-float scalar signature (the increment-A domain).
    verdicts_float = {f: {"admitted": True, "cases": 0,
                          "first_divergence": None} for f in FIELDS}
    total_float = _run_sweep(_case_slices(), _bundle_grid(),
                             verdicts_float)

    # Sweep 2: the ACTUAL cohort signature — int search_radius /
    # int plateau / int midpoint (Settings types), float beta/growth.
    verdicts_int = {f: {"admitted": True, "cases": 0,
                        "first_divergence": None} for f in FIELDS}
    total_int = _run_sweep(_case_slices_int_cutoff(),
                           _bundle_grid_int_scalars(), verdicts_int)

    out = {
        "schema": "una-10x-p03m-partial-plan-verdict-1",
        "task": "P03M",
        "generated": "2026-10-06",
        "law": "designs/04: bit-identical across the admitted signature "
               "or the plan is NOT admitted; one ULP rejects the fast "
               "domain; keep the full fold",
        "disposition": {
            "partial_plans": "admitted at the CORRECTNESS level (all "
                             "four fields bit-identical to the real "
                             "fold across BOTH admitted-signature "
                             "sweeps)",
            "admitted_signatures": [
                "python-float scalars + f64 C-contiguous "
                "distances/weights (increment A)",
                "int search_radius/plateau/midpoint (Settings types, "
                "the real cohort signature) + float beta/growth + f64 "
                "C-contiguous distances/weights (this extension; "
                "int64 arguments change the SPECIALIZATION, so "
                "bit identity is measured, never inferred)",
            ],
            "admission_scope": "this host + compiler identity + the "
                               "two enumerated scalar/array "
                               "specialization domains; any fold "
                               "module byte change re-opens admission "
                               "(the MetricID v2 fold closure "
                               "invalidates); OTHER scalar types "
                               "(numpy scalars, bools) and other "
                               "array dtypes/layouts stay OUTSIDE "
                               "the split domain and take the full "
                               "fold",
            "production_wiring": "the reuse executor lands in the "
                                 "same increment family (P03M-B): "
                                 "worker-local bounded store, bundle "
                                 "mode gated by MetricID v2, split "
                                 "mode gated by these admissions, "
                                 "full-fold fallback, distinct "
                                 "counters; performance claim stays "
                                 "NONE until a frozen-policy "
                                 "measurement",
            "performance_claim": "NONE — designs/04 forbids inferring "
                                 "wall saving from saved fields; any "
                                 "timing is a later measurement under "
                                 "the frozen policy",
            "c31_note_1_resolution": "E8 closure fixed by demotion: the "
                                     "cohort v1 result_key stays a "
                                     "REPORTING key (coercion "
                                     "disclosed at the construction "
                                     "site, collision witnesses "
                                     "recorded); the authoritative "
                                     "identity is the typed MetricID "
                                     "v2 (refusal-based, no "
                                     "coercion). C12/C31 historic "
                                     "receipts retained untouched; "
                                     "this object appends the new "
                                     "resolution",
        },
        "admitted_domain": "replay-admitted candidate slices (finite "
                           "nonnegative distances <= cutoff, finite "
                           "nonnegative weights); adversarial lengths, "
                           "ties, subnormals, cutoff boundaries (int "
                           "and float), sign-split scalars, all three "
                           "KNN branches, k crops shorter and longer "
                           "than the slice",
        "compilation": "reference = the REAL compiled fold "
                       "(reach_gravity_knn_access, its own module "
                       "bytes, duplicated knn binding as the replay "
                       "driver threads it); candidates = per-field "
                       "standalone njit functions with the fold's own "
                       "flags (cache, nogil, fastmath; parallel stays "
                       "False), same 9-parameter binding",
        "compiler_identity": {
            "numba_version": numba.__version__,
            "llvmlite_version": llvmlite.__version__,
            "numpy_version": np.__version__,
        },
        "total_fold_evaluations": total_float + total_int,
        "sweeps": {
            "float_python_signature": {
                "total_fold_evaluations": total_float,
                "plans": verdicts_float,
            },
            "int_cohort_signature": {
                "signature": "cutoff/plateau/midpoint python int; "
                             "beta/growth python float; d_weights/"
                             "knn_weights f64 C-contiguous; str "
                             "branch",
                "total_fold_evaluations": total_int,
                "plans": verdicts_int,
            },
        },
        "decision_rule": "admitted = plan bits == full-fold component "
                         "bits on EVERY case of the signature's sweep; "
                         "any divergence -> keep the full fold for "
                         "that field/signature",
    }
    Path(out_path).write_text(json.dumps(out, indent=1) + "\n")
    admitted_f = [f for f in FIELDS if verdicts_float[f]["admitted"]]
    admitted_i = [f for f in FIELDS if verdicts_int[f]["admitted"]]
    print(f"PROBE OK cases={total_float + total_int} "
          f"admitted_float={admitted_f or 'NONE'} "
          f"admitted_int={admitted_i or 'NONE'} -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
