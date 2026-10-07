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

from . import _metric_identity as _mid
from ._trace_access_scratch import (_frame, _frame_fields, _identity,
                                    binding_refusal,
                                    trace_structure_refusal)

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

    P03M-B hardening (designs/04: the agreement IS the admission): both
    sides close over the MetricID v2 typed frames — the SAME encoder
    that mints the authoritative identity, one law no drift.

    * Scalars agree only at equal dtype AND equal bits: -0.0 never
      agrees with +0.0 (C12-NOTE-2), distinct NaN payloads disagree,
      and an ``int`` cutoff no longer agrees with the equal-valued
      ``float`` (the pre-hardening ``np.float64`` coercion collapsed
      them — the E8 class at split level; the replay refuses that pair
      as ``cutoff_mismatch_with_trace_identity``).
    * Arrays agree only at equal dtype (byte order included), shape,
      layout/writability flags and logical-order bytes — a strided view
      and its C-contiguous twin DISAGREE (they hit different numba
      specializations).
    * Anything either frame refuses to encode counts as disagreement
      (conservative: no reuse, the full fold decides).
    """
    va, vb = a[param], b[param]
    try:
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray) \
                or hasattr(va, "__len__") or hasattr(vb, "__len__"):
            fa, fb = [], []
            _mid.typed_array_frame(fa, "v", np.asarray(va))
            _mid.typed_array_frame(fb, "v", np.asarray(vb))
            return fa == fb
        sa, sb = [], []
        _mid.typed_scalar_frame(sa, "v", va)
        _mid.typed_scalar_frame(sb, "v", vb)
        return sa == sb
    except (TypeError, ValueError):
        return False


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


# ===========================================================================
# P03M-B production executor (designs/04 implementation order 2-5).
#
# The C12 planner above is a pure PLANNER; this section owns the
# worker-local store, the compiled split outer loops and the per-row
# entry point the cohort route calls.  Laws:
#
# * Gates BEFORE any hit: the metric-reuse path runs the EXACT
#   ``replay_trace`` gate sequence (``binding_refusal`` +
#   ``trace_structure_refusal``) on every row, bundle hits included —
#   read/validation/failure chronology survives cache hits.
# * Bundle mode is gated by MetricID v2 (the authoritative typed
#   identity).  A bundle hit copies four FRESH per-row public arrays
#   out of immutable private storage (``register`` normalizes every
#   stored array to read-only).
# * Split mode is admitted ONLY inside the probe-admitted signature
#   domains (tests/tenx/post_p00/metrics/partial_plan_probe.py):
#   f64 native C-contiguous d_weights/knn_weights and python
#   int/float scalars.  Anything else — and any fold-module byte
#   change via the v2 fold closure — falls back to the full fold.
# * Four unsourced outputs collapse to ONE full fold (the C12
#   collapse law): the caller's existing ``replay_trace`` path runs.
# * Failed or refused rows NEVER register (failed outputs stay out
#   of the store).
# * Distinct counters: bundle hits, split rows, copied fields, split
#   folds, collapses, domain refusals, gate refusals, registrations,
#   evictions (designs/04: hit/miss/produced/copy/trace counters are
#   distinct).
# ===========================================================================
ALL_OUTS = ("reach", "gravity_exponential", "gravity_logistic",
            "knn_access")

# The reuse-mode labels the cohort route reports per row.
REUSE_MODE_BUNDLE = "metric_reuse_bundle"
REUSE_MODE_SPLIT = "metric_reuse_split"


@nb.njit(
    cache=NUMBA_CACHE,
    nogil=NUMBA_NOGIL,
)
def _trace_replay_loop_reach(indptr, ids_all, dist_all, d_weights, cutoff,
                             gravity_beta, gravity_plateau,
                             gravity_logistic_midpoint, knn_weights,
                             knn_decay, gravity_growth_rate, reach_out):
    """Split outer loop (designs/04 step 3): the admitted
    ``_reach_only`` extraction per origin, composed exactly like
    ``_trace_replay_loop`` — same fresh per-origin working copies, the
    same duplicated 13-parameter binding, the engine's own flags (the
    loop adds none).  Serial/nogil: no nested prange, no total-CPU
    increase."""
    n_origin = indptr.shape[0] - 1
    for o in range(n_origin):
        lo = indptr[o]
        hi = indptr[o + 1]
        cand_dist = dist_all[lo:hi].copy()
        cand_wts = d_weights[ids_all[lo:hi]].copy()
        reach_out[o] = _reach_only(
            cand_dist, cand_wts, cutoff,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_plateau, knn_weights, gravity_beta, knn_decay,
            gravity_logistic_midpoint, gravity_growth_rate,
            gravity_growth_rate)


@nb.njit(
    cache=NUMBA_CACHE,
    nogil=NUMBA_NOGIL,
)
def _trace_replay_loop_ge(indptr, ids_all, dist_all, d_weights, cutoff,
                          gravity_beta, gravity_plateau,
                          gravity_logistic_midpoint, knn_weights,
                          knn_decay, gravity_growth_rate, ge_out):
    n_origin = indptr.shape[0] - 1
    for o in range(n_origin):
        lo = indptr[o]
        hi = indptr[o + 1]
        cand_dist = dist_all[lo:hi].copy()
        cand_wts = d_weights[ids_all[lo:hi]].copy()
        ge_out[o] = _gravity_exponential_only(
            cand_dist, cand_wts, cutoff,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_plateau, knn_weights, gravity_beta, knn_decay,
            gravity_logistic_midpoint, gravity_growth_rate,
            gravity_growth_rate)


@nb.njit(
    cache=NUMBA_CACHE,
    nogil=NUMBA_NOGIL,
)
def _trace_replay_loop_gl(indptr, ids_all, dist_all, d_weights, cutoff,
                          gravity_beta, gravity_plateau,
                          gravity_logistic_midpoint, knn_weights,
                          knn_decay, gravity_growth_rate, gl_out):
    n_origin = indptr.shape[0] - 1
    for o in range(n_origin):
        lo = indptr[o]
        hi = indptr[o + 1]
        cand_dist = dist_all[lo:hi].copy()
        cand_wts = d_weights[ids_all[lo:hi]].copy()
        gl_out[o] = _gravity_logistic_only(
            cand_dist, cand_wts, cutoff,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_plateau, knn_weights, gravity_beta, knn_decay,
            gravity_logistic_midpoint, gravity_growth_rate,
            gravity_growth_rate)


@nb.njit(
    cache=NUMBA_CACHE,
    nogil=NUMBA_NOGIL,
)
def _trace_replay_loop_knn(indptr, ids_all, dist_all, d_weights, cutoff,
                           gravity_beta, gravity_plateau,
                           gravity_logistic_midpoint, knn_weights,
                           knn_decay, gravity_growth_rate, kn_out):
    n_origin = indptr.shape[0] - 1
    for o in range(n_origin):
        lo = indptr[o]
        hi = indptr[o + 1]
        cand_dist = dist_all[lo:hi].copy()
        cand_wts = d_weights[ids_all[lo:hi]].copy()
        kn_out[o] = _knn_access_only(
            cand_dist, cand_wts, cutoff,
            gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_plateau, knn_weights, gravity_beta, knn_decay,
            gravity_logistic_midpoint, gravity_growth_rate,
            gravity_growth_rate)


_SPLIT_LOOPS = {
    "reach": _trace_replay_loop_reach,
    "gravity_exponential": _trace_replay_loop_ge,
    "gravity_logistic": _trace_replay_loop_gl,
    "knn_access": _trace_replay_loop_knn,
}

_SPLIT_OUT_DTYPE_FIELD = {
    "reach": "reach_dtype",
    "gravity_exponential": "metric_dtype",
    "gravity_logistic": "metric_dtype",
    "knn_access": "metric_dtype",
}


def split_domain_admitted(d_weights, gravity_beta, gravity_plateau,
                          gravity_logistic_midpoint, gravity_growth_rate,
                          knn_weights, cutoff):
    """True only inside the probe-admitted specialization domains:
    f64 native-byte-order C-contiguous ``d_weights``/``knn_weights``
    and python ``int``/``float`` scalars (the int form covers the REAL
    Settings signature: ``search_radius: int``, ``gravity_plateau:
    int``, ``gravity_logistic_midpoint: int``).  Numpy scalar types,
    other dtypes, non-contiguous or read-only views stay OUTSIDE and
    take the full fold."""
    for a in (d_weights, knn_weights):
        if not (isinstance(a, np.ndarray) and a.ndim == 1
                and a.dtype == np.float64 and a.flags.c_contiguous):
            return False
    for s in (cutoff, gravity_beta, gravity_plateau,
              gravity_logistic_midpoint, gravity_growth_rate):
        if type(s) not in (int, float):
            return False
    return True


def binding_job(d_weights, gravity_beta, gravity_plateau,
                gravity_logistic_midpoint, gravity_growth_rate,
                knn_decay, knn_weights, cutoff):
    """The per-row metric binding dict the store's agreement checks
    consume (exactly the ``_PARAM_TO_DEP`` parameter names plus the
    branch selector)."""
    return {"cutoff": cutoff, "d_weights": d_weights,
            "gravity_beta": gravity_beta,
            "gravity_plateau": gravity_plateau,
            "gravity_logistic_midpoint": gravity_logistic_midpoint,
            "gravity_growth_rate": gravity_growth_rate,
            "knn_decay": knn_decay, "knn_weights": knn_weights}


class MetricReuseStore:
    """Worker-local bounded canonical store (designs/04 step 5: the
    WORKER-LOCAL reuse arm — no cross-worker sharing, so no trace-store
    lock is ever held and no lease protocol is needed).  Entries are
    (trace_id, binding, immutable private output copies) keyed by the
    typed MetricID v2; eviction is LRU by entry count.  Handouts are
    always fresh copies; stored arrays are normalized read-only.
    No persistence, no disk, process lifetime only."""

    def __init__(self, max_entries: int = 32):
        self._entries = {}
        self._order = []
        self._max = max(1, int(max_entries))
        self.evictions = 0

    def __len__(self):
        return len(self._order)

    def lookup(self, metric_id):
        """LRU-touched entry for the typed MetricID, or None."""
        entry = self._entries.get(metric_id)
        if entry is not None:
            self._order.remove(metric_id)
            self._order.append(metric_id)
        return entry

    def entries_for_trace(self, trace_id):
        """Canonical entries for ONE trace in LRU (recency) order —
        ``lookup`` reorders the underlying list, so this is NOT
        insertion order; candidate source selection among agreeing
        entries is LRU order (bit-irrelevant: agreeing candidates
        serve identical bytes — pinned by the wiring suite)."""
        return [self._entries[m] for m in self._order
                if self._entries[m]["trace_id"] == trace_id]

    def register(self, metric_id, trace_id, job, outs):
        """Store PRIVATE immutable copies of one canonical row's four
        outputs.  A repeated MetricID is a no-op (the first-arrival
        canonical stays).  Raises nothing."""
        if metric_id in self._entries:
            return
        stored = {}
        for name in ALL_OUTS:
            arr = np.array(outs[name], copy=True)
            arr.flags.writeable = False
            stored[name] = arr
        # designs/04 lease law (P03R F1): the stored binding records
        # the bytes the canonical outputs were computed FROM — the two
        # array-valued fields get private copies, so a caller mutating
        # a previously-passed array in place cannot turn a later
        # agreement check into a stale serve.
        job_copy = dict(job)
        for _k in ("d_weights", "knn_weights"):
            if _k in job_copy:
                job_copy[_k] = np.array(job_copy[_k], copy=True)
        self._entries[metric_id] = {"trace_id": trace_id, "job": job_copy,
                                    "outs": stored}
        self._order.append(metric_id)
        while len(self._order) > self._max:
            evicted = self._order.pop(0)
            del self._entries[evicted]
            self.evictions += 1


def row_replay(store, trace, metric_id, d_weights, gravity_beta,
               gravity_plateau, gravity_logistic_midpoint,
               gravity_growth_rate, knn_decay, knn_weights, cutoff,
               counters):
    """The production per-row metric-reuse entry point.

    Returns ``(outs_tuple, mode)`` with four FRESH public arrays when
    the row is served from the store (bundle or split), or None to run
    the caller's full ``replay_trace`` path unchanged.  The EXACT
    replay gates run BEFORE any hit; refused rows are counted and fall
    through.  ``counters`` is the caller's flat dict; every increment
    uses ``setdefault`` so partial integrations stay honest.
    """
    def _bump(key, n=1):
        counters[key] = counters.get(key, 0) + n

    # 1. Gates first — the identical validation chronology on cache
    #    hits (designs/04: validation still runs for every row).
    refusal = binding_refusal(trace, d_weights, cutoff)
    if refusal is None:
        refusal = trace_structure_refusal(trace)
    if refusal is not None:
        _bump("reuse_refusals")
        return None

    # 2. Bundle mode: the authoritative MetricID v2 hit.
    entry = store.lookup(metric_id)
    if entry is not None:
        if entry["trace_id"] != trace.trace_id:
            # Defensive: v2 binds the trace id, so this cannot happen;
            # refuse rather than trust (fail-safe, counted).
            _bump("reuse_refusals")
            return None
        _bump("reuse_bundle_hits")
        outs = tuple(entry["outs"][f].copy() for f in ALL_OUTS)
        return outs, REUSE_MODE_BUNDLE

    # 3. Split mode: admitted-signature rows only.
    if not split_domain_admitted(d_weights, gravity_beta, gravity_plateau,
                                 gravity_logistic_midpoint,
                                 gravity_growth_rate, knn_weights, cutoff):
        _bump("reuse_split_domain_refused")
        return None

    job = binding_job(d_weights, gravity_beta, gravity_plateau,
                      gravity_logistic_midpoint, gravity_growth_rate,
                      knn_decay, knn_weights, cutoff)
    deps = output_dependencies(knn_decay)
    branch = knn_branch(knn_decay)
    candidates = store.entries_for_trace(trace.trace_id)
    sources = {}
    for out in ALL_OUTS:
        need = set(deps[out])
        for cand in candidates:
            if out == "knn_access" and \
                    knn_branch(cand["job"]["knn_decay"]) != branch:
                continue
            # Membership/distances/casts come from the shared trace
            # (same trace_id); the agreement check covers the map's
            # PARAMETER deps for this output (cutoff/weights/scalars).
            if all(_job_agrees(job, cand["job"], p)
                   for p in _PARAM_TO_DEP if _PARAM_TO_DEP[p] in need):
                sources[out] = cand
                break
    unsourced = [f for f in ALL_OUTS if f not in sources]
    if len(unsourced) == len(ALL_OUTS):
        # Nothing to reuse: ONE full combined fold (the collapse law),
        # not four extractions.
        _bump("reuse_collapse_full_folds")
        return None

    outs = {}
    for out in unsourced:
        arr = np.empty(trace.origin_count,
                       dtype=np.dtype(getattr(
                           trace, _SPLIT_OUT_DTYPE_FIELD[out])))
        _SPLIT_LOOPS[out](trace.indptr, trace.destination_ids,
                          trace.distance_bits, d_weights, cutoff,
                          gravity_beta, gravity_plateau,
                          gravity_logistic_midpoint, knn_weights,
                          knn_decay, gravity_growth_rate, arr)
        outs[out] = arr
        _bump("reuse_split_folds")
    for out, cand in sources.items():
        outs[out] = cand["outs"][out].copy()
        _bump("reuse_split_copies")
    _bump("reuse_split_rows")
    return (outs["reach"], outs["gravity_exponential"],
            outs["gravity_logistic"], outs["knn_access"]), REUSE_MODE_SPLIT
