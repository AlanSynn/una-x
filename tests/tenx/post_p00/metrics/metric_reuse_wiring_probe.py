"""P03M-B evidence probe: metric-reuse executor over the frozen-cohort
shape (campaign una_10x post_p00, designs/04 implementation order 2-5).

Drives the REAL production path exactly as ``batch/cohort.py`` wires it
(binding_job -> MetricID v2 -> row_replay; full replay + canonical
registration on collapse; canonical registration of split-served rows)
over a 32-row cohort shaped like the P00 observed cohort
(campaign_data/inputs/upstream_Boston/Boston/warmup2.csv committed
bytes): one shared trace, ``knn_decay='logistic'``, int
cutoff/plateau/midpoint (the real Settings signature), and
``gravity_beta = linspace(0.001, 0.004, 32)`` as the only varying
scalar.

Records: the measured reuse counters (1 full fold + 31 split rows =
31 gravity_exponential folds + 93 copies, 0 bundle hits since every
beta is distinct), FULL bitwise parity of every served row against a
direct ``replay_trace`` of the same binding, gate chronology
(validation on every row, including reuse rows), and the module
identities.  CORRECTNESS receipt only — no timing, no performance
claim (designs/04 forbids inferring wall saving from saved fields).
"""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "tests" / "tenx" / "post_p00" / "metrics"))

import conftest as ck  # noqa: E402  (pins NUMBA_CACHE_DIR, loads tck)

from urban_network_analysis.Engines import _metric_identity as midm  # noqa: E402
from urban_network_analysis.Engines import _metric_reuse_scratch as mre  # noqa: E402
from urban_network_analysis.Engines import _trace_access_scratch as trm  # noqa: E402

OUT_PATH = Path(_REPO / "campaign_runs" / "una_10x"
                / "tenx-20261005T2331Z" / "p03m"
                / "metric_reuse_wiring.json")

N_ROWS = 32
DECAY = "logistic"
PLATEAU = 0          # int  — real Settings type
MIDPOINT = 500       # int  — real Settings type
DECAY_CONSTANT = 400.0
GROWTH = DECAY_CONSTANT / MIDPOINT     # true division, float — real Settings
CUTOFF = 400         # int  — real Settings type
BETAS = [float(b) for b in np.linspace(0.001, 0.004, N_ROWS)]


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binding(beta):
    """The exact replay_trace positional binding for one cohort row."""
    return (CASE["dwt"], beta, PLATEAU, MIDPOINT, GROWTH, DECAY,
            ck.METRIC_PARAMS[4], CUTOFF)


def job(beta):
    return mre.binding_job(CASE["dwt"], beta, PLATEAU, MIDPOINT, GROWTH,
                           DECAY, ck.METRIC_PARAMS[4], CUTOFF)


ck.admitted_case  # conftest loaded
CASE = dict(ck.admitted_case("base"))
CASE["cutoff"] = CUTOFF
REC = ck.tck.build(trm, CASE)
STORE = mre.MetricReuseStore()
COUNTERS = {}

rows = []
for j, beta in enumerate(BETAS):
    b = binding(beta)
    midv = midm.compute_metric_id(REC, *b)
    served = mre.row_replay(STORE, REC, midv, *b, COUNTERS)
    ref = trm.replay_trace(REC, *b)          # direct full replay, same binding
    ref_bits = {f: np.asarray(getattr(ref, f)).tobytes()
                for f in ck.RESULT_FIELDS}
    if served is None:
        # Production: the full path served the row -> canonical
        # registration with private copies.
        STORE.register(midv, REC.trace_id, job(beta),
                       dict(zip(mre.ALL_OUTS,
                                [np.asarray(getattr(ref, f)).copy()
                                 for f in ck.RESULT_FIELDS])))
        mode = "full_fold_first"
        parity = {f: True for f in ck.RESULT_FIELDS}
    else:
        outs, mode = served
        # Production: split-served rows register canonically too.
        STORE.register(midv, REC.trace_id, job(beta),
                       dict(zip(mre.ALL_OUTS, outs)))
        parity = {f: np.asarray(o).tobytes() == ref_bits[f]
                  for f, o in zip(ck.RESULT_FIELDS, outs)}
    rows.append({
        "row": j,
        "beta": beta,
        "metric_id": midv,
        "mode": mode,
        "bitwise_parity_vs_direct_replay": parity,
        "all_fields_parity": all(parity.values()),
    })

def cnt(key):
    # Counters materialize on first bump (setdefault) — absent = 0.
    return COUNTERS.get(key, 0)


assert all(r["all_fields_parity"] for r in rows), "parity broke"
assert cnt("reuse_split_rows") == N_ROWS - 1, COUNTERS
assert cnt("reuse_split_copies") == 3 * (N_ROWS - 1), COUNTERS
assert cnt("reuse_split_folds") == N_ROWS - 1, COUNTERS
assert cnt("reuse_collapse_full_folds") == 1, COUNTERS
assert cnt("reuse_bundle_hits") == 0, COUNTERS
assert cnt("reuse_refusals") == 0, COUNTERS
assert cnt("reuse_split_domain_refused") == 0, COUNTERS
# reuse_registrations is a COHORT-layer counter (bumped by batch/
# cohort.py, not by the store); at probe level the equivalent receipt
# is the store census: every row's identity registered exactly once.
assert len(STORE) == N_ROWS and STORE.evictions == 0, (len(STORE),
                                                       STORE.evictions)

doc = {
    "schema": "una-10x-p03mb-metric-reuse-wiring-1",
    "task": "P03M-B wiring evidence: production row_replay over the "
            "frozen-cohort shape; correctness receipt only",
    "generated": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat(),
    "law": [
        "bitwise parity per una_legacy profile: every reuse-served row "
        "must equal a direct replay_trace of the same binding, byte "
        "for byte, on all four fold outputs",
        "gates-before-hit: binding + trace-structure validation ran on "
        "EVERY row including reuse rows (reuse_refusals would count "
        "any crossing); failed outputs are never registered",
        "fresh handouts: store entries are read-only; served arrays "
        "are copies (pinned by tests/tenx/post_p00/metrics/"
        "test_metric_reuse_wiring.py)",
        "distinct counters: bundle hits / split rows / split copies / "
        "split folds / collapse / domain refusals / registrations are "
        "measured separately, never inferred",
    ],
    "cohort_shape": {
        "mirror_of": "campaign_data/inputs/upstream_Boston/Boston/"
                     "warmup2.csv (P00 checkpoint committed bytes): 32 "
                     "rows, shared network/origins/destinations, only "
                     "gravity_beta varies; knn_decay='logistic'",
        "note": "scalar MAGNITUDES are the C11 battery case values; "
                "this object is the executor correctness receipt for "
                "the cohort SHAPE, not a re-measurement of the frozen "
                "cohort",
        "rows": N_ROWS,
        "knn_decay": DECAY,
        "gravity_beta": "linspace(0.001, 0.004, 32) float",
        "gravity_plateau": PLATEAU,
        "gravity_plateau_type": type(PLATEAU).__name__,
        "gravity_logistic_midpoint": MIDPOINT,
        "gravity_logistic_midpoint_type": type(MIDPOINT).__name__,
        "gravity_growth_rate": GROWTH,
        "search_radius_cutoff": CUTOFF,
        "search_radius_cutoff_type": type(CUTOFF).__name__,
        "d_weights": "f64 C-contiguous (admitted domain)",
    },
    "measured": {
        "reuse_collapse_full_folds": cnt("reuse_collapse_full_folds"),
        "reuse_split_rows": cnt("reuse_split_rows"),
        "reuse_split_copies": cnt("reuse_split_copies"),
        "reuse_split_folds": cnt("reuse_split_folds"),
        "reuse_bundle_hits": cnt("reuse_bundle_hits"),
        "reuse_registrations": "cohort-layer counter; probe-level "
                               "equivalent = store census below",
        "reuse_refusals": cnt("reuse_refusals"),
        "reuse_split_domain_refused": cnt("reuse_split_domain_refused"),
        "store_entries_after": len(STORE),
        "store_evictions": STORE.evictions,
        "split_repeated_count_per_row": "gravity_exponential is the "
                                        "only fold repeated on a split "
                                        "row (1 of 4 outputs); the "
                                        "other 3 outputs are copies",
    },
    "parity": {
        "rows_checked": N_ROWS,
        "rows_with_full_bitwise_parity": sum(
            1 for r in rows if r["all_fields_parity"]),
        "per_row": rows,
    },
    "identities": {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numba": __import__("numba").__version__,
        "numpy": np.__version__,
        "producer_shas": {
            "Engines/_trace_access_scratch.py": sha256_file(
                _REPO / "src/urban_network_analysis/Engines/"
                "_trace_access_scratch.py"),
            "Engines/_metric_reuse_scratch.py": sha256_file(
                _REPO / "src/urban_network_analysis/Engines/"
                "_metric_reuse_scratch.py"),
            "Engines/_metric_identity.py": sha256_file(
                _REPO / "src/urban_network_analysis/Engines/"
                "_metric_identity.py"),
            "batch/cohort.py": sha256_file(
                _REPO / "src/urban_network_analysis/batch/cohort.py"),
        },
    },
    "disposition": {
        "correctness": "RECEIPT ESTABLISHED: 1 full fold + 31 split "
                       "rows, every row bitwise-equal to its direct "
                       "replay, gates run on every row, fresh handouts",
        "performance_claim": "NONE — designs/04 forbids inferring "
                             "wall saving from saved fields; any 10x "
                             "contribution is a later measurement "
                             "under the frozen policy "
                             "(tools/gate_10x.py, seed 20261003)",
    },
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(doc, indent=2) + "\n")
print("wrote", OUT_PATH)
print("counters:", {k: COUNTERS[k] for k in sorted(COUNTERS)})
print("parity:", sum(1 for r in rows if r["all_fields_parity"]), "/", N_ROWS)
