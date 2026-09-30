"""S01 frozen sweep matrix (task #36) -- pure data + pure functions, no IO.

Frozen at matrix freeze per proposal amendment #2 (custody
proposal_s01_20260928T220026Z_h05_amendment2.md, sha 353f822f...) and h04's
section-6 fold rulings.  Every figure here traces to a governing document or a
recorded tool output; the supervisor (benchmarks/large_e2e/harness/s01_run.py)
imports this module and its tranche gates ENFORCE the recorded counts -- any
deviation is a STOP, never an extended matrix.

Binding sources:
  - BENCHMARKS.md "Screening and sweep" (<=6 W/H configs/family incl. W1H1;
    flow K fixed; queue initially W; writer limit W; K_jobs divisible by the
    largest compared W; >=60 s warm-window target, never truncate a job).
  - DECISION.md "Run budget" (pilots <=2/cell; sweep <=6 configs x 2
    batches/family; one selected config/arm; 4 h campaign default).
  - RESOURCES.md :11 (flow K freeze / labelled explicit profile), :13
    (queue starts W), :15/:17 (load-rejection, no auto-retry).
  - CONTRACT.md :19 (H<K forbidden without schedule proof -- NOT pursued).
  - control.json budgets (C=9, pressure stop, ceiling policy, ledger 14,400).
  - W00 final lease log (custody pin 7d35e910...) chains the single ledger.
"""
from __future__ import annotations

# --- base (resolved: Commit B landed; h04 corroborated) -------------------
from _legacy_paths import LEGACY_WS  # portable-path fix (HARNESS 2026-09-30)
BASE_COMMIT = "6b53a274e8a8d3e7360ab5e1b38f741107ce3358"
BASE_SRC_TREE = "3c59bb0d3f9bc1fba7547be47ec63add88dd3031"

# --- single campaign ledger (amendment #2 SS3; h04 SS6.1) ------------------
# Full 64-hex pin computed fresh in-session 2026-09-28 (shasum -a 256 of the
# live W00 final log bytes; prefix matches the fire-#4 close relay).
W00_CHAIN_PIN = "7d35e9101c46dfda71298521c5a710942b63cdf70377beb885a138d2b1e933b7"
LEDGER = {
    "budget_total_s": 14400.0,
    "w00_final_log": (str(LEGACY_WS) + "/campaign_data/"
                      "w00_lease_log.json"),
    "w00_final_log_sha256": W00_CHAIN_PIN,
    "w00_charged_s": 1991.9,
    "s01_cap_s": 7000.0,          # S01 share cap (under the 7,500 ceiling)
    "s01_ceiling_s": 7500.0,
    "reserve_floor_s": 3500.0,    # BINDING at every tranche gate (Q10/final)
}

# --- resource policy (control.json budgets; W00 parity) --------------------
C_SLOTS = 9
PRESSURE_STOP_BYTES = 1717986918
CEILING_MIB_CAP = 10650
CEILING_FRACTION = 0.80

# --- frozen W/H matrix (amendment #2 SS3 closure arithmetic) ---------------
ACCESSIBILITY_CONFIGS = [
    (1, 1),  # W1H1 floor (BENCHMARKS: W1H1 mandatory member)
    (2, 2),  # W2H2 (=4)
    (3, 3),  # W3H3 (=9)
    (4, 2),  # W4H2 (=8)
    (2, 4),  # W2H4 (=8)
    (9, 1),  # W9H1 (=9) -- largest compared W of the accessibility family
]
# --- flow family (AMD7: D10, VOID-8 fire #8, key r8, log 4b35ea9b) ---------
# The freeze-era flow family (1, 9) + production-default stripes is
# statically infeasible on this frame: the child gate (spec.py:294-313,
# frozen, correct) demands H_numba+K <= C with K = max(1, cpu_count-1) = 9
# under --flow-stripes default, so H >= 9 AND H+9 <= 9 is unsatisfiable.
# Fire #8 refused the pilot verbatim ("H_numba+K = 9+9 = 18 exceeds C=9",
# receipt 725ac8f2); the p1 and confirm flow windows are covered
# identically (confirm chains strongest_feasible_b0 cfgkey).  Observed/B0
# topology on frozen records: NUMBA_NUM_THREADS=10 (L2 call2_flow process
# record; H05 memory_lifetimes pools get_num_threads=10) with engine
# default stripes K=9 (Topology.py:79; L2 logger DIRECT: "Running Flow
# with 9 clusters and 9 threads.") -- over-subscribed 19 threads on 10
# cores, which the equal-resource frame cannot run by construction.
# Per RESOURCES.md :11 ("If default K does not fit the budget, report
# default-profile qualification unavailable or use a separately labelled
# explicit thread profile equally in both arms.  Never lower K silently.")
# the family runs the LABELLED EXPLICIT PROFILE projected from the
# observed posture (H >= K, near-parity 10:9) under the designed gate:
# H=5, K=4 -- H+K = 9 = C exactly (boundary pass; the child comparison is
# strict >), W*(H+K) = 9 <= 9, posture 5 >= 4, charge W*H = 5 <= 9.
# Criterion is SELECTION-VALIDITY, not observational fidelity: both arms
# run the SAME maximal feasible topology, and the deviation from
# observation is recorded (K_FIT).  Explicit-K results are a DISTINCT
# NUMERICAL PROFILE from default-K runs (stripe membership
# range(slot, n_origins, n_threads) depends on K; float sum order
# differs) -- cross-profile output comparison is never
# regression/improvement evidence; the S01 arm comparison is strictly
# within-profile.
FLOW_K = 4
FLOW_CONFIGS = [
    (1, 5),  # W1H5 + explicit K=FLOW_K: the exactly-one flow config
]
ARMS = ["b0", "selected"]
BATCHES_PER_CELL = 2

# --- K-fit record (AMD7 correction of the freeze-era check; h04 D10) --------
# Freeze-era record (intake 5eb7227) ruled "production_default_fits" on
# K <= C alone; the child's actual contract is the CONJUNCTION
# H_numba+K <= C (RESOURCES.md :9 "W*H_numba<=C is insufficient for flow
# ... conservative sum of each job's worst admitted concurrency";
# spec.py:294-313), which makes the default profile infeasible for flow
# on this frame.  Correction authorized by h04's AMD7 design PASS (Set A,
# H=5, K=4); the labelled explicit profile is taken per RESOURCES.md :11.
K_FIT = {
    "branch": "labelled_explicit_profile",
    "labelled_profile_used": True,
    "explicit_k": FLOW_K,
    "default_k": 9,
    "c_slots": C_SLOTS,
    "default_profile_qualification": (
        "UNAVAILABLE for the flow family on this frame: default K=9 with "
        "the minimum feasible numba pool H=1 gives H+K = 10 > C=9 -- the "
        "production default profile cannot be admitted; RESOURCES.md :11 "
        "remedy taken"),
    "criterion": (
        "H_numba + K <= C_SLOTS and W*(H_numba+K) <= C_SLOTS (child gate "
        "spec.py:294-313; posture H >= K preserved from the observed "
        "10:9 topology)"),
    "numerical_profile": (
        "explicit K=4 is a DISTINCT numerical profile from default-K=9 "
        "runs: stripe membership range(slot, n_origins, n_threads) "
        "depends on K, so float sum order differs; cross-profile output "
        "comparison is never regression/improvement evidence; the S01 "
        "arm comparison is strictly within-profile"),
    "static_source": "Topology.py:79 num_threads = mp.cpu_count()-1 (also "
                     ":1143, :2060); constructor override only when not None",
    "runtime_probe": {
        "utc": "2026-09-28",
        "frame": "w00_selected wheel venv",
        "mp_cpu_count": 10,
        "instance_num_threads": 9,
    },
    "flow_stripes_cli": "4",
}

# --- workloads (families x cells; O3_HOLDOUT protected, never swept) -------
SWEEP_WORKLOADS = {
    # Premise note (flagged to h04 with the K-fit note): parent SS1 phrased
    # the accessibility family as "O2 + O3_ACCESS", but the amendment-#2
    # closure arithmetic (24 accessibility batches = 6 x 2 x 2) binds ONE
    # swept accessibility workload: O3_ACCESS, the primary observed proxy
    # cell.  O2 participates through the P0 pilots only (cold/warm cell).
    "accessibility": "O3_ACCESS",
    "flow": "O3_FLOW",
}
PILOT_CELLS = ["O2", "O3_ACCESS", "O3_FLOW"]   # parent SS3: 3 cells x 2
PILOTS_PER_CELL = 2                            # DECISION :22 cap <= 2/cell
HOLDOUT_NEVER_SWEPT = True
FLOW_VARIANT = {
    "variant_id": "sel1024",
    "basis": "H05 flow-variant decision (campaign_data/"
             "h05_flow_decision.json: chosen_variant=sel1024; sel-all "
             "capacity-refused)",
    "origins_fixture": (str(LEGACY_WS) + "/"
                        "campaign_data/inputs/"
                        "O3_FLOW_origins_sel1024.geojson"),
    "origins_fixture_sha256": (
        "75fc10b37fbf0cfd09490979a8619d30f3b270cef6048cfb36de0dbd86cdaa80"),
}
ACCESS_ORIGINS = {
    "selection": ("bounded fixture, 1,024 origins (h04 Ruling 2, "
                  "2026-09-29: bounded fixture ordered; the former sel_all "
                  "identity-selection block is SUPERSEDED and must not "
                  "survive to the declaration freeze)"),
    "origins_fixture": (str(LEGACY_WS) + "/"
                        "campaign_data/inputs/"
                        "s01_O3_ACCESS_origins_sel1024.geojson"),
    "origins_fixture_sha256": (
        "70a1083f937a01bb6694606d5bd804a53bbfec24daf6a35298936cd71ee5bafa"),
    "provenance": {
        "parent_manifest": ("tests/large_e2e/inputs/"
                            "O3_ACCESS_sel_all.origin_indices.json"),
        "parent_manifest_sha256": (
            "2e535cbc45531d86b15a748e96888fe8bb007be7a05afee79683b718ce3586cd"),
        "extraction_rule": ("sorted(raw_draw_order[:1024]) of the parent "
                            "manifest; rows written in sorted-index order; "
                            "seed-permutation reproduction cross-check exact"),
        "source_file": ("campaign_data/inputs/"
                        "Cambridge_building_centroids.geojson"),
        "source_file_sha256": (
            "14999756f30cc0f3245ac104b64f807dce76765e7c0e28af1da499c892d3bcdc"),
        "derivation_record": (str(LEGACY_WS) + "/"
                              "campaign_data/"
                              "s01_O3_ACCESS_fixture_derivation.json"),
        "derivation_record_sha256": (
            "d22243ab7481d7bdb09e002101323919d7c8d620f5826dc7bf8680a2097f6a04"),
        "custody_pin": ("campaign_data/custody/"
                        "interim_70a1083f_s01_O3_ACCESS_origins_sel1024"
                        ".geojson"),
        "authorization": ("h04 Ruling 2 (order side) + team-lead GO on "
                          "pre-write declaration sha 9f7be44e"),
        "identity": ("single fixture file consumed IDENTICALLY by both arms "
                     "and all accessibility configs"),
        "q10_transfer_note": ("config ranked on the bounded fixture; Q10 "
                              "confirms the selected config on the mandatory "
                              "full workload (h04 Ruling 2 transfer-risk "
                              "bound)"),
    },
}
RUN_MANIFESTS = {
    "O3_ACCESS": (str(LEGACY_WS) + "/wt-large-e2e/"
                  "tests/large_e2e/scheduling/"
                  "s01_O3_ACCESS.run.manifest.json"),
    "O3_FLOW": (str(LEGACY_WS) + "/wt-large-e2e/"
                "tests/large_e2e/scheduling/"
                "s01_O3_FLOW_sel1024.run.manifest.json"),
}
ARM_VENVS = {
    "b0": str(LEGACY_WS) + "/campaign_data/venvs/w00_b0",
    "selected": (str(LEGACY_WS) + "/campaign_data/"
                 "venvs/w00_selected"),
}
ARM_IDENTITIES = {
    "b0": (str(LEGACY_WS) + "/campaign_data/"
           "w00_identities/b0.json"),
    "selected": (str(LEGACY_WS) + "/campaign_data/"
                 "w00_identities/selected.json"),
}
# Reference configs for the P0 baseline-only pilots (per-job wall source).
# flow: the AMD7 labelled explicit profile (W1, H5, explicit K=FLOW_K) --
# the feasible projection of the observed posture; see FLOW_CONFIGS.
PILOT_REFERENCE_CONFIG = {"accessibility": (1, 1), "flow": (1, 5)}

# --- SS6.3 frozen thresholds (freeze at matrix freeze; no mid-sweep change) -
QUEUE_POLICY = {
    "initial_depth_rule": "queue-depth = W per cell",
    "producer_wait_source": ("harness-recorded per-job spans "
                             "(pool.dispatch_and_drain producer_wait_ns_by_"
                             "job) -- never wall deltas"),
    "producer_wait_share_max": 0.10,   # >10% of job wall
    "live_memory_frac_of_ceiling_max": 0.50,
    "reference_measurement": "B0 pilot leg",
    "decision_scope": "per_family",
    "disagreement_note": ("selected-arm disagreement recorded in "
                          "memory.json; matrix never splits (equal-resource)"),
    "trigger_action": ("STOP before P1 and route through review: a 2W config "
                       "is a matrix change; the cap never stretches"),
}
WRITER_POLICY = ("writer limit = W everywhere; no lowering proposed (no "
                 "implemented ownership-preserving mechanism exists)")

# --- tranche caps -----------------------------------------------------------
BATCH_TIMEOUT_S = 3600      # bounded subprocess timeout, every child
PILOT_TIMEOUT_S = 3600
TOTAL_BATCHES = 28          # 24 accessibility + 4 flow (amendment #2 SS3)


def accessibility_batches() -> int:
    return len(ACCESSIBILITY_CONFIGS) * BATCHES_PER_CELL * len(ARMS)


def flow_batches() -> int:
    return len(FLOW_CONFIGS) * BATCHES_PER_CELL * len(ARMS)


def total_batches() -> int:
    return accessibility_batches() + flow_batches()


def largest_w(family: str) -> int:
    configs = ACCESSIBILITY_CONFIGS if family == "accessibility" else FLOW_CONFIGS
    return max(w for w, _h in configs)


def k_jobs_divisible(k_jobs: int, family: str) -> bool:
    """BENCHMARKS: fixed K_jobs divisible by the largest compared W."""
    return k_jobs % largest_w(family) == 0


def config_within_c(w: int, h: int) -> bool:
    """spec.py :268-276 necessary-not-sufficient charge check."""
    return w <= C_SLOTS and w * h <= C_SLOTS


def matrix_valid() -> bool:
    if total_batches() != TOTAL_BATCHES:
        return False
    if len(ACCESSIBILITY_CONFIGS) > 6 or len(FLOW_CONFIGS) > 6:
        return False
    if (1, 1) not in ACCESSIBILITY_CONFIGS:
        return False
    for w, h in ACCESSIBILITY_CONFIGS + FLOW_CONFIGS:
        if not config_within_c(w, h):
            return False
    for w, h in FLOW_CONFIGS:
        if h < FLOW_K:                     # H >= K posture (observed 10:9)
            return False
        if h + FLOW_K > C_SLOTS:           # child gate term 1 (spec.py:298)
            return False
        if w * (h + FLOW_K) > C_SLOTS:     # child gate term 2 (spec.py:306)
            return False
    if not HOLDOUT_NEVER_SWEPT:
        return False
    return True
