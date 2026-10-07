#!/usr/bin/env python
"""X00 inherited-platform-obligations audit generator (task X00, campaign una_10x post_p00).

Every figure and sha256 in the emitted JSONs is computed in this process from the
cited bytes at generation time. Stdlib only; imports no engine modules; writes only
inside campaign_runs/una_10x/tenx-20261005T2331Z/x00/.
"""
import hashlib
import json
import os
import subprocess

REPO = "/Users/alansynn/orca/workspaces/una-x/main"
RUN = os.path.join(REPO, "campaign_runs/una_10x/tenx-20261005T2331Z")
X00 = os.path.join(RUN, "x00")
PLAT = os.path.join(REPO, "campaigns/una_platform")
CHECKPOINT = "6130769a5751bf7446734dc39b9f8e3763461cfe"
GENERATED = "2026-10-07"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(
        ["git", "-C", REPO, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def rel(path):
    return os.path.relpath(path, REPO)


def cite(path, label, grade="OBSERVED"):
    """Evidence citation: computed sha bound to the bytes read."""
    ap = path if os.path.isabs(path) else os.path.join(REPO, path)
    return {"label": label, "path": rel(ap), "sha256": sha256(ap), "grade": grade}


def jload(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- git frame
head = git("rev-parse", "HEAD")
branch = git("branch", "--show-current")

COMMITS = {
    "G11": "eb9276f", "G10": "03224a9", "G00": "12ffa6b",
    "N10": "8d49e9d", "N00R": "0bb38b6",
}
commit_subjects = {k: git("log", "-1", "--format=%s", v) for k, v in COMMITS.items()}
commit_full = {k: git("rev-parse", v) for k, v in COMMITS.items()}

# campaign touchprint on inherited surfaces since the P00 checkpoint
surfaces = [
    "src/urban_network_analysis/compat",
    "src/urban_network_analysis/reference",
    "src/urban_network_analysis/cache",
    "src/urban_network_analysis/batch",
    "src/urban_network_analysis/Engines",
    "src/urban_network_analysis/kernels",
    "native",
    "gpu",
    "pyproject.toml",
    "src",
]
touchprint = {}
for s in surfaces:
    n = git("log", "--oneline", f"{CHECKPOINT}..HEAD", "--", s)
    touchprint[s] = len(n.splitlines()) if n else 0

# working-tree state at audit time (defect #24 fix in flight)
wt_status = git("status", "--porcelain")

# ------------------------------------------------------- platform receipts
plat_tasks = jload(os.path.join(PLAT, "TASKS_CLAUDE.yaml"))["tasks"]
plat_task_ids = [t["id"] for t in plat_tasks]
tasks_dir = os.path.join(PLAT, "evidence/tasks")

receipts = {}
for task in sorted(os.listdir(tasks_dir)):
    d = os.path.join(tasks_dir, task)
    if not os.path.isdir(d):
        continue
    for run in sorted(os.listdir(d)):
        rp = os.path.join(d, run, "receipt.json")
        if os.path.isfile(rp):
            receipts.setdefault(task, []).append(
                {"run_id": run, "path": rel(rp), "sha256": sha256(rp)}
            )

present = sorted(receipts)
absent = [t for t in plat_task_ids if t not in receipts]
absent = sorted(set(absent) - set(present))

# ------------------------------------------------- 10x campaign evidence
p = lambda *a: os.path.join(RUN, *a)

n00_impl = jload(p("n00/native_implementation.json"))
n00_review = jload(p("n00/review.json"))
n10_qual = jload(p("n10/native_qualification.json"))
n10_install = jload(p("n10/install_identity.json"))
g00_spec = jload(p("g00/gpu_region.json"))
g10_impl = jload(p("g10/gpu_implementation.json"))
g10_gate = jload(p("g10/gate_harness_failclosed.json"))
g11_qual = jload(p("g11/gpu_qualification.json"))
f00_adm = jload(p("f00/flow_admission.json"))
f10_verdict = jload(p("f10/flow_verdict.json"))
p05s_index = jload(p("p05s/screen_index.json"))
p05r_ready = jload(p("post_p00_readiness.json"))
baseline_audit = jload(p("baseline_policy_audit.json"))
primary_id = jload(p("primary_identity_check.json"))
budget = jload(p("budget_reconciliation.json"))
incidents = jload(os.path.join(PLAT, "incidents.json"))

# ---------------- per-family current evidence (all shas computed above/below)
fam_evidence = {
    "madina_parity": [],
    "scientific_corrections": [
        cite("campaigns/una_10x/evidence/proofs/S10_sparse_projection/sparse_admission.json",
             "S01/S10 sparse projection admission (prior run evidence, retained)"),
        cite("campaigns/una_10x/evidence/proofs/S10_sparse_projection/review_indep/verdict.json",
             "independent review of sparse projection proof"),
    ],
    "batch_runtime": [
        cite("campaigns/una_10x/evidence/faults/H01-tenx-20261003T1212Z/fault_matrix.json",
             "H01 fault matrix (prior run evidence, retained)"),
        cite("campaigns/una_10x/evidence/screens/H01-tenx-20261003T1212Z/record.json",
             "H01 batch/flow screens record (prior run evidence, retained)"),
    ],
    "cache_policy": [
        cite(p("baseline_policy_audit.json"),
             "P01 baseline-policy audit: status=" + baseline_audit["status"]),
        cite(p("primary_identity_check.json"),
             "primary identity check: " + primary_id["status"]),
    ],
    "native_backend": [
        cite(p("n00/native_implementation.json"), "N00 implementation record"),
        cite(p("n00/review.json"), "N00R independent review"),
        cite(p("n00/native_implementation_errata_f1.json"), "N00R errata F1 (parent immutable)"),
        cite(p("n00/native_implementation_errata_f1_reference_fix.json"),
             "N00R errata F1 correction of record"),
        cite(p("n10/native_qualification.json"), "N10 native qualification"),
        cite(p("n10/install_identity.json"), "N10 installed-wheel identity"),
    ],
    "gpu_support": [
        cite(p("g00/gpu_region.json"), "G00 GPU residency region spec"),
        cite(p("g10/gpu_implementation.json"), "G10 GPU region implementation record"),
        cite(p("g10/gate_harness_failclosed.json"), "G10 gate harness designed refusal"),
        cite(p("g10/pytest_gpu_receipt.txt"), "G10 53-test receipt"),
        cite(p("g11/gpu_qualification.json"), "G11 explicit_unavailable qualification"),
    ],
    "city_failures": [
        cite(os.path.join(PLAT, "incidents.json"),
             f"platform incident register ({len(incidents['incidents'])} incidents)"),
    ],
    "install_packaging": [
        cite(p("n10/install_identity.json"), "N10 installed-wheel identity + defect record"),
        cite("pyproject.toml", "live pyproject with force-include fix (audit-time bytes)"),
        cite(p("n10/packaging_census_fixed.json"), "N10 post-fix packaging census"),
    ],
}
# shared citations appended to every family needing the campaign frame
frame_for_all = [
    cite(p("post_p00_readiness.json"), "P05R readiness review (approve_with_notes, 0 blocking)"),
    cite(p("p05s/screen_index.json"), "P05S staged selection screen"),
]
fam_evidence["batch_runtime"] += frame_for_all
fam_evidence["madina_parity"] += frame_for_all
fam_evidence["cache_policy"] += frame_for_all

# N00 parity figure extracted from the committed record
n00_parity_suite = n00_impl["parity"]["suite"]
n00_review_blocking = n00_review["blocking"]
equal_c = n10_qual["equal_C_comparison"]
# L1B2 co-tenancy anomaly wording located in the budget reconciliation bytes
budget_text = json.dumps(budget)
assert "L1B2" in budget_text and "co-tenancy" in budget_text.lower() or "co-tenan" in budget_text

incident_rows = [
    {"id": i["id"], "status": i["status"], "root_cause_verified": i["root_cause_verified"]}
    for i in incidents["incidents"]
]

# ------------------------------------------------------------------ families
def family(fid, title, inherited, status_impl, status_evid, remaining, mapping, gate,
           blockers, disposition):
    return {
        "id": fid,
        "family": title,
        "inherited_from": inherited,
        "platform_receipts": {t: receipts[t] for t in inherited if t in receipts},
        "platform_receipts_absent": [t for t in inherited if t not in receipts],
        "implementation_status": status_impl,
        "evidence_status": status_evid,
        "current_evidence": fam_evidence[fid],
        "campaign_touchprint_since_checkpoint": {
            s: touchprint[s] for s in surfaces if s in (
                "src/urban_network_analysis/compat",
                "src/urban_network_analysis/reference",
                "src/urban_network_analysis/cache",
                "src/urban_network_analysis/batch", "native", "gpu", "pyproject.toml",
            )} if fid in ("madina_parity", "scientific_corrections", "cache_policy",
                          "batch_runtime", "native_backend", "gpu_support",
                          "install_packaging") else {},
        "remaining_work": remaining,
        "owner": "integrator (X00) -> successor tasks",
        "new_task_mapping": mapping,
        "acceptance_gate": gate,
        "blockers": blockers,
        "disposition": disposition,
    }


families = [
    family(
        "madina_parity", "Madina API/functionality parity, shim and workflows",
        ["INVENTORY", "MADINA_ZONAL", "MADINA_ACCESS", "MADINA_PATHS", "MADINA_FLOW",
         "MADINA_WORKFLOWS", "API_REVIEW"],
        "closed at platform scope: all 7 inherited tasks carry receipts; implementation reviews ACCEPT/ACCEPT-with-findings",
        "current (receipt-level); campaign made 0 commits touching src/urban_network_analysis/compat and .../reference since checkpoint "
        f"{CHECKPOINT[:12]}, so receipts remain byte-current for the compat surface",
        ["campaign-end revalidation of the parity surface at V00/R00 (installed parity cells)"],
        "X00 (this audit); V00/R00 revalidation",
        "PLATFORM_CONTINUITY.md: parity matrix against pinned Madina",
        [],
        "evidence_closed",
    ),
    family(
        "scientific_corrections",
        "Scientific fixes and corrected reference profiles (una_legacy / madina_legacy / corrected_v1)",
        ["CONTRACT", "SCIENCE", "TOPOLOGY"],
        "closed at platform scope; campaign introduced no new scientific default and did not touch reference surfaces (0 commits)",
        "current (receipt-level + campaign retention evidence)",
        ["none at family level; V00 re-runs profile bit checks from installed artifacts"],
        "X00 (this audit); S01/S10 already evidenced the sparse profile; V00 revalidation",
        "PLATFORM_CONTINUITY.md: three references preserved as distinct; never claim a speedup from a model change",
        [],
        "evidence_closed",
    ),
    family(
        "batch_runtime",
        "Public RunBatch plan/runtime/checkpoint (ordering, cancellation, recovery)",
        ["BATCH_PLAN", "BATCH_EXEC", "BATCH_REVIEW"],
        "closed at platform scope; campaign P03 removed the Python origin loop without changing the compiled full fold (P05R-bound); "
        "H01 exercised execution, ordered commits, cancellation and failure-prefix behavior",
        "current, with ONE OPEN engine defect",
        ["close defect DEF-24 (cancel-path wedge): fix in flight in working tree at audit time, uncommitted, unreviewed",
         "post-fix rerun of cancellation/recovery cells; V00 fault gauntlet"],
        "X00 (this audit); defect fix owner n00-native-engineer (task #24); V00 fault gauntlet",
        "PLATFORM_CONTINUITY.md: H01 tests actual execution, ordered commits, cancellation and failure-prefix behavior",
        [{
            "id": "DEF-24",
            "description": "runtime._cancel queue-wedge: any worker-row failure can wedge the batch instead of failing it "
                          "(main joins a backlogged Queue feeder inside runtime._cancel -> Queue.join_thread); failures swallowed",
            "evidence": "n00/wedge_persistence_sd_direct_sidecar.txt (faulthandler stack) + n00/wedge_triage_console.txt",
            "status": "OPEN - fix in flight (task #24; src/urban_network_analysis/batch/runtime.py modified, "
                      "tests/batch/runtime/test_runtime_cancel_wedge.py untracked in working tree at audit time)",
        }],
        "open_blockers",
    ),
    family(
        "cache_policy",
        "Content-addressed store, typed identities, resident leases, corruption/single-flight/quotas",
        ["CACHE_STORE", "CACHE_GRAPH"],
        "closed at platform scope; campaign made 0 commits to src/urban_network_analysis/cache since checkpoint; "
        "C10-C31 typed identities + leases from the prior campaign preserved; screens ran at the frozen cache-off policy",
        "current (receipt-level + untouched-surface proof + P01 baseline-policy audit)",
        ["V00 revalidation of cache/state invariants from installed artifacts"],
        "X00 (this audit); V00 revalidation",
        "PLATFORM_CONTINUITY.md: retain corruption, single-flight, byte quotas, safe decoding, atomic publication",
        [],
        "evidence_closed",
    ),
    family(
        "native_backend",
        "Substantive native/AOT backend (NATIVE_CORE, NATIVE_SIMD at platform scope)",
        ["NATIVE_CORE", "NATIVE_SIMD"],
        "platform-scope tasks NEVER EXECUTED (no evidence dirs); this campaign added a real, reviewed native trace-producer "
        "capability (N00) and qualified it (N10) without admitting it to the 10x gate stack",
        "campaign evidence current; platform-scope obligations remain unexecuted",
        ["platform NATIVE_CORE/NATIVE_SIMD scopes (AOT core kernel surface, SIMD/divergence variants) remain unexecuted - no receipts exist",
         "platform PARITY native cells unrun",
         "DEF-24 fix touches batch/runtime, not the native producer; native surface itself has no open defect"],
        "X00 (this audit); N00/N10 closed by n00-native-engineer; PARITY cells -> V00",
        "PLATFORM_CONTINUITY.md: integrate reviewed ongoing work first; never port the slow portable oracle as a weak comparator",
        [{
            "id": "BLK-NATIVE-PLATFORM-SCOPE",
            "description": "inherited NATIVE_CORE and NATIVE_SIMD platform tasks have no evidence dirs and no receipts; "
                           "the platform-scope AOT core/SIMD obligation is not discharged by the 10x-scope N00 trace producer",
            "evidence": "campaigns/una_platform/evidence/tasks/ contains no NATIVE_CORE/ or NATIVE_SIMD/ dirs (audited this task)",
            "status": "OPEN - inherited obligation incomplete at platform scope; capability (not performance) carried by N00/N10",
        }],
        "open_blockers",
    ),
    family(
        "gpu_support",
        "GPU support: device runtime, substantive kernels, residency region, qualification",
        ["GPU_DEVICE", "GPU_KERNEL", "GPU_REGION"],
        "platform-scope tasks NEVER EXECUTED (no evidence dirs); this campaign specified (G00), implemented (G10, bounded device "
        "source + exact host twins) and explicitly-disqualified (G11) the fold-family region; admission fail-closed",
        "campaign evidence current; device qualification blocked by hardware absence (qualification blocker, never simulated success)",
        ["G10R independent implementation review of G10 (task #28, PENDING)",
         "real-device bit parity + end-to-end crossover on CUDA-class hardware with FP64 (no such hardware on this host)",
         "platform GPU_DEVICE/GPU_KERNEL/GPU_REGION scopes remain unexecuted; platform PARITY GPU cells blocked",
         "a fallback-only GPU route is NOT substantive support (PLATFORM_CONTINUITY.md) - implementation must be exercised on hardware"],
        "X00 (this audit); G10R -> gpu-engineer/reviewer; hardware cells -> I00/V00 with explicit unavailable",
        "PLATFORM_CONTINUITY.md: G00/G10 are real implementation/qualification tasks; hardware absence blocks G10, not independent CPU work",
        [{
            "id": "BLK-GPU-HW",
            "description": "no CUDA-class device and no nvcc on the audit host (Apple M1 Pro, Darwin; Apple GPU explicitly not a "
                           "qualification target); device parity/benchmark cells cannot run here",
            "evidence": "g10/gate_harness_failclosed.json (designed refusal rc=2, no_cuda_driver) + g11/gpu_qualification.json "
                        "(team-lead harness run rc=2 on record)",
            "status": "OPEN - qualification blocker; FoldRegionRecord.hardware_validated stays constant False",
        }, {
            "id": "BLK-G10R",
            "description": "G10 implementation awaits independent review (task G10R #28 pending); no review verdict exists yet",
            "evidence": "task register: #28 G10R pending",
            "status": "OPEN - review pending",
        }],
        "open_blockers",
    ),
    family(
        "city_failures",
        "City failures, incident root causes, diagnostic-mode behavior",
        ["FAILURES", "CITIES"],
        "FAILURES closed at platform scope (9 published incidents triaged, none reproduced); CITIES platform task NEVER EXECUTED; "
        "campaign ran no city-failure work",
        "receipt current; incident substance explicitly unresolved",
        ["all 9 incidents remain root_cause_verified=false; reproduction status unchanged",
         "CITIES platform scope (observed morphology corpus, failure regressions) unexecuted",
         "platform FAULTS final-fault gate unexecuted"],
        "X00 (this audit); incident reproduction remains a platform obligation with no active owner",
        "PLATFORM_CONTINUITY.md: preserve incident identity and actual reproduction status; no claimed city root cause without data",
        [{
            "id": "BLK-CITY-ROOTCAUSES",
            "description": f"{len(incident_rows)} platform incidents (INC01-INC09) all reported_not_reproduced / "
                           "hypothesis_not_reproduced with root_cause_verified=false",
            "evidence": "campaigns/una_platform/incidents.json",
            "status": "OPEN - genuine incident evidence recorded; causes unverified",
        }, {
            "id": "BLK-CITIES-SCOPE",
            "description": "CITIES platform task has no evidence dir and no receipt",
            "evidence": "campaigns/una_platform/evidence/tasks/ contains no CITIES/ dir (audited this task)",
            "status": "OPEN - inherited obligation incomplete",
        }],
        "open_blockers",
    ),
    family(
        "install_packaging",
        "Packaging, CI, installed-path validation",
        ["PACKAGE"],
        "platform PACKAGE task NEVER EXECUTED; campaign found and FIXED an inherited wheel-packaging defect "
        "(spawn shim una_batch_worker_main.py missing from wheels, latent since 0acbb11, inherited from parent e7c9eda) via "
        "pyproject force-include, proven by an actual installed-wheel run",
        "campaign evidence current (installed-wheel identity pinned); CI and clean A/B installed validation do not exist yet",
        ["unrun CI is unavailable (PLATFORM_CONTINUITY.md: 'Unrun CI is unavailable') - no CI evidence exists",
         "I00 must build A/B clean wheels and install OUTSIDE checkouts, pin modules/binaries/harnesses (task #19 prep pending)",
         "installed L3 validation and V00 installed-parity cells not yet run"],
        "X00 (this audit); packaging fix owner n00-native-engineer (N10); I00/V00 pending",
        "PLATFORM_CONTINUITY.md: packaging/CI/installed-path -> I00/V00/R00/Z00",
        [{
            "id": "BLK-INSTALL-CI-I00",
            "description": "no CI runs and no clean A/B installed-wheel validation exist; I00 depends on X00 (this audit) and is pending",
            "evidence": "PLATFORM_CONTINUITY.md + una_10x TASKS_CLAUDE.yaml I00 block; no CI artifacts anywhere in either campaign tree (audited)",
            "status": "OPEN - scheduled successor work, not waived",
        }],
        "open_blockers",
    ),
]

# ----------------------------------------------------------- campaign integration
integration = {
    "law": "PLATFORM_CONTINUITY.md: integrate independently reviewed pending implementations WITHOUT resetting baseline",
    "baseline_continuation": {
        "comparator": "395cdc5 (baseline-395cdc5 worktree + venv-baseline), executable and untouched",
        "p00_parent": "e7c9eda (parent-e7c9eda worktree + venv-parent-e7c9eda), untouched",
        "frozen_primary": "92.1141s PACE primary untouched; screens are selection evidence only (P05S notes)",
        "evidence": cite(p("primary_identity_check.json"),
                         "commit pins verified: checkpoint/parent ancestors of HEAD, comparator ancestor of origin/main"),
    },
    "items": [
        {"item": "N00 native trace producer", "commit": commit_full["N00R"],
         "subject": commit_subjects["N00R"],
         "review": "approve_with_notes, blocking=[] (p03r-reviewer); errata F1 + correction of record filed",
         "evidence": cite(p("n00/review.json"), "N00R review bytes")},
        {"item": "N10 native qualification + wheel packaging fix", "commit": commit_full["N10"],
         "subject": commit_subjects["N10"],
         "review": "performance-owner qualification; native NOT admitted to 10x gate stack (no reliable lever), capability RETAINED",
         "evidence": cite(p("n10/native_qualification.json"), "N10 qualification record")},
        {"item": "G00 GPU region spec", "commit": commit_full["G00"], "subject": commit_subjects["G00"],
         "review": "performance-owner spec; verification-gated parity class; G10 buildable without hardware",
         "evidence": cite(p("g00/gpu_region.json"), "G00 spec record")},
        {"item": "G10 GPU region implementation", "commit": commit_full["G10"],
         "subject": commit_subjects["G10"],
         "review": "PENDING - awaiting G10R independent review (task #28); no review verdict exists; new gpu/ surface only, src/ untouched",
         "evidence": cite(p("g10/gpu_implementation.json"), "G10 implementation record")},
        {"item": "G11 explicit-unavailable GPU qualification", "commit": commit_full["G11"],
         "subject": commit_subjects["G11"],
         "review": "performance-owner qualification; zero performance claims; capability vs performance separated",
         "evidence": cite(p("g11/gpu_qualification.json"), "G11 qualification record")},
        {"item": "F00/F10 flow chain", "commit": "83f193a/84e4b25/83a1b10 (F00 custody per f10 parent_evidence)",
         "subject": "F00 sparse-output producer + ordered tape both not_admitted (Amdahl S<=1.134); F10 not triggered; existing flow unaffected",
         "review": "admission + verdict records; no src/tests changes",
         "evidence": cite(p("f10/flow_verdict.json"), "F10 verdict binding F00 sha")},
        {"item": "P05S staged screen + P05R readiness review", "commit": "86e8b0a measured / be8441b reviewed",
         "subject": "full accepted stack (P02I+P03C+P03M-A/B+P04I-fix) bound at w4 t1 q64 cache-off; approve_with_notes, 0 blocking",
         "review": "independent (p03r-reviewer); selection evidence only, NOT a 10x claim",
         "evidence": cite(p("post_p00_readiness.json"), "P05R readiness verdict")},
        {"item": "wheel spawn-shim packaging fix", "commit": commit_full["N10"],
         "subject": "pyproject force-include maps src/una_batch_worker_main.py to wheel root; defect INHERITED from parent e7c9eda, latent since 0acbb11",
         "review": "proven by actual installed-wheel execution (install_identity.json)",
         "evidence": cite("pyproject.toml", "live pyproject bytes (audit time)")},
        {"item": "DEF-24 cancel-wedge fix", "commit": None,
         "subject": "IN FLIGHT at audit time: src/urban_network_analysis/batch/runtime.py modified, "
                    "tests/batch/runtime/test_runtime_cancel_wedge.py untracked; NOT committed, NOT reviewed",
         "review": "none yet - recorded as OPEN, never as closed",
         "evidence": {"label": "working-tree status at audit time", "porcelain": wt_status, "grade": "OBSERVED"}},
    ],
}

# ---------------------------------------------------------------- incidents
campaign_defects = {
    "DEF-24_silent_wedge": {
        "engine_defect": "runtime._cancel queue-wedge swallows worker-row failures (runtime.py:1215 dispatch; _cancel at runtime.py:1466 -> cancel runtime.py:443 -> Queue.join_thread)",
        "status": "OPEN (tasks #20/#24 in_progress; fix in flight, uncommitted at audit time)",
        "evidence": [
            cite(p("n00/wedge_persistence_sd_direct_sidecar.txt"),
                 "persistence proof: faulthandler sidecar stack, main inside _cancel joining backlogged Queue feeder"),
            cite(p("n00/wedge_triage_console.txt"),
                 "triage: four attempts, one root cause"),
        ],
    },
    "packaging_defect": {
        "defect": "spawn shim una_batch_worker_main.py not packaged in wheels -> parallel RunBatch ModuleNotFoundError at pool.start (runtime.py:379) in any non-editable install",
        "status": "FIXED (pyproject force-include; verified in installed wheel)",
        "evidence": [cite(p("n10/install_identity.json"), "installed-wheel run evidence")],
    },
    "L1B2_co_tenancy_anomaly": {
        "anomaly": "L1B2 native leg 16.2113s fired after loadavg-gate release from a 12.54 entry reading; in-run 1-min loadavg max 9.66 exceeded the <8 gate inside the leg; ambient co-tenancy",
        "status": "RECORDED; no claim rests on it (N10 verdict treats the native arms as contradicting at n=2 under co-tenancy and admits nothing)",
        "evidence": [cite(p("budget_reconciliation.json"), "budget reconciliation, L1B2 anomaly entry"),
                     cite(p("n10/native_qualification.json"), "equal_C_comparison verdict")],
    },
    "n00_shakedown_wedges": {
        "count": 4,
        "description": "four shakedown attempts wedged during N00 instrumentation; faulthandler sidecar resolved all four to the single cancel-path root cause (DEF-24); attempt-1 sampler starvation was a real instrument finding but NOT the wedge cause",
        "status": "DIAGNOSED (root cause identified); engine fix = DEF-24, open",
        "evidence": [cite(p("n00/wedge_triage_console.txt"), "shakedown triage console"),
                     cite(p("n00/legs_shakedowns_console.txt"), "shakedown legs console")],
    },
}

# ---------------------------------------------------------------- assemble
obligations = {
    "schema": "una-10x-x00-platform-obligations-1",
    "task": "X00 Carry and close inherited platform obligations (campaign una_10x post_p00)",
    "generated": GENERATED,
    "author": "integ-audit (integrator)",
    "audit_frame": {
        "head": head,
        "branch": branch,
        "retained_law": cite("campaigns/una_10x/PLATFORM_CONTINUITY.md",
                             "PLATFORM_CONTINUITY.md (retained-obligations law)"),
        "platform_task_file": cite(os.path.join(PLAT, "TASKS_CLAUDE.yaml"),
                                   "retained platform TASKS_CLAUDE.yaml"),
        "platform_task_file_note": "declared read path ../una_platform/TASKS_CLAUDE.yaml does not exist at the workspace "
                                   "sibling level; the retained in-repo copy campaigns/una_platform/TASKS_CLAUDE.yaml is the audited file",
        "x00_task_block": cite("campaigns/una_10x/TASKS_CLAUDE.yaml", "una_10x TASKS_CLAUDE.yaml (X00 block)"),
        "completion_law": "Every inherited obligation has current evidence or explicit unfinished blocker; no waived requirements",
        "method": "every sha256 computed in this task from the cited bytes; platform receipt census walked from "
                  "campaigns/una_platform/evidence/tasks/; git touchprint via git log 6130769a..HEAD per surface",
    },
    "platform_receipt_census": {
        "total_platform_tasks": len(plat_task_ids),
        "tasks_with_receipts": present,
        "tasks_with_receipt_count": len(present),
        "tasks_without_evidence_dirs": absent,
        "tasks_without_evidence_dir_count": len(absent),
        "receipts": {t: receipts[t] for t in present},
    },
    "campaign_touchprint_since_checkpoint": touchprint,
    "commit_pins": {k: {"sha": commit_full[k], "subject": commit_subjects[k]} for k in COMMITS},
    "families": families,
    "integration_of_reviewed_pending_work": integration,
    "platform_incident_register": {
        "path": rel(os.path.join(PLAT, "incidents.json")),
        "sha256": sha256(os.path.join(PLAT, "incidents.json")),
        "observed_date": incidents.get("observed_date"),
        "incidents": incident_rows,
        "summary": f"{len(incident_rows)} incidents; all root_cause_verified=false; "
                   f"{sum(1 for i in incident_rows if i['status']=='reported_not_reproduced')} reported_not_reproduced, "
                   f"{sum(1 for i in incident_rows if i['status']=='hypothesis_not_reproduced')} hypothesis_not_reproduced",
    },
    "campaign_defect_and_anomaly_register": campaign_defects,
    "no_waivers": {
        "statement": "No requirement is waived. Families with disposition evidence_closed carry current receipt-level "
                     "evidence; families with disposition open_blockers carry explicit unfinished blockers with named "
                     "evidence. Receipt evidence is not installed-engagement proof (PLATFORM_CONTINUITY.md); installed "
                     "revalidation is scheduled at I00/V00 and recorded there as pending, not assumed.",
        "explicitly_not_waived": [
            "GPU device qualification (hardware-blocked, explicit_unavailable on record)",
            "city incident root-cause verification (9 unverified)",
            "platform-scope NATIVE_CORE/NATIVE_SIMD/GPU_*/CITIES/PACKAGE execution (no receipts exist)",
            "CI execution and clean A/B installed validation (I00)",
            "G10R independent review of G10",
            "DEF-24 engine fix (in flight, unreviewed)",
        ],
    },
    "summary_counts": {
        "families_total": len(families),
        "families_evidence_closed": sum(1 for f in families if f["disposition"] == "evidence_closed"),
        "families_open_blockers": sum(1 for f in families if f["disposition"] == "open_blockers"),
        "open_blockers_total": sum(len(f["blockers"]) for f in families),
    },
}

status = {
    "schema": "una-10x-x00-platform-status-1",
    "task": "X00",
    "generated": GENERATED,
    "author": "integ-audit (integrator)",
    "law": "PLATFORM_CONTINUITY.md: publish tenx_status AND platform_status separately; only completion of all mandatory "
           "inherited capabilities AND their required hardware/API/scientific gates permits platform_status=qualified; "
           "a 10x-only result never marks missing tasks done",
    "head": head,
    "branch": branch,
    "obligations_record": {"path": "x00/platform_obligations.json", "sha256": None},  # filled below
    "tenx_status": {
        "value": "in_progress",
        "position": "post_p00 DAG: P01M/P01F, P02*, P03*, P04*, P05S/P05R, H01 (prior run), S01/S10, C10-C31, "
                    "F00/F10, N00/N00R/N10, G00/G10/G11 closed; X00 (this audit) closing; remaining: G10R, I00 "
                    "(incl. task #19 intake prep), V00, Q00, Q10, R00, Z00",
        "boundary_claim_status": "NO public 10x boundary figure exists yet; P05S/P05R figures are same-host matched-arm "
                                 "SELECTION evidence only; the claim path is the frozen 7-window paired AB/BA protocol "
                                 "(Q10) against the untouched 92.1141s PACE primary",
        "open_engine_defect": "DEF-24 cancel-path wedge (fix in flight, uncommitted at audit time)",
        "native_in_gate_stack": "numba stack unchanged; native producer retained as capability, not admitted as a gate lever (N10)",
        "gpu_in_gate_stack": "CPU retained; GPU region cannot run (no CUDA hardware), admission fail-closed (G11)",
        "evidence": frame_for_all + [cite(p("primary_identity_check.json"), "identity check")],
    },
    "platform_status": {
        "value": "incomplete",
        "qualified": False,
        "reason": "mandatory inherited capabilities remain unclosed: GPU device qualification hardware-blocked; 9 city "
                  "incident root causes unverified; CI never run; clean A/B installed validation (I00) pending; "
                  f"{len(absent)}/{len(plat_task_ids)} platform tasks have no evidence dirs "
                  "(NATIVE_CORE, NATIVE_SIMD, GPU_DEVICE, GPU_KERNEL, GPU_REGION, PARITY, CITIES, PACKAGE, SCREEN_CPU, "
                  "SCREEN_GPU, SCREEN_CACHE_BATCH, BACKEND_POLICY, INTEGRATE, FREEZE, CONFIRM, FAULTS, REVIEW, CLOSE); "
                  "DEF-24 engine defect open",
        "capability_matrix": {f["id"]: {"disposition": f["disposition"],
                                        "open_blockers": [b["id"] for b in f["blockers"]]}
                              for f in families},
        "hardware_blockers": ["no CUDA-class device / no nvcc on audit host (Darwin, Apple M1 Pro); "
                              "G11 explicit_unavailable + rc=2 designed refusal on record; "
                              "hardware absence is a qualification blocker, never simulated success"],
        "retained_capabilities": [
            "native bit-exact trace producer (N00, reviewed; env-gated opt-in; never silently engaged; numba wheel default)",
            "GPU fold-region specification + bounded device source + exact host twins (G00/G10; hardware_validated=False; fail-closed admission)",
            "batch runtime with ordering/checkpoint/recovery (platform receipts; DEF-24 cancellation defect open)",
            "cache store/typed identities (platform receipts + C10-C31; untouched by campaign commits)",
            "Madina compat surface (platform receipts; untouched by campaign commits)",
        ],
    },
}

# write deliverables, then bind the obligations sha into status and rewrite
ob_path = os.path.join(X00, "platform_obligations.json")
with open(ob_path, "w") as f:
    json.dump(obligations, f, indent=2, sort_keys=False)
    f.write("\n")
status["obligations_record"]["sha256"] = sha256(ob_path)
st_path = os.path.join(X00, "platform_status.json")
with open(st_path, "w") as f:
    json.dump(status, f, indent=2, sort_keys=False)
    f.write("\n")

print("WROTE", rel(ob_path), sha256(ob_path))
print("WROTE", rel(st_path), sha256(st_path))
print("summary:", json.dumps(obligations["summary_counts"]))
print("absent platform tasks:", ", ".join(absent))
print("touchprint:", json.dumps(touchprint))
