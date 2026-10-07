# I00 design note — drafted 2026-10-07, pre-review-gate (integrator scope)

Status: I00 EXECUTION is gated by the X00-accepted sequencing note — the freeze waits for
DEF-24R (fix landed ab357c6) and G10R (gpu/ surface), both with p03r-reviewer. This note pins
the already-determined design decisions so the fire ladder starts the moment the last verdict
lands. Facts below are tool outputs from this session unless cited otherwise.

## Law hooks

- TASKS_CLAUDE.yaml I00 (line 1102): integrate accepted deltas only; build A/B clean wheels and
  install outside checkouts; pin all modules/binaries/harnesses, check no research runtime
  imports; deliverables installed_manifest.json + freeze_candidate.json; completion: "Actual
  installed candidate and comparator verified; capability gaps retained".
- campaigns/una_10x/dossiers/09_final_integration.md: baseline and candidate wheels from
  immutable source; fresh NONEDITABLE environments; launch outside checkouts; assert imported
  module/binary hashes in parent AND workers; candidate must demonstrably engage its new route;
  refusal test exercises the old route too; source/binary changes after freeze invalidate
  dependent tests/timings.
- post_p00 designs/07_measurement_and_gates.md:4-6 — baseline 395cdc5 and P00 e7c9eda stay
  SEPARATE arms; the 10x comparator is 395cdc5 "or stronger verified runtime"; :53 — any
  source/harness/environment change invalidates affected frozen measurements (Q10 re-measures
  both arms post-freeze; prior files are kept, never edited).
- post_p00_readiness.json (dossier 07:58 prerequisite) is PRESENT at
  campaign_runs/una_10x/tenx-20261005T2331Z/post_p00_readiness.json.
- post_p00/designs/ holds only 01..07; dossier 09 lives in the original packet
  campaigns/una_10x/dossiers/09_final_integration.md (I00 "reads" path).

## Decision A1 — baseline-arm packaging parity (the A-wheel question)

Facts:
- 0acbb11 (BATCH_EXEC: parallel RunBatch runtime + top-level spawn shim) IS an ancestor of
  comparator 395cdc5; the comparator is NOT an ancestor of 0acbb11 (comparator is downstream).
- Comparator runtime.py:325 `from una_batch_worker_main import pool_main as _spawn_main`;
  runtime.py:169 `_transport_identity()` folds the installed shim's sha256 + python/numpy
  versions into the batch identity (worker-visible fingerprint already built in).
- Comparator pyproject [tool.hatch.build.targets.wheel] packages ONLY
  src/urban_network_analysis — NO force-include. A clean wheel from comparator source ships no
  spawn shim, so installed parallel RunBatch crashes with ModuleNotFoundError at pool.start —
  the exact defect N10 found in the candidate lineage. INHERITED from 0acbb11 into BOTH arms.
- Shim bytes identical across the lineage so far:
  git show 395cdc5:src/una_batch_worker_main.py sha256 =
  037d6c47c7ec3a083829f8e861fb278b77593c5ceacfe704fc31f89a37bb320f =
  working-tree src/una_batch_worker_main.py (shasum, same session).

Resolution: the A wheel is built in a DISPOSABLE comparator worktree whose ONLY tree delta vs
395cdc5 is the pyproject force-include block (src/una_batch_worker_main.py ->
una_batch_worker_main.py at wheel root). Declared in installed_manifest.json as a
build-config parity backport — packaging-only; the packaged shim must hash to 037d6c47...
(comparator's own bytes) at census time. Every other file byte-comparator-exact.

Rejected alternatives:
- editable baseline install — violates dossier 09 noneditable law;
- serial/no-cohort baseline as main comparator — dossier 07:5 forbids;
- backporting runtime fixes into the baseline — would alter the measured implementation.

## Decision A2 — no runtime backport to the baseline arm

The DEF-24 cancel_join_thread fix stays candidate-only. Comparator cancel semantics untouched.
Happy-path cohort legs (zero failed jobs) never enter cancel-with-backlog, so baseline legs
complete normally; the wedge remains a recorded comparator capability gap (X00 defect register),
retained per dossier 09 "capability gaps retained".

## Decision B1 — candidate arm = integrated HEAD at freeze

B wheel built from the integrated tree only AFTER G10R and DEF-24R verdicts are filed and any
findings/errata landed (unreviewed surfaces are not "accepted deltas"). pyproject force-include
already in this tree. Same census asserts as N10 S30b: spawn shim at wheel root, wheel
native/extension/sitecustomize-free.

## Decision B2 — installed identity asserts (parent AND workers)

- Parent probes (N10 S50 pattern): package __file__ under the install venv, not a checkout;
  una_batch_worker_main resolves from the wheel; no native extension importable; no repo
  sitecustomize on sys.path; UNA constructs.
- Worker side: both arms' runtime folds _transport_identity() (spawn_shim sha, python, numpy)
  into batch identity — the leg harness must surface it and the fire ladder asserts the
  recorded spawn_shim equals the wheel-census sha (037d6c47... for A; census sha for B).
  Exact field path in leg output to be confirmed during the read-only rehearsal (contract-probe
  before firing; read main()'s dispatch block).
- No research runtime imports: after a full leg, scan sys.modules of the installed interpreter
  for campaigns./campaign_runs. imports and run explicit import refusals.

## Scratch/venv plan (disposable, outside checkouts)

- /tmp/i00/{venv_a,venv_b}: cp -a venv-p01m, `python -m pip uninstall` editable first, then
  install the arm wheel (N10 S40 pattern; MetaPathFinder shadowing law; invoke as
  venv/bin/python -m pip — copied bin/pip shebangs point at the ORIGINAL venv).
- Fresh build venv venv-i00wheel (hatchling). hatchling has no -o and no comma targets:
  `-t wheel -t sdist`, in-tree dist/ moved out, rmdir (gitignored).
- Legs reuse p05s/run_screen_leg.py with the frozen cohort pairing
  campaign_data/inputs/upstream_Boston/Boston/local_audit_manifest.csv, w4/t1/q64, cache off,
  NUMBA_CACHE_DIR fresh per leg, PYTHONDONTWRITEBYTECODE=1, loadavg gate < 8 before public legs.
- All scratch under /tmp/i00; consoles sha-named per fire-launch template; budget charged in
  consumption_log at fire time.

## Open items at fire time

1. G10R verdict → land findings/errata before the B-wheel build.
2. DEF-24R verdict → same for runtime.py.
3. Re-hash at commit: readiness JSON, HEAD pin, every peer-authored artifact (intake re-hash
   law); figures only from same-block tool outputs (no hardcoded literals in S-stage scripts —
   the S80 lesson from N10 fire-1).
4. A-arm wheel RECORD differs from any B build by construction (different sources); the parity
   claim is about packaged MODULE bytes, asserted per-entry, not wheel-file identity.

## Rehearsal result (2026-10-07 ~05:00Z, pre-gate, /tmp only)

- A wheel BUILT from a git-archive export of 395cdc5 with the pyproject-only
  force-include delta: 61 entries / 56 .py; spawn shim at root, sha256
  037d6c47c7ec... == comparator blob; ALL packaged .py byte-identical vs
  comparator except generated _build_info.py (build hook, commit-date bake);
  native/ext/sitecustomize-free. Census: /tmp/i00/a_wheel_census.json
  (wheel sha256 f0504f886eb9...; full values in the census JSON).
- Install rehearsal in /tmp/i00/venv_a_rehearsal (cp -a venv-p01m, editable
  uninstalled, wheel installed): package AND shim resolve under the rehearsal
  venv, pool_main present, UNA constructs.
- Both artifacts are DISPOSABLE /tmp copies: the fire ladder re-derives them
  from the declared source and re-hashes at fire time (resume law); the
  rehearsal only proves the route.

## Resolution B2-final — worker-side identity WITHOUT changing the frozen leg policy

The batch journal (the only place _transport_identity persists) exists only when
ExecutionOptions.checkpoint is set; frozen-cohort legs run checkpoint-free, and adding
checkpoint to the timed legs would change the measured configuration. Resolution: the timed
legs stay frozen-policy; worker-side identity is captured by a SEPARATE untimed 2-row probe
per arm (sliced pairing in /tmp, checkpoint dir in /tmp): the runtime's own
journal.create(identity, identity_detail) records transport.spawn_shim from a REAL spawn
worker; the probe asserts it == the wheel-census shim sha. The same probe interpreter scans
sys.modules after a real batch for campaigns./campaign_runs./tests. prefixes (research-import
refusal). Parent-side: package __file__ + shim file sha under the install venv == census sha.
B wheel is built from `git archive HEAD` (committed bytes only; the author's in-flight test
file is uncommitted and cannot enter the wheel by construction).

## Fire outcome (2026-10-07, post-verdicts)

Executed after G10R + DEF-24R verdicts landed; results live in the generated artifacts
(installed_manifest.json, freeze_candidate.json, wheel_census.json, install_identity.json,
worker_identity_probes.json, legs/*.json) and consoles (sha-named .txt files here). Headlines:
identity chain complete per arm (wheel census shim == parent probe == real-spawn-worker journal
spawn_shim, 037d6c47c7ec... both arms), both legs 64 files / 147,829,892 B bitwise-equal to the
n00 L2A1 reference, zero failed jobs, research-import scan empty.

Two fire lessons (recorded where they bit, restated here for the design record):

1. RunBatch row settings are FRESH Settings() per row and resolve relative inputs at the CSV's
   own directory "regardless of the instance's data_folder" (UNA.py:202-226). The untimed
   worker probe therefore could NOT reuse a /tmp-sliced pairing by setting
   `settings.data_folder` (fire-1) nor by `cwd=` (fire-2 attempt 1); the fix is a row-level
   absolute `data_folder` column, which ApplyRow applies as an ordinary Settings field and
   which disables the CSV-dir anchor (UNA.py:225). Full history: installed_manifest.json
   fire_history; attempt-1/2 consoles were overwritten by reruns (failure signatures preserved
   verbatim in fire_history and the session transcript).
2. freeze_source must name the commit the B wheel was actually built from (fire-1 S10's
   HEAD_AT_FIRE = 1e7237b), not HEAD at manifest-generation time (5e2a0ef, the in-flight
   control-test commit; tests are unpackaged, outside the wheels — see
   installed_manifest.json in_flight_exclusions).

Known deviation at intake: the S50 probe subprocess does not pin NUMBA_CACHE_DIR (JIT cache
may land in the disposable /tmp venv site-packages); identity asserts are source-sha based,
and the script was not edited mid-run. The timed legs (S40/S41) DO pin NUMBA_CACHE_DIR fresh
per leg as designed.

