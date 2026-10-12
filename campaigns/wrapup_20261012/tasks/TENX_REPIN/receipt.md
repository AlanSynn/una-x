# TENX_REPIN receipt — tenx island failure groups closed on the merge train

Branch `perf/wrapup-train`, base `48d2c7d9c465ca21c48279961fd18902b9f98127`,
interpreter `/storage/home/hcoda1/1/dsynn6/micromamba/envs/alan/bin/python`,
`PYTHONHASHSEED=0`, 2026-10-12. Evidence/test-only change; ZERO src/ changes.

## Summary

| Island | Before | After |
|---|---|---|
| tests/tenx/identity | 3 failed / 44 passed | **47 passed / 0 failed** |
| tests/tenx/metrics | 2 failed / 67 passed | **69 passed / 0 failed** |
| tests/tenx/trace | 1 failed / 54 passed, 2 skipped | **55 passed, 2 skipped / 0 failed** |
| tests/tenx/native | 1 failed / 85 passed | **86 passed / 0 failed** |

7 failures -> 0. One repair attempt per failure group; the only mid-task
regression (a dropped import in test_native_fold.py, caught by the island
re-run as 83 NameError failures) was reverted before any commit and the
island re-run green.

## Task A — identity table generation 2 (6 failures)

Root cause (confirmed mechanically, not assumed): the preserved generation-1
table `campaigns/una_10x/identity/metric_dependencies.json` pins engine sha
`efc4dd4a74dc...0df36` and call-site linenos 395/435/472. The gated
lazy-import change (exl-20261011T164228Z) removed the dead module-level
`import networkx as nx` from
`src/urban_network_analysis/Engines/AccessibilityWElevation.py`, shifting
every recorded C10 call site by exactly -1 and flipping the file sha.

Mechanical verification (the tests' own probe
`tests/tenx/identity/_engine_facts.py`, run on the live bytes — NOT
hand-copied):

- live engine sha256: `9b2225079f292469eb8d1276c241bf897466a937c45fc4fb1dec7db65fe1288a`
- live call sites: `[394, 434, 471]` == table keys - 1 (asserted)
- `fold` facts equal to the preserved record modulo `engine_sha256` (asserted, all
  params/outputs/branches/decorators identical)
- per_site binding VALUES identical at the -1 rekey (asserted, all 39 expressions)
- bindings unanimous; alias bindings and wrapper derivation consistent (asserted)
- numba 0.67.0 / numpy 2.4.6 == preserved numerical_env (no version drift)

Disposition: generation 2 written by the checked-in generator
`make_gen2.py` (this directory), which derives every changed value from
`_engine_facts.py` and HARD-asserts the delta is exactly the known shift
before writing; it aborts on any mismatch beyond it.

- New file `campaigns/una_10x/identity/metric_dependencies.gen2.json`
  (sha256 `eb11aeb8cb80e1b3f8c35ace8916ff1f3ad61ea7ef33e942f9dbeee200701666`)
  = preserved table + live fold + rekeyed call_sites + live
  numerical_env.engine_sha256 + `generation` metadata block (generation 2,
  generated_commit 48d2c7d9..., date, generator, lineage pointing at
  exl-20261011T164228Z / G4RC g4rc-20261012T0512Z, generation_1_sha256).
  Leaf-level delta vs gen1: 1 (fold sha) + 79 (call_sites rekey: 39 old
  per_site leaves out, 39 in, linenos) + 1 (env sha) + 8 (metadata). Nothing else.
- Generation 1 PRESERVED untouched: worktree bytes ==
  `48d2c7d:campaigns/una_10x/identity/metric_dependencies.json`
  (sha256 `3d86e4c55fad531c5f59a9c33c19215057ae3abd1dc48ed272da756ff4068595`),
  `git diff 48d2c7d -- <gen1>` empty. NOTE for reviewers: the briefed guard
  `git diff c1f8614..HEAD -- <gen1>` cannot be empty for ANYone on this train —
  the file does not exist at c1f8614 (authored in be5e9d0 on perf/una-10x, first
  merged onto the train after c1f8614); the enforceable form, satisfied here, is
  the empty delta vs the pre-work base plus no commit of mine touching gen1.
- Load points repointed (assertion LOGIC untouched, generation-lineage comment
  added at each): tests/tenx/identity/conftest.py:50-56 (`table` fixture),
  tests/tenx/metrics/conftest.py:82-88 (`table` fixture),
  tests/tenx/trace/test_trace_gates.py:11-14 (docstring) and :30-35 (DEPS_PATH).
- Other stale pins: grep of tests/tenx for `efc4dd4a`, `9b222507`, and the old
  linenos 395/435/472 found NOTHING outside the three load points — no test
  hardcodes the sha or linenos; historical sha records under campaigns/evidence
  trees are preserved records and were not touched.

## Task B — native fold pin law (1 failure)

Root cause: `tests/tenx/native/test_native_fold.py::test_so_is_the_measured_bits`
pinned the WHOLE-FILE sha256 of
`src/urban_network_analysis/Engines/_native/libuna_metric_fold.so`. That
digest is build-variant by construction: recorded builds of the identical
`metric_fold.cpp` give Z00 measurement `2fe93a29...54e40bd6`, battery island
build `e1fab518...`, battery forced rebuild `95d7dd3c...81c8c98` (the observed
failure value), and this task's fresh deterministic build
`1d9c3b4e6f18fe3cc3078645307bff6d489b84a2f460089e3695505c7e4dae64` — four
builds, four whole-file digests.

Recoverability of Z00-era section digests: NOT RECOVERABLE. The N00 deliverable
(`campaign_runs/una_10x/tenx-20261004T155640Z/n00_native/native_implementation.json`)
records only whole-file shas (cpp `a82a58c9...497e132`, so `2fe93a29...`), no
section digests; the Z00 .so bytes were never preserved (the .so is an
uncommitted build artifact per the P00 raw-evidence law; no .so exists anywhere
under campaigns/ or evidence/ in either checkout). The task's fallback branch
applies: re-pin to section digests of a FRESH deterministic build.

What pins identity now (all recorded in the test's `_CODE_PIN` with the law
comment; digests computed by a self-contained ELF64 reader in the test, no
binutils dependency):

- `.text` section: `f0103bfa146f50a7856ce5eff0f8a06f0602545ee8e85f9c3e8f79a81ad5b834`
- R+E PT_LOAD segment file bytes: `5d0e5172cfce1a57b1fbed77e1195ec0c4e8775b3f78d13e0625380782097c1c`
- `.init`+`.plt`+`.text`+`.fini` concatenated: `042e3d735d4128ef2edf325b964027294bed5d1e1a2e9d64d07b11f2232416c2`

Law evidence on this artifact: the forced rebuild (95d7dd3c, preserved at
/tmp/fold_build_A_95d7dd3c.so for this task, sha above) and the fresh
`ensure_built(force=True)` build via the bridge's own recorded BUILD_CMD
(g++ (Spack GCC) 12.3.0, same command string as N00) agree bit-for-bit on all
three code digests while differing in whole-file sha — the in-the-wild proof
that code digests are build-invariant. Consistent with the NATIVE_CORE
precedent on this same train (natc-20261011T1821Z: whole-file deltas are XALT
link metadata only; code sections stable across builds/links).

Source identity to the measured build is retained: `metric_fold.cpp` on the
train is sha256 `a82a58c94e8590cea893fd02013d63a426caf8aa48abf25600b81762a497e132`
== the N00 deliverable's recorded cpp sha; the gxx fingerprint
(`g++ (Spack GCC) 12.3.0`) matches `_measured_map.FINGERPRINT`. The stronger
behavioral check is untouched: the 72-row measured map
(`test_measured_map_live_reproduction`) re-verified live on the fresh build,
and `_measured_map.py` is unmodified (its FINGERPRINT so_sha256 stays as the
historical Z00 measurement record, no longer consumed by the test).

Disposition of the on-disk build artifact: my fresh deterministic build
overwrote the untracked `libuna_metric_fold.so` (95d7dd3c -> 1d9c3b4e
whole-file). It is a content-keyed, untracked build artifact; its loadable
code is unchanged (digests above), and it remains untracked.

## Files changed (both commits)

| File | sha256 (final) |
|---|---|
| campaigns/una_10x/identity/metric_dependencies.gen2.json (NEW) | eb11aeb8cb80e1b3f8c35ace8916ff1f3ad61ea7ef33e942f9dbeee200701666 |
| campaigns/wrapup_20261012/tasks/TENX_REPIN/make_gen2.py (NEW) | see commit |
| tests/tenx/identity/conftest.py | 645924e629222bdabc63097126ace60339576af44cac1172d7daed4830eaf1e7 |
| tests/tenx/metrics/conftest.py | da640bd0b79102615a401d62b50e77366f91cd642400286c97e6fe4625c40e0c |
| tests/tenx/trace/test_trace_gates.py | b692a499df2bf0517f606cfa8d1c081fae9a8c176ab1f4345d2dd9f85d33f151 |
| tests/tenx/native/test_native_fold.py | 4e34e1a408a14c135df907f78988704bc2803feb8dd4c4146c435fafcb6673ba |

## Guards

- `git diff 48d2c7d..HEAD -- src/` EMPTY (src bytes identical to the
  gate-measured bytes). The briefed `9d82a0d` form shows 12 files / 2275
  insertions — those are the train's own committed NATIVE_CORE / compat-d3
  merges between 9d82a0d and the base 48d2c7d, predating this task; not mine.
- `git diff 48d2c7d -- campaigns/una_10x/identity/metric_dependencies.json` empty.
- Untracked battery artifacts still untracked:
  `src/.../Engines/_native/libuna_metric_fold.so`,
  `src/.../backends/native/_native_core.cpython-311-x86_64-linux-gnu.so`.
- No bare whole-repo pytest; islands run one at a time.

## Reviewer re-check list

1. Re-run `make_gen2.py` from a clean checkout of the commit and diff the
   output against the committed gen2 (byte-identical expected on this
   toolchain; json is dumped sorted, indent=2).
2. Confirm gen1 is untouched by the commits
   (`git diff 48d2c7d..HEAD -- campaigns/una_10x/identity/metric_dependencies.json`).
3. Re-run the four islands (one at a time, PYTHONHASHSEED=0): identity 47,
   metrics 69, trace 55+2s, native 86.
4. Recompute the three `_CODE_PIN` digests on a fresh `ensure_built(force=True)`
   build and against `readelf -lW` (single R+E LOAD, 0xc61 bytes at file offset
   0x1000) — must match the pins.
5. Confirm the trace/metrics/identity assertion logic is unchanged modulo the
   load path + comment (diff is 4 lines per file).
