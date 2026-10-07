# N00 measurement intent (filed 2026-10-06, agent n00-native-engineer)

Benchmark-host coordination notice per the campaign law: this file is
the pre-timed-leg intent. NO timed leg has been run. The timed legs
below start only after the team-lead's go lands in this agent's
mailbox (single-owner benchmark host).

## Claim under measurement
PUBLIC 10x boundary accounting for the trace-producer region: the
native AOT build leg replaces the singleflight first-trace serial
block (p04i: build_s 3.476s of a 16.8253s full-stack wall, 3 of 4
workers waiting). The region's ceiling is bounded by the full-stack
wall; the leg measures the real boundary, not a prototype ratio.

## Planned legs (identical to the stub list in native_implementation.json)
- L0 cold build+install cost (declared cold boundary, untimed legs).
- L1 C-arm: control cohort w4/t1/q64/cache-off, native hook installed,
  timer brackets ONLY public RunBatch, fresh NUMBA_CACHE_DIR + outdir,
  PYTHONDONTWRITEBYTECODE=1, census 64 files / 147829892 B.
- L2 numba control leg or p05s control receipt reuse (host-dependent).
- L3 bitwise cross-leg determinism (digests + census equal; mismatch
  voids the leg).

## Resource parity
Same host, same control cohort bytes, same schedule flags as p05s;
per-leg walls recorded exactly from the timer brackets, never
estimated.

## Abort conditions
Any census or digest mismatch vs the numba route -> leg void, verdict
stays stubbed. Any other benchmark agent active on the host -> do not
start.
