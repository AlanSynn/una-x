# Local audit — T00 inventory disposition

Date: 2026-09-25. Role: source-auditor (coordinator-executed). Reviewer: coordinator.

## What was actually done

1. `python3 tools/validate_packet.py` → `{"ok": true, "task_count": 15}` (packet structure intact).
2. `python3 tools/verify_prior.py` → `{"ok": true}` (8 decisive prior files present and checksummed).
3. `python3 tools/probe_laptop.py` → arm64, 10 logical CPUs, macOS 26.6.2; memory/swap fields null because the system interpreter (3.14.7) has no psutil. Richer facts gathered directly via `sysctl`/`vm_stat` and re-recorded in `laptop.json`.
4. `python3 tools/audit_checkout.py ..` → HEAD `98f498e5b0ac97b05271d03812fa883b5e4afaba`, branch `AlanSynn/main`, status clean, origin `https://github.com/alansynn/una-x`.

## Findings that change packet assumptions (reconciliation records)

| Packet assumption | Observed fact | Reconciliation |
|---|---|---|
| Upstream commit `c15ebda…` is the audited baseline | The upstream commit object is **absent from the fork ODB** (seeded by content, not shared history) | Provenance proven by **content-level blob-hash bridge** (see `../contract/oracle_bridge.json`); 5/5 audited files match upstream blob IDs exactly |
| Packet written against `City-Form-Lab/urban_network_analysis` | Target fork is `AlanSynn/una-x`, main = `98f498e`, clean | Fork main SHA recorded; development on `perf/una-cpu-throughput` worktree only |
| Historical prior evidence ran under a 4-CPU cgroup sandbox | This laptop: Apple M1 Pro, 10 logical (8P+2E), 16 GiB, no quota | Historical 2×2 worker/thread configuration **not transplanted**; T07 re-derives settings on this machine |
| History contains the CSR candidate verified against pinned blobs | Fork ODB lacks upstream objects, so `verify_repository.py` object lookup cannot resolve upstream blobs from the fork alone | Blob bridge computed against working-tree files (content-addressed, equivalent check); recorded in oracle_bridge.json |

## Environment decisions

- Campaign interpreter: homebrew **Python 3.11** (`/opt/homebrew/bin/python3.11`) in `/Users/alansynn/orca/workspaces/una-x/venvs/campaign`. System 3.14 has no scientific stack and numba support for 3.14 is not the package's tested profile; 3.11 is inside the packet's stated "Python 3.11 or newer" and the project's `requires-python = ">=3.11"`.
- Admitted CPU budget **C = 8** (`Σ workers×threads ≤ 8`), conservatively excluding the 2 efficiency cores; no hard quota exists on macOS — this is a preregistered self-limit, recorded before any benchmark.
- The machine is **shared**: load average ~7.2 at inventory, swap 3.4/4.0 GiB used. Consequences recorded now: (a) all performance claims carry recorded load-average context; (b) baseline/candidate arms are always compared in interleaved sessions on the same loaded machine, so shared background load is common-mode; (c) memory gates use conservative margins against 16 GiB shared.

## Preserved user work

No stash, no checkout, no file in the user's main worktree was modified. Everything campaign-written lives in `wt-integration` (branch `perf/una-cpu-throughput`) and `venvs/` + scratch dirs outside the repositories.

## Disposition

T00 complete. No unavailable gate blocks dependent tasks; laptop facts recorded; baseline immutable SHA chosen = `98f498e5b0ac97b05271d03812fa883b5e4afaba` (fork main == AlanSynn/main, upstream content bridged at blob level).
