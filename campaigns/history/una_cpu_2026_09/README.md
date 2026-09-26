# Historical stage: first UNA CPU campaign

Status: completed historical campaign, not active. Archived from main `361928e4ba38f34622cafe065b0025244db61368`. `packet/` reuses the exact original `optimization_una_cpu` tree `917475d210218a7f3ae2765d5b8e6a1c9c587842`; this includes raw evidence and binary wheels. `entrypoints/` preserves the former CLAUDE.md and /goal command blobs.

Historical source improvements remain in the runtime package. Archiving the packet does not revert ordered CSR. Existing `tests/perf_contract/` and `benchmarks/una_cpu/` remain at their old locations for compatibility, but their old sweep scripts are not the active benchmark plan.

Path translation for reading historical reports: a former repository-root `optimization_una_cpu/<suffix>` now means `campaigns/history/una_cpu_2026_09/packet/<suffix>`. Relative paths inside packet remain relative to packet. Absolute laptop paths in old scripts are provenance, not executable instructions. For exact historical reproduction, use a detached worktree at the archived source commit rather than editing the archive.

The historical RESULTS/decision claims are preserved verbatim. The active campaign SOURCE_AUDIT supplies corrections: generated grid versus observed-network distinction; diagnostic second constructor excluded from public-call totals; per-worker versus whole-pool memory; source-arm versus installed-wheel throughput. Do not edit old measurements to make these qualifications disappear.

All new evidence belongs to the active campaign. Do not write new decisions, qualification records, or outputs into this archive.
