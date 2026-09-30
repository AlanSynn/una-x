# Branch, ownership, packaging and execution operations

Source baseline and publication parent: 16bf404d2cc5e776a78dc073c80405f6d186da29.
The packet commit changes no src/build/tests/benchmark runtime. At BOOT compare protected
source trees to source_pins.json. Expected documentation-only drift is recorded; numerical
source drift requires re-audit. Develop on perf/una-platform. Do not reset a dirty checkout.

Tasks with overlapping source paths use distinct immutable worktrees. Shared Settings,
UNA dispatch, exports, facade __init__ files and package metadata have one integration owner.
Workers return commits and proofs; they never push directly to main. Benchmark execution has
one exclusive owner. Model provider routing is recorded when visible, never inferred from
an alias. Missing independent review stays unavailable. No credentials are printed.

Packaging: maintain the base package/import and ordinary CPU-only installation. Put optional
native/GPU/drop-in-shim builds in separate distributions or justified platform wheels without
forcing compiler installation for ordinary binary users. Record actual optional dependency
versions and wheel tags. Do not install two distributions that both own madina without an
explicit isolated-environment migration. No hidden on-import compiler/download.

CI: add targeted CPU API/bitparity/fault tests and clean-wheel installation checks. Native
sanitizer and ARM/x86 platform builds are required for claims on those platforms. Actual GPU
self-hosted/authorized-runner jobs must fail or explicitly skip as unavailable on missing
hardware; a skipped GPU job is not release qualification. Never add workflows that buy compute,
expose secrets or run untrusted code with privileged tokens. Full matrix runs near final
integration; not after every exploratory patch. No YAML-only claim that CI passed.

Historical handling: campaigns/history retains immutable packets/evidence; prior large-e2e
paths remain accessible for old tests and source proof links but are marked historical and
its START_HERE redirects. Current goal/CLAUDE/registry select only this campaign. Fix test
hardcoding and append-only evidence behavior in new code, without editing old raw records.

After each completed task store state.json with reviewed dependency identities and ownership.
A resumed session revalidates these before using old evidence; do not rerun all large jobs
solely because conversational context was lost. Final cleanup is opt-in and preceded by a
custody manifest proving retained bytes, not just checksums. Production runtime must never
import campaign helpers or resolve private worktree paths.
