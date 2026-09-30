# Evidence custody, schema and provenance

New outputs belong under campaigns/una_platform/evidence/<TASK>/<RUN_ID>/, never historical
paths. Every run directory is create-new; fail on overwrite. A record includes source/wheel/
binary hashes, reference identity, numerical profile, input/settings identities, command,
environment, status, timestamps, measurement boundaries and raw relative artifact paths.
Only facts with actual execution receipts are measured. Label all numbers as measured,
derived_from_measured, hypothetical or unavailable. Selection defaults/budgets are policy,
not measured facts. Reference issue statements are reported, not reproduced.

Retain decisive raw JSONL/CSV, paired summaries, memory traces, correctness logs, source patches,
compiler metadata and minimized failures. A checksum of purged data cannot support a fresh
recomputation claim. Compress text/binary numerical evidence where useful. Large GIS inputs
may be re-fetched from immutable content-pinned authorized sources; retain licenses and exact
transformation manifests. Do not publish private city layers without permission. No credentials,
absolute private home paths or expiring signed URLs in committed evidence.

Changing numerical code invalidates affected goldens/crossover records, but never rewrites
old evidence. Append superseding records with exact parent identities. Review is tied to an
immutable source digest, not a moving branch name. Final docs-only commits may change metadata
without rerunning numerical tests when dependency closure is explicitly unchanged.

Required final files: RESULTS.md, decision.json, api_matrix.json, bitparity.json,
bug_deltas.json, incidents.json, accepted_rejected.json, performance.json, configuration.json,
commands.jsonl, raw_manifest.json, installed.json, review.json, ci.json, merge_manifest.json.
Every required feature status is machine-readable; unsupported/blocked dimensions are listed.
All replay commands must work without a particular historical session/worktree name.
