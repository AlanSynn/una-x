# Dossier 08: final freeze, review, merge rehearsal

Freeze one candidate SHA after tuning. No source change after final performance evidence without invalidating affected evidence.

Independent reviewer inspects immutable diff, equivalence argument, fallback domain, tests, raw timings, memory, packaging, errors, and claim boundaries. Required changes return to owner and trigger revalidation.

Create disposable worktree at current target main SHA and rehearse the exact intended integration. Build/install wheel from rehearsed source and run required local correctness/genuine tests.

Produce merge_manifest.json with source branch/SHA, target SHA, merge base, file list, runtime path/fallback, tests, installed result, performance/memory evidence, actual-target status, review/CI status, limitations and disposition.

Do not actually merge/push/release unless user separately authorizes.