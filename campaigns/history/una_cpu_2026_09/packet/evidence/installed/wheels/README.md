# Wheel preservation (post-campaign cleanup)

Copied here at user-requested cleanup time so the evidence ledger stays
self-contained after `/tmp` scratch removal. These are the exact artifacts
recorded in `installed/manifest.json` and `freeze/candidate.json`:

| wheel | sha256 | manifest role |
|---|---|---|
| `base/urban_network_analysis-2.6.0-py3-none-any.whl` | `653f6017e31de40fa5738f52f9557b259673289403261c1d28354d828c5246f6` | baseline wheel (built from disposable worktree at 98f498e) |
| `cand/urban_network_analysis-2.6.0-py3-none-any.whl` | `8b944b3cc4fb5a8721f027973739fb05b6c0d931186213456a5d59d8cbfdda02` | frozen candidate wheel (source identical to f57825d) |

sha256 re-verified at copy time against the manifest values. The
rehearsed-merge wheel was proven byte-identical to the candidate wheel
(`integration/merge/rehearsal.json`).
