# Branch and CI policy

Target: `AlanSynn/una-x`.
Development branch: `perf/una-cpu-throughput`.
Upstream provenance: `City-Form-Lab/urban_network_analysis@c15ebda6981397f46eed5c2d55229f71e57d44fb`.

At campaign start record current main SHA, merge base, remotes, status, and untracked files. Never discard or overwrite user work. If main has moved, reconcile packet assumptions against source before patching.

Create isolated worktrees from immutable SHAs for baseline, candidate, reviewer, packaging, and performance-owner roles. One writable owner per source scope. Only the integrator writes reviewed commits to the integration branch.

Do not push, open PRs, merge main, release, rewrite history, force-clean, rebase user work, change credentials, or modify global provider configuration without explicit later authorization.

CI sequence: focused local L0/L1 -> family tests -> genuine L2 -> installed-wheel gate -> bounded L3 -> final candidate -> required hosted CI near merge/release. Do not trigger broad hosted CI after every experiment. Never call an unrun check passed.

Before merge-ready status, create a disposable worktree at the intended target main SHA, apply/merge the reviewed candidate exactly as proposed, run required local checks and installed-wheel verification, and record the resulting tree/commit relationship.