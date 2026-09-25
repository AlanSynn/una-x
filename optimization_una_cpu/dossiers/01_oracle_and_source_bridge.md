# Dossier 01: oracle and source bridge

Objective: prove that the fork baseline corresponds to the audited upstream numerical implementation before candidate qualification.

Actions:
1. Record fork main SHA and git status.
2. Compare Git blob hashes for baseline Accessibility.py and AccessibilityWElevation.py against upstream audited blobs.
3. Inspect the historical extracted reference and patch context in prior evidence.
4. Run baseline tiny synthetic CSR/search cases directly from source.
5. Build a baseline wheel and record import path/version.

If the baseline source differs materially from upstream c15ebda, do not blind-apply the historical patch. Re-derive the candidate against current source and invalidate old direct-extraction evidence. Completion requires a source reconciliation record and immutable oracle SHA.