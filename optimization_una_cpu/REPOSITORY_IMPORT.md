# Repository import note

This fork contains the executable engineering packet, authoring guide, task DAG, contracts, dossiers, Claude role prompts, CSR patch, and decisive prior benchmark/test/environment summaries needed to continue the campaign.

The original chat-generated packet also contained redundant per-job CSV/GeoJSON artifact directories and one binary NPZ synthetic fixture. Those bulk artifacts are not embedded in this GitHub handoff because the available repository connector writes text files. They were historical synthetic evidence, not production qualification. Recreate development fixtures from immutable source when needed and treat the preserved summary/raw CSV files as historical evidence only.

The fork package source itself was copied from the audited upstream commit. Large upstream tutorial datasets are also not vendored here; pin/download the required genuine tutorial inputs by upstream commit/file hash during workload setup rather than silently using mutable latest data.
