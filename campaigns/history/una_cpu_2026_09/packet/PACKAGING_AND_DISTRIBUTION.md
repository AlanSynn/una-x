# Packaging and distribution

The package remains `urban-network-analysis`, import `urban_network_analysis`, Python >=3.11, with existing ordinary dependencies. The CPU CSR candidate uses NumPy/Python already present; it must not require a new compiler, binary download, vendor runtime, or user environment variable.

Build baseline and candidate wheels from immutable clean source states. Install each into a fresh environment without editable mode. Verify:
- package metadata/version behavior,
- import location is the installed environment, not repository source,
- public imports and examples still work,
- candidate optimized path actually executes when admitted,
- unsupported-domain test reaches the trusted original path,
- CSV/GeoJSON/Feather dependencies and writers behave as baseline.

Do not count source-tree execution as wheel qualification. Do not count a fallback run as proof that the optimized path is packaged.

If packaging metadata must change for a correctness reason, separate that commit/evidence from the performance claim and re-run installed-path gates.