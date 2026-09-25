# Dossier 07: wheel and execution path

Build immutable baseline and candidate wheels using project packaging. Install into separate clean environments. Run import/version smoke test outside repository and prove import path is site-packages.

Candidate qualification must demonstrate:
- ordered-CSR helper is packaged,
- admitted genuine input executes optimized path,
- unsupported crafted input executes original fallback,
- no packet/reference module is imported by runtime,
- CSV/GeoJSON/Feather writers work,
- no new user install steps/dependencies.

Instrument path selection only in a test/debug mechanism outside final performance timing, or infer from controlled monkeypatch/counter in a dedicated non-performance test. Remove temporary production instrumentation before freeze.