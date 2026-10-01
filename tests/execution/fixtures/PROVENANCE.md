# Fixture provenance

`network.geojson`, `origins.geojson`, `destinations.geojson` are the frozen
3x3 smoke grid built by campaign task HARNESS
(`campaigns/una_platform/evidence/tasks/HARNESS/har-20260930T153809Z/fixture/`,
sha256 in that task's `manifest.json` under `workload.input_hashes` — the
copies here hash identically).  They are duplicated into `tests/` so the
EXECUTION suite is self-contained and does not depend on evidence-retention
paths.  Settings that drive a real run on this grid are recorded in the same
manifest under `settings_patch` (network weight column `Geometric`, origin /
destination weight column `Count`, `search_radius=1000`).
