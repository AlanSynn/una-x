# Dossier 03: genuine fixtures and state

Create immutable small GIS fixtures or pin upstream tutorial files by SHA. Exercise CRS, LineString network topology, origins/destinations snapping, optional elevation and obstacle paths, and a flow run whose gravity cap is derived by an accessibility pass.

Capture baseline:
- network node/edge arrays and access-point terminal indices/weights,
- engine result arrays and dtypes,
- relevant Settings mutations and readiness flags,
- gravity-cap value/writeback,
- CSV/GeoJSON/Feather schemas, metadata and values,
- expected warnings/errors for invalid cases.

Use baseline generated only by immutable original source. Candidate must compare against it under same dependencies and controlled timestamp policy. Do not accept final-result equality if intermediate state used later differs.