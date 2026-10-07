#!/usr/bin/env python3
"""F00 fixture builder: deterministic bounded origins fixture.

Derivation (declared, recorded in the evidence JSON):
  * source  : campaign_data/inputs/upstream_Boston/Boston/
              Cambridge_building_centroids.geojson  (14,751 point features,
              file order = ascending 'fid' as shipped)
  * rule    : the FIRST N features in source file order, geometries and all
              original properties preserved verbatim (no reordering, no
              filtering, no coordinate change); N = 256 by default.
  * weights : settings keep origin_weight_column="Count" /
              destination_weight_column="Count" -> unit weights for every
              point (BuildAccessPoints sentinel; no NaN risk, declared).
  * destinations: the FULL MA_transit_stations.geojson layer is used
              unmodified (6,919 features); no destination fixture is created.

The fixture is a NEW declared bounded flow input (WORKLOADS.md L2 rule:
"deterministic sampling ... is a NEW declared fixture with retained
selected IDs").  It is written ONLY inside the f00/ directory.
"""
from __future__ import annotations

import hashlib
import json
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(source: str, out_path: str, n_origins: int) -> dict:
    with open(source, "r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"]
    if n_origins > len(feats):
        raise SystemExit(f"requested {n_origins} > source {len(feats)}")
    sel = feats[:n_origins]
    out = {
        "type": "FeatureCollection",
        "name": gj.get("name", "origins"),
        "crs": gj.get("crs"),
        "features": sel,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return {
        "source": source,
        "source_sha256": sha256_file(source),
        "source_features": len(feats),
        "selected_first_n": n_origins,
        "selection_rule": "first N features in source file order (fid ascending); verbatim properties and geometry",
        "fixture": out_path,
        "fixture_sha256": sha256_file(out_path),
        "fixture_features": len(sel),
        "first_fid": sel[0]["properties"].get("fid"),
        "last_fid": sel[-1]["properties"].get("fid"),
    }


if __name__ == "__main__":
    src = ("campaign_data/inputs/upstream_Boston/Boston/"
           "Cambridge_building_centroids.geojson")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    out = sys.argv[2] if len(sys.argv) > 2 else "f00_origins.geojson"
    info = build(src, out, n)
    print(json.dumps(info, indent=2))
