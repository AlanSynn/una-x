"""H05: derive O3_FLOW origin fixtures (sel256, sel1024) from pinned selection
manifests. sel_all is the identity selection (ordered_indices == 0..N-1, verified
here), so the frozen source file is used directly and no new fixture is written.

Light task (no UNA import, no B0 execution): gpd read of the 6 MB origins layer,
row subsetting by pinned ordered_indices, per-row attribute cross-check,
GeoJSON write, round-trip verify. Follows the proven h04_derive_o2.py pattern.

Run:
  campaign-python campaigns/una_large_e2e/evidence/H05/h05_derive_origins.py
Outputs (campaign_data, outside the repo):
  /Users/alansynn/orca/workspaces/una-x/campaign_data/O3_FLOW_origins_sel256.geojson
  /Users/alansynn/orca/workspaces/una-x/campaign_data/O3_FLOW_origins_sel1024.geojson
  /Users/alansynn/orca/workspaces/una-x/campaign_data/h05_origin_fixture_derivation.json
"""
import hashlib
import json
import os
import sys

import geopandas as gpd

REPO = "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e"
INPUTS = f"{REPO}/tests/large_e2e/inputs"
SRC = "/Users/alansynn/orca/workspaces/una-x/campaign_data/inputs/Cambridge_building_centroids.geojson"
SRC_SHA = "14999756f30cc0f3245ac104b64f807dce76765e7c0e28af1da499c892d3bcdc"
OUT_DIR = "/Users/alansynn/orca/workspaces/una-x/campaign_data"
VARIANTS = ["O3_FLOW_sel256", "O3_FLOW_sel1024", "O3_FLOW_sel_all"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    record = {"task": "H05", "record": "origin_fixture_derivation",
              "role": "performance-owner", "source": SRC, "source_sha256_expected": SRC_SHA,
              "variants": []}
    src_sha = sha256_file(SRC)
    if src_sha != SRC_SHA:
        record["fatal"] = f"source sha mismatch {src_sha}"
        print(record["fatal"])
        sys.exit(2)
    record["source_sha256_verified"] = src_sha

    gdf = gpd.read_file(SRC)
    n = len(gdf)
    record["source_feature_count"] = n

    for name in VARIANTS:
        mpath = f"{INPUTS}/{name}.origin_indices.json"
        with open(mpath) as f:
            sel = json.load(f)
        entry = {"selection_manifest": mpath, "selection_manifest_sha256": sha256_file(mpath),
                 "requested_k": sel["requested_k"], "selected_count": sel["selected_count"]}
        if sel["source_sha256"] != SRC_SHA or sel["source_feature_count"] != n:
            entry["status"] = "REFUSED_source_identity_mismatch"
            record["variants"].append(entry)
            continue
        ordered = sel["ordered_indices"]
        if sel["requested_k"] == "all":
            identity = ordered == list(range(n))
            entry.update({"identity_selection": identity,
                          "fixture": SRC, "fixture_sha256": SRC_SHA,
                          "fixture_note": "identity selection: use the frozen source file "
                                          "directly; no derived fixture written",
                          "status": "ok_identity" if identity else "REFUSED_not_identity"})
            record["variants"].append(entry)
            continue
        if sorted(ordered) != ordered or len(set(ordered)) != len(ordered):
            entry["status"] = "REFUSED_indices_not_ascending_unique"
            record["variants"].append(entry)
            continue
        sub = gdf.iloc[ordered].reset_index(drop=True)
        # per-row attribute cross-check against the source rows
        mismatches = []
        for j, i in enumerate(ordered):
            a = gdf.iloc[i].drop("geometry")
            b = sub.iloc[j].drop("geometry")
            if not a.equals(b):
                mismatches.append({"row": j, "src_index": i})
        out = f"{OUT_DIR}/{name.replace('O3_FLOW_sel', 'O3_FLOW_origins_sel')}.geojson"
        sub.to_file(out, driver="GeoJSON")
        rt = gpd.read_file(out)
        rt_ok = len(rt) == len(ordered) and all(
            rt.iloc[j].drop("geometry").equals(sub.iloc[j].drop("geometry"))
            for j in range(len(ordered)))
        entry.update({
            "fixture": out, "fixture_sha256": sha256_file(out),
            "fixture_bytes": os.path.getsize(out),
            "attribute_mismatches": mismatches,
            "roundtrip_rows": len(rt), "roundtrip_ok": rt_ok and len(rt) == len(ordered),
            "status": "ok" if (not mismatches and rt_ok) else "REFUSED_crosscheck_failed",
        })
        record["variants"].append(entry)
        print(f"[h05] {name}: {entry['status']} -> {out}")

    outrec = f"{OUT_DIR}/h05_origin_fixture_derivation.json"
    with open(outrec, "w") as f:
        json.dump(record, f, indent=1)
    print(f"[h05] derivation record: {outrec}")
    if not all(v.get("status", "").startswith(("ok", "REFUSED")) for v in record["variants"]):
        sys.exit(2)


if __name__ == "__main__":
    main()
