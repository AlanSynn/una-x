"""Q00 errata regeneration: protected-cell inputs v2.

The frozen generator (q00/generate_q00.py, commit ee5652a) omitted the CSV
HEADER ROW in write_rows(): pandas then consumed the first data row as the
header (pc1 parsed 0 records; pc5 parsed 1 of 2 with garbled keys) and the
first protected leg died with RuntimeError("No settings to run").
Frozen bytes are kept, never edited (errata law): v1 files stay in place with
their sha256s still recorded in freeze.json; this regenerator writes
pc*_v2.csv with the header row + row-level absolute data_folder column and
the errata record binds the supersession.
"""
import csv
import hashlib
import json

MAIN = "/Users/alansynn/orca/workspaces/una-x/main"
Q00 = f"{MAIN}/campaign_runs/una_10x/tenx-20261005T2331Z/q00"
PAIRING = f"{MAIN}/campaign_data/inputs/upstream_Boston/Boston/local_audit_manifest.csv"
DATAFOLDER = f"{MAIN}/campaign_data/inputs/upstream_Boston/Boston"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


rows = list(csv.reader(open(PAIRING)))
hdr, body = rows[0], rows[1:]
assert len(body) == 32
i_rad = hdr.index("search_radius")


def write_v2(name, picked):
    path = f"{Q00}/inputs/{name}.csv"
    with open(path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(hdr + ["data_folder"])          # THE FIX: header row first
        for r in picked:
            cw.writerow(r + [DATAFOLDER])
    return path, sha(path)


specs = {
    "pc1_tiny_gis_beta00_v2": [body[0]],
    "pc2_tiny_gis_beta31_v2": [body[31]],
    "pc3_search_radius1200_v2": [[*body[0][:i_rad], "1200", *body[0][i_rad + 1:]]],
    "pc4_search_radius1600_v2": [[*body[0][:i_rad], "1600", *body[0][i_rad + 1:]]],
    "pc5_cold_2row_v2": body[0:2],
}
out = {}
for name, picked in specs.items():
    p, s = write_v2(name, picked)
    out[name] = {"file": p, "sha256": s}

# verify: pandas parses the intended record count with real keys
import pandas as pd
for name, want in (("pc1_tiny_gis_beta00_v2", 1), ("pc2_tiny_gis_beta31_v2", 1),
                   ("pc3_search_radius1200_v2", 1), ("pc4_search_radius1600_v2", 1),
                   ("pc5_cold_2row_v2", 2)):
    recs = pd.read_csv(out[name]["file"], dtype=str).to_dict("records")
    assert len(recs) == want, (name, len(recs))
    assert recs[0].get("network_file") == "20260703_PercLenNetwork_InnerCore.geojson"
    assert recs[0].get("data_folder") == DATAFOLDER
print("V2 INPUTS OK", {k: v["sha256"][:12] for k, v in out.items()})

errata = {
 "schema": "una-10x-q00-errata-1",
 "date": "2026-10-07",
 "subject": "protected-cell derived inputs (NOT the primary workload)",
 "defect": "generate_q00.py write_rows omitted the CSV header row; pd.read_csv consumed "
           "the first data row as header (pc1: 0 records, pc5: 1 of 2, garbled keys); "
           "Q10 fire-1 protected leg pc1 died with RuntimeError('No settings to run')",
 "evidence": "venv-p01m pandas reproduction on the frozen v1 files (records 0/1/1/1/1); "
             "console_fire_q10.log LEG FAIL tag='q10_pc1_tiny_gis_beta00_a b' rc=1 "
             "(also documents the secondary zsh $ARMS non-split defect, fixed in fire_q10b.sh)",
 "correction": "pc1..pc5 regenerated WITH header + row-level absolute data_folder column "
               "as *_v2.csv (this regenerator); v1 bytes kept in place, never edited; "
               "freeze.json protected_inputs_sha256 remains the record of v1; v2 shas "
               "below supersede for protected-cell execution only",
 "v2_inputs_sha256": {k: v["sha256"] for k, v in out.items()},
 "primary_untouched": "workload_sha256 235405de67cf22dc76c3e89bd0c3d198927d0ed0870172a0e86b942fecb8a60a "
                      "is the frozen cohort manifest; the 7 primary windows completed before "
                      "this errata and are unaffected",
 "note": "no self-hash (self-reference is undefined); this file's sha256 is recorded by "
         "q10/raw_index.json and the q10b commit message after the bytes are final",
}
with open(f"{Q00}/errata_pc_inputs.json", "w") as f:
    f.write(json.dumps(errata, indent=1) + "\n")
print("ERRATA WRITTEN", sha(f"{Q00}/errata_pc_inputs.json")[:16])
