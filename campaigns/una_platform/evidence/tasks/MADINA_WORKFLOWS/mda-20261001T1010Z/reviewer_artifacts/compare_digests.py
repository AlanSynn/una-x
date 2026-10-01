"""Independent digest comparison (reviewer): my fresh run vs retained set,
and facade vs reference content modulo arm tags."""
import json
import sys
from pathlib import Path

R = Path("/storage/scratch1/1/dsynn6/una-x/campaigns/una_platform/evidence/tasks/MADINA_WORKFLOWS/mda-20261001T1010Z/artifacts/arm_runs")
M = Path("/tmp/una_wf_review_artifacts/madina_workflows")

def strip(d):
    d = json.loads(json.dumps(d))
    d.pop("arm", None); d.pop("scenario", None); d.pop("sabotage", None)
    d.get("env_marker", {}).pop("workflows_file", None)
    return d

files = sorted(p.name for p in (R / "facade").glob("*.json"))
print(f"{len(files)} digest files in retained facade arm")
bad = 0
for f in files:
    ret_f = (R / "facade" / f).read_text()
    ret_r = (R / "reference" / f).read_text()
    mine_f = (M / "facade" / f).read_text()
    mine_r = (M / "reference" / f).read_text()
    row = []
    if ret_f != mine_f:
        row.append("facade:mine!=retained")
    if ret_r != mine_r:
        row.append("reference:mine!=retained")
    # facade vs reference content modulo tags, on MY run
    if strip(json.loads(mine_f)) != strip(json.loads(mine_r)):
        row.append("facade!=reference(modulo tags)")
    if row:
        bad += 1
        print("  DIFF", f, row)
print("all digest files: my-run == retained AND facade==reference modulo tags"
      if bad == 0 else f"{bad} files with differences")

# reflection
for arm in ("facade", "reference"):
    a = (R / "reflection" / f"{arm}.json").read_text()
    b = (M / "reflection" / f"{arm}.json").read_text()
    print(f"reflection/{arm}: my-run == retained: {a == b}")
rf = strip(json.loads((M / "reflection" / "facade.json").read_text()))
rr = strip(json.loads((M / "reflection" / "reference.json").read_text()))
print("reflection facade==reference modulo tags:", rf == rr)
if rf != rr:
    keys = set(rf) | set(rr)
    for k in sorted(keys):
        if rf.get(k) != rr.get(k):
            print("  ", k, "\n    FAC:", rf.get(k), "\n    REF:", rr.get(k))
sys.exit(0)
