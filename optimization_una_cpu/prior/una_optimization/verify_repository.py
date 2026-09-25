#!/usr/bin/env python3
from pathlib import Path
import subprocess, json, sys
root=Path(sys.argv[1] if len(sys.argv)>1 else ".").resolve()
expected={
"src/urban_network_analysis/Engines/Accessibility.py":"246fe439591379038c5e73d0388f0720726213e9",
"src/urban_network_analysis/Engines/AccessibilityWElevation.py":"69061684380ffa9d890b75804bbcd3b4642eae90",
}
rows=[]
ok=True
for rel,want in expected.items():
    p=root/rel
    if not p.is_file():
        rows.append({"path":rel,"ok":False,"reason":"missing"}); ok=False; continue
    got=subprocess.run(["git","hash-object",str(p)],text=True,capture_output=True).stdout.strip()
    good=got==want
    rows.append({"path":rel,"expected_blob":want,"actual_blob":got,"ok":good})
    ok &= good
print(json.dumps({"ok":ok,"files":rows},indent=2))
sys.exit(0 if ok else 1)
