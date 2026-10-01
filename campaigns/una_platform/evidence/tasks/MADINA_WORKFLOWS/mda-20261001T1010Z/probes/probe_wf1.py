"""MADINA_WORKFLOWS scoping probe 1: upstream betweenness_flow_simulation
end-to-end on file fixtures in the reference venv. Decides:
  - does keep_diagnostics=True (hardcoded at the workflow call site) crash
    the workflow (FLOW pinned a post-engine TypeError)?
  - does pydeck to_html run and is its output deterministic?
  - which output files land, and what nc=min(origins, num_cores) picks?
Run: .refs/venv_madina_legacy/bin/python probe_wf1.py <data_folder> <out_folder>
"""
import json
import os
import sys
from pathlib import Path

data_folder = Path(sys.argv[1])
out_folder = Path(sys.argv[2])

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / ".refs" / "madina_ref" / "src"))
from madina.una.workflows import betweenness_flow_simulation  # noqa: E402

result = {"workflow_error": None, "out_files": None}
try:
    betweenness_flow_simulation(
        data_folder=str(data_folder),
        output_folder=str(out_folder),
    )
except Exception as exc:  # noqa: BLE001
    result["workflow_error"] = f"{type(exc).__name__}: {exc}"
finally:
    listing = {}
    for p in sorted(out_folder.rglob("*")):
        if p.is_file():
            listing[str(p.relative_to(out_folder))] = p.stat().st_size
    result["out_files"] = listing

print("PROBE_RESULT_JSON<<<")
print(json.dumps(result, indent=1))
print(">>>")
