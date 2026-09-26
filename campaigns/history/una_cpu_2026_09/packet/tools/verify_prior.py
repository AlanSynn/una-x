#!/usr/bin/env python3
from pathlib import Path
import json, sys
root=Path(__file__).resolve().parents[1]
paths=[
"prior/una_optimization/REPORT.md",
"prior/una_optimization/patches/ordered_csr.patch",
"prior/una_optimization/results/final_batch.csv",
"prior/una_optimization/results/final_batch_jobs.csv",
"prior/una_optimization/results/final_components.csv",
"prior/una_optimization/results/environment_final.json",
"prior/una_optimization/results/tests_final.txt",
"prior/una_optimization/results/final_memory.json",
]
missing=[p for p in paths if not (root/p).is_file()]
print(json.dumps({"ok":not missing,"preserved_summary_files":paths,"missing":missing,
"note":"This GitHub handoff preserves the decisive prior summaries/patch. Redundant per-job artifact directories and the binary NPZ fixture from the original 21 MB packet are not embedded."},indent=2))
sys.exit(0 if not missing else 1)
