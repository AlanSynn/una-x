#!/usr/bin/env python3
from pathlib import Path
import json, sys
root=Path(__file__).resolve().parents[1]
required=[
"START_HERE.txt","CLAUDE_CODE_EXECUTION_PROMPT.md","DESIGN_AUTHORITY.md","SOURCE_AUDIT.md","PLAN.md",
"TASKS_CLAUDE.yaml","SEMANTIC_CONTRACT.md","NUMERICAL_CONTRACT.md","PERFORMANCE_MODEL.md",
"BENCHMARK_PROTOCOL.md","VALIDATION_POLICY.md","PROMOTION_POLICY.md","RESOURCE_POLICY.md",
"BRANCH_AND_CI.md","DECISION_CONTRACT.md","MERGE_SCOPE.md","PACKAGING_AND_DISTRIBUTION.md",
"DX_CONTRACT.md","WORKLOADS.md","CANDIDATE_REGISTER.md","EVIDENCE_LEDGER.md","DELEGATION.md","ARCHITECTURE.md",
"dossiers/01_oracle_and_source_bridge.md","dossiers/02_ordered_csr_integration.md",
"dossiers/03_genuine_fixtures_and_state.md","dossiers/04_public_api_benchmark_harness.md",
"dossiers/05_cpu_profile_and_resource_selection.md","dossiers/06_optional_private_scope_tail.md",
"dossiers/07_wheel_and_default_path.md","dossiers/08_final_freeze_review_and_merge.md",
"prior/una_optimization/patches/ordered_csr.patch","prior/una_optimization/REPORT.md"
]
missing=[p for p in required if not (root/p).is_file()]
tasks=json.loads((root/"TASKS_CLAUDE.yaml").read_text())
ids=[]
if isinstance(tasks,dict):
    seq=tasks.get("tasks",[])
elif isinstance(tasks,list):
    seq=tasks
else:
    seq=[]
for t in seq:
    if isinstance(t,dict) and "id" in t: ids.append(t["id"])
dups=sorted({x for x in ids if ids.count(x)>1})
result={"ok":not missing and not dups,"missing":missing,"duplicate_task_ids":dups,"task_count":len(ids)}
print(json.dumps(result,indent=2))
sys.exit(0 if result["ok"] else 1)
