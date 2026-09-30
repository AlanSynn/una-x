"""Validate the authored campaign, task DAG, requirements and content manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{40}")

def validate(root: Path) -> dict:
    root = Path(root).resolve()
    checks = []
    def ck(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
    required = ["README.md", "START_HERE.txt", "AUTHORITY.md", "EXECUTION.md", "CONTRACTS.md",
        "NUMERICS.md", "SOURCE_AUDIT.md", "MODEL.md", "BENCHMARKS.md", "RESOURCES.md",
        "HARNESS.md", "VALIDATION.md", "EVIDENCE.md", "DECISION.md", "WORKLOADS.md", "OPERATIONS.md",
        "TASKS_CLAUDE.yaml", "API_PARITY.json", "incidents.json", "source_pins.json", "sources.json",
        "campaign.json", "packet_files.json"]
    for rel in required:
        ck("file:" + rel, (root / rel).is_file())
    if any(not x["passed"] for x in checks):
        return {"passed": False, "checks": checks}
    campaign = json.loads((root / "campaign.json").read_text())
    dag = json.loads((root / "TASKS_CLAUDE.yaml").read_text())
    tasks = [dict(dag.get("task_defaults", {}), **task) for task in dag["tasks"]]
    ids = [t["id"] for t in tasks]
    ck("task_ids_unique", len(set(ids)) == len(ids))
    ck("task_count", campaign["task_count"] == len(tasks) and len(tasks) > 0)
    ck("branch_consistency", dag["integration_branch"] == campaign["integration_branch"] == "perf/una-platform")
    ck("baseline_sha", SHA.fullmatch(campaign["source_baseline_sha"]))
    ck("madina_sha", SHA.fullmatch(campaign["madina_reference_sha"]))
    ck("start_prompt_size", 1000 <= len((root / "START_HERE.txt").read_text()) <= 5500)
    by_id = {t["id"]: t for t in tasks}
    for t in tasks:
        required_fields = {"id", "depends_on", "role", "worktree_scope", "base_commit", "owned_files", "read_context",
            "objective", "constraints", "proof_obligations", "validation_tier", "resource_budget", "outputs", "reviewer", "completion_condition", "mandatory"}
        ck("fields:" + t["id"], required_fields <= set(t))
        ck("dependencies:" + t["id"], all(d in by_id and d != t["id"] for d in t["depends_on"]))
        ck("base:" + t["id"], t["base_commit"] == "reviewed_dependency_closure" or bool(SHA.fullmatch(t["base_commit"])))
        for rel in t["read_context"]:
            path = (root / rel).resolve()
            ck("context:" + t["id"] + ":" + rel, path.is_relative_to(root) and path.is_file())
        ck("nonempty_owned:" + t["id"], bool(t["owned_files"]))
        ck("role:" + t["id"], (root / "agents" / (t["role"] + ".md")).is_file())
    done = set()
    for _ in range(len(tasks) + 1):
        added = {t["id"] for t in tasks if set(t["depends_on"]) <= done}
        if added <= done:
            break
        done |= added
    ck("acyclic_dag", len(done) == len(tasks))
    for name, owners in campaign["requirements"].items():
        ck("requirement:" + name, bool(owners) and all(i in by_id and by_id[i]["mandatory"] for i in owners))
    needed = {"madina_api", "scientific_fixes", "city_failure_investigation", "public_parallel_runbatch",
              "caching", "native_backend", "gpu_backend", "bitwise_parity", "end_to_end_frontier", "installed_and_review"}
    ck("required_capabilities", needed <= set(campaign["requirements"]))
    api = json.loads((root / "API_PARITY.json").read_text())
    ck("seed_not_falsely_complete", api["complete_census"] is False and len(api["rows"]) >= 18)
    pins = json.loads((root / "source_pins.json").read_text())
    for ref in ("una", "madina"):
        ck("source_pins:" + ref, bool(pins[ref]["files"]) and all(SHA.fullmatch(x) for x in pins[ref]["files"].values()))
    manifest = json.loads((root / "packet_files.json").read_text())
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        good = path.is_relative_to(root) and path.is_file()
        if good:
            good = hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        ck("digest:" + item["path"], good)
    return {"passed": all(c["passed"] for c in checks), "check_count": len(checks), "task_count": len(tasks),
            "scope": "campaign consistency and supplied file identity only; no runtime qualification", "checks": checks}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    result = validate(a.root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)

if __name__ == "__main__":
    main()
