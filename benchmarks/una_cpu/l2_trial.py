"""L2 genuine public-API trial driver (T06).

For each leg, runs the SAME frozen W1 public job against the immutable
baseline source (twice — determinism gate) and the candidate source (once),
then compares:
  - result artifacts byte-for-byte (content hashes; timestamped folder
    names are output policy, not content, and are normalized away),
  - in-process observable state (resolved gravity cap, result lengths),
  - error-path exception type+message (l2_error_probe.py).

Writes evidence JSON: legs, hashes, verdicts. No timings are recorded here;
L2 is a correctness tier.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _env(src_root: Path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = str(src_root)
    env["NUMBA_NUM_THREADS"] = "4"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    return env


def _run_job(src_root: Path, analysis: str, fixture_dir: Path, out_root: Path,
             overrides=None):
    """Run one W1 public job from the given source tree."""
    out_root.mkdir(parents=True, exist_ok=True)
    outputs = out_root / "outputs"
    outputs.mkdir(exist_ok=True)
    sys.path.insert(0, str(HERE))
    from workload_gen import settings_for
    settings = settings_for("W1_small_genuine", fixture_dir, outputs, analysis)
    for k, v in (overrides or {}).items():
        settings[k] = v
    spec = {"job_id": f"l2_{analysis}", "analysis": analysis,
            "settings": settings, "assert_mode": "source",
            "source_root": str(src_root), "output_root": str(outputs)}
    spec_path = out_root / "spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f)
    proc = subprocess.run(
        [sys.executable, str(HERE / "job.py"), "--spec", str(spec_path),
         "--result", str(out_root / "result.json")],
        capture_output=True, text=True, env=_env(src_root),
        cwd=str(out_root), timeout=3600)
    assert proc.returncode == 0, (
        f"job failed for src={src_root}\nSTDOUT tail:\n{proc.stdout[-2000:]}\n"
        f"STDERR tail:\n{proc.stderr[-3000:]}")
    with open(out_root / "result.json") as f:
        result = json.load(f)
    # normalize artifact hashes: strip the timestamped policy folder layer
    normalized = {}
    for entry in result["output_files"]:
        parts = Path(entry["path"]).parts
        rel = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
        normalized[str(rel)] = {"sha256": entry["sha256"],
                                "bytes": entry["bytes"]}
    result["normalized_artifacts"] = normalized
    return result


def _run_odm(src_root: Path, fixture_dir: Path, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    outputs = out_root / "outputs"
    outputs.mkdir(exist_ok=True)
    driver = out_root / "odm_driver.py"
    with open(driver, "w") as f:
        f.write(
            "import sys, json\n"
            "sys.path.insert(0, {here!r})\n"
            "from workload_gen import settings_for\n"
            "import numpy as np\n"
            "from urban_network_analysis import UNA\n"
            "settings = settings_for('W1_small_genuine', {fix!r}, {out!r}, 'accessibility')\n"
            "settings.update(origin_uid_column='point_id', "
            "destination_id_column='point_id', search_radius=350)\n"
            "una = UNA(verbosity=0)\n"
            "for k, v in settings.items():\n"
            "    setattr(una.settings, k, v)\n"
            "una.settings.knn_weights = np.asarray(una.settings.knn_weights)\n"
            "una.RunODM(format='feather', speed=5.0)\n"
            "print('@@ODM@@' + json.dumps({{'ok': True}}))\n"
            .format(here=str(HERE), fix=str(fixture_dir), out=str(outputs)))
    proc = subprocess.run([sys.executable, str(driver)], capture_output=True,
                          text=True, env=_env(src_root), cwd=str(out_root),
                          timeout=3600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    sys.path.insert(0, str(HERE))
    from workload_gen import hash_dir
    normalized = {}
    for entry in hash_dir(outputs):
        parts = Path(entry["path"]).parts
        rel = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
        normalized[str(rel)] = {"sha256": entry["sha256"],
                                "bytes": entry["bytes"]}
    return {"normalized_artifacts": normalized}


def _run_probe(src_root: Path, fixture_dir: Path, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(HERE / "l2_error_probe.py"),
         str(fixture_dir), str(out_root)],
        capture_output=True, text=True, env=_env(src_root),
        cwd=str(out_root), timeout=1800)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines()
            if ln.startswith("@@PROBE@@")][-1]
    return json.loads(line[len("@@PROBE@@"):])


def _compare(a, b):
    na = a.get("normalized_artifacts", {})
    nb = b.get("normalized_artifacts", {})
    if set(na) != set(nb):
        return {"match": False, "reason": "artifact set mismatch",
                "only_a": sorted(set(na) - set(nb)),
                "only_b": sorted(set(nb) - set(na))}
    diffs = [k for k in na if na[k]["sha256"] != nb[k]["sha256"]]
    state_keys = ("reach_len", "edge_flow_len", "resolved_gravity_cap")
    state_diffs = [k for k in state_keys if a.get(k) != b.get(k)]
    return {"match": not diffs and not state_diffs,
            "artifact_diffs": diffs, "state_diffs": state_diffs,
            "n_files_compared": len(na)}


def _compare_probe(a, b):
    if set(a) != set(b):
        return {"match": False, "reason": "probe case set mismatch"}
    for k in a:
        if a[k] != b[k]:
            return {"match": False, "case": k, "a": a[k], "b": b[k]}
    return {"match": True, "cases": sorted(a)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-src", type=Path, required=True)
    ap.add_argument("--candidate-src", type=Path, required=True)
    ap.add_argument("--fixture-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    legs = []
    work_root = args.out / "work"

    leg_defs = [
        ("accessibility_w1", "accessibility", None),
        ("accessibility_w1_noelev", "accessibility", {"elevation": False}),
        ("flow_w1_gravitycap", "flow", None),
    ]
    for leg, analysis, overrides in leg_defs:
        leg_out = work_root / leg
        base_a = _run_job(args.baseline_src, analysis, args.fixture_dir,
                          leg_out / "base_a", overrides)
        base_b = _run_job(args.baseline_src, analysis, args.fixture_dir,
                          leg_out / "base_b", overrides)
        cand = _run_job(args.candidate_src, analysis, args.fixture_dir,
                        leg_out / "cand", overrides)
        det = _compare(base_a, base_b)
        cmp = _compare(base_a, cand)
        legs.append({
            "leg": leg, "analysis": analysis, "overrides": overrides,
            "baseline_determinism": {"match": det["match"],
                                     "detail": {k: v for k, v in det.items()
                                                if k != "match"}
                                     if not det["match"] else None},
            "baseline_vs_candidate": cmp,
            "baseline_state": {k: base_a.get(k) for k in
                               ("reach_len", "edge_flow_len",
                                "resolved_gravity_cap")},
            "n_artifacts": len(base_a.get("normalized_artifacts", {})),
        })
        print(f"[{leg}] determinism={det['match']} "
              f"candidate_match={cmp['match']}", flush=True)

    odm_out = work_root / "odm_w1"
    base_a = _run_odm(args.baseline_src, args.fixture_dir, odm_out / "base_a")
    base_b = _run_odm(args.baseline_src, args.fixture_dir, odm_out / "base_b")
    cand = _run_odm(args.candidate_src, args.fixture_dir, odm_out / "cand")
    det = _compare(base_a, base_b)
    cmp = _compare(base_a, cand)
    legs.append({"leg": "odm_w1", "analysis": "odm",
                 "baseline_determinism": {"match": det["match"]},
                 "baseline_vs_candidate": cmp,
                 "n_artifacts": len(base_a.get("normalized_artifacts", {}))})
    print(f"[odm_w1] determinism={det['match']} candidate_match={cmp['match']}",
          flush=True)

    probe_out = work_root / "error_paths"
    pb = _run_probe(args.baseline_src, args.fixture_dir, probe_out / "base")
    pc = _run_probe(args.candidate_src, args.fixture_dir, probe_out / "cand")
    pcmp = _compare_probe(pb, pc)
    legs.append({"leg": "error_paths", "baseline_vs_candidate": pcmp,
                 "baseline_cases": pb})
    print(f"[error_paths] candidate_match={pcmp['match']}", flush=True)

    all_ok = all(
        l["baseline_determinism"]["match"]
        and l["baseline_vs_candidate"]["match"] for l in legs)
    summary = {"all_match": all_ok, "legs": legs}
    with open(args.out / "l2_results.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("ALL_MATCH" if all_ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
