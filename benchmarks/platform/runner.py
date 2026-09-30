"""Parent-side runner: one real job, engagement leg, batch windows
(HARNESS.md "One real job", "Batch run"; BENCHMARKS.md boundaries).

The timed run is uninstrumented.  A separate engagement run executes the
same plan with a dispatch trace and reports requested-vs-effective backend
evidence; bitwise equality of the two runs' outputs is the binding that the
traced kernel is the same semantic kernel as the timed kernel (validation
isolation).  Baseline determinism is that same comparison when the plan is
a reference run.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .errors import (BatchRunFailed, EngagementError, IdentityError,
                     ModelDispatchError, ResourceLedgerError)
from .identity import scrubbed_env
from .manifest import resource_policy_of
from .sampler import ProcessTreeSampler

HARNESS_VERSION = "una-platform-2026-09-h1"

BOOTSTRAP = (
    "import sys; sys.path.insert(0, sys.argv[sys.argv.index('--repo') + 1]); "
    "from benchmarks.platform import child as _c; raise SystemExit(_c.main(sys.argv))"
)

# Which public entry point each requested analysis must trace to.
_DISPATCH_EXPECTATION = {
    "accessibility": "UNA.RunAccessibility",
    "od": "UNA.RunODM",
    "flow": "UNA.RunFlow",
    "batch": "UNA.RunBatch",
}


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


class _MarkerReader:
    """Reads the child's JSONL marker file as it grows."""

    def __init__(self, path: Path):
        self.path = path
        self.records: list[dict] = []

    def drain(self) -> None:
        try:
            with open(self.path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn tail line while writer is mid-append
                    if rec not in self.records:
                        self.records.append(rec)
        except FileNotFoundError:
            pass

    def get(self, marker: str) -> dict | None:
        for rec in self.records:
            if rec.get("marker") == marker:
                return rec
        return None


def _child_command(plan_path: Path, marker_path: Path, repo: Path,
                   python_exe: str):
    return [python_exe, "-c", BOOTSTRAP,
            "--repo", str(repo), "--plan", str(plan_path),
            "--markers", str(marker_path)]


def _run_child(plan: dict, workdir: Path, repo: Path, python_exe: str,
               timeout_s: float = 600.0, sample_memory: bool = True,
               extra_env: dict | None = None) -> tuple[int, _MarkerReader,
                                                       float, dict | None]:
    """Spawn one child process for *plan*.

    Returns (exitcode, markers, wall, sampler_stats).  The sampler sums the
    simultaneous parent+descendant RSS of the child tree while it runs;
    per-worker maxima are never summed.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    plan_path = workdir / "plan.json"
    marker_path = workdir / "markers.jsonl"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if marker_path.exists():
        marker_path.unlink()
    reader = _MarkerReader(marker_path)
    env_vars = {
        "NUMBA_CACHE_DIR": str(workdir / "numba_cache"),
        "PYTHONHASHSEED": "0",
    }
    if plan.get("child_env"):
        env_vars.update(plan["child_env"])
    if extra_env:
        env_vars.update(extra_env)
    env = scrubbed_env(env_vars)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        _child_command(plan_path, marker_path, repo, python_exe),
        cwd=str(workdir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sampler = None
    if sample_memory:
        sampler = ProcessTreeSampler(proc.pid, interval_s=0.2, label="child")
        sampler.start()
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        reader.drain()
        stats = sampler.stop() if sampler is not None else None
        raise BatchRunFailed(
            f"child exceeded timeout_s={timeout_s}; censored, not "
            f"deadlocked-proof (sampler={stats and stats['samples_taken']})")
    wall = time.perf_counter() - t0
    stats = sampler.stop() if sampler is not None else None
    reader.drain()
    if proc.returncode not in (0, 3):
        detail = (err or b"").decode(errors="replace")[-2000:]
        child_err = next((r.get("error", "")[-1200:] for r in reader.records
                          if r.get("marker") == "child_error"), "")
        raise BatchRunFailed(
            f"child exited {proc.returncode}: {detail}\n"
            f"child_error={child_err}\n"
            f"markers={[(r.get('marker')) for r in reader.records]}")
    return proc.returncode, reader, wall, stats


def _assert_dispatch(manifest: dict, markers: _MarkerReader) -> list[str]:
    """Requested analysis must be the executed public entry point (control 1)."""
    rec = markers.get("dispatch")
    if rec is None:
        raise ModelDispatchError("engagement leg produced no dispatch marker")
    called = rec.get("called") or []
    expected = _DISPATCH_EXPECTATION[manifest["model"]["analysis"]]
    short = expected.split(".")[-1]
    if not any(c.split(".")[-1] == short for c in called):
        raise ModelDispatchError(
            f"requested analysis {manifest['model']['analysis']!r} expects "
            f"{expected} but trace captured {sorted(set(called))}")
    forbidden = {"UNA.RunAccessibility", "UNA.RunODM", "UNA.RunFlow",
                 "UNA.RunBatch"} - {expected}
    for entry in called:
        if entry in forbidden or entry.split(".")[-1] in {
                f.split(".")[-1] for f in forbidden}:
            raise ModelDispatchError(
                f"requested {expected} but trace also captured {entry}")
    return called


def _assert_engagement(manifest: dict, markers: _MarkerReader) -> dict:
    """A claimed backend needs substantive counters (control 3)."""
    rec = markers.get("backend")
    if rec is None:
        raise EngagementError("no backend marker from engagement leg")
    effective = rec.get("effective")
    counters = rec.get("counters") or {}
    requested = manifest["backend_requested"]
    if requested in ("native", "gpu"):
        key = f"{requested}_module_loaded"
        if not counters.get(key):
            raise EngagementError(
                f"backend reported {effective!r} but no {requested} module "
                f"loaded and no work counters (counters={counters})")
    if effective not in ("reference", "native", "gpu"):
        raise EngagementError(f"effective backend {effective!r} invalid")
    return {"requested": requested, "effective": effective,
            "counters": counters}


def run_one_real_job(manifest: dict, *, repo: str, python_exe: str,
                     out_dir: str | Path, run_id: str | None = None,
                     timeout_s: float = 600.0, mutant: str | None = None,
                     child_env: dict | None = None,
                     outdir_guard=None) -> dict:
    """Execute the frozen manifest as one real public-API job.

    Returns the full measurement record (validated elsewhere).  Raises the
    first failed check; timed vs engagement legs are separate child
    processes so the trace never instruments the timed boundary.
    `mutant` names an intentionally broken candidate (negative controls).
    """
    from .guards import OutputDirGuard
    from .identity import resolve_identity
    run_id = run_id or new_run_id()
    repo = str(Path(repo).resolve())
    out_dir = Path(out_dir)
    guard = outdir_guard or OutputDirGuard(out_dir, run_id, repo)
    guard.prepare()

    # pre-flight installed identity for the record (the child re-checks
    # independently; a mismatch there is the shadowing negative control)
    package = manifest.get("expected_identity", {}).get(
        "package", "urban_network_analysis")
    identity_info = resolve_identity(python_exe, package)

    base_plan = {
        "harness": {"repo": repo, "package": manifest.get(
            "expected_identity", {}).get("package", "urban_network_analysis")},
        "model": manifest["model"],
        "settings_patch": manifest["settings_patch"],
        "backend_reported": manifest["backend_requested"],
        "run": {"out_dir": str(out_dir)},
    }
    if mutant:
        base_plan["mutant"] = mutant
    if child_env:
        base_plan["child_env"] = child_env

    # ---- leg 1: timed, uninstrumented --------------------------------
    timed_dir = out_dir / "leg_timed"
    t_launch0 = time.perf_counter()
    plan = dict(base_plan, run={"out_dir": str(timed_dir)})
    code, m1, wall, mem_stats = _run_child(plan, out_dir / "_work_timed",
                                           Path(repo), python_exe, timeout_s)
    t_launch = time.perf_counter() - t_launch0
    if code == 3:
        ident = m1.get("identity") or {}
        raise IdentityError(
            f"child identity rejected: module_file={ident.get('module_file')} "
            "(source shadowing the installed wheel)")

    # ---- leg 2: engagement (traced, separately timed) -----------------
    eng_dir = out_dir / "leg_engagement"
    t_eng0 = time.perf_counter()
    eng_plan = dict(base_plan,
                    run={"out_dir": str(eng_dir), "engagement_leg": True})
    code2, m2, wall_eng, _ = _run_child(eng_plan, out_dir / "_work_engagement",
                                        Path(repo), python_exe, timeout_s,
                                        sample_memory=False)
    wall_eng_total = time.perf_counter() - t_eng0

    called = _assert_dispatch(manifest, m2)
    engagement = _assert_engagement(manifest, m2)

    # outputs of both legs must be byte-identical (same-kernel binding)
    from .validation import compare_dirs_bitwise
    compare_dirs_bitwise(timed_dir, eng_dir,
                         label="timed-vs-engagement (same semantic kernel)")

    imported = m1.get("imported") or {}
    compute = m1.get("compute_done") or {}
    outputs = m1.get("outputs") or {}
    synced = m1.get("export_synced")

    record = {
        "run_id": run_id,
        "harness_version": HARNESS_VERSION,
        "campaign": manifest.get("campaign"),
        "profile": manifest.get("profile"),
        "model": manifest["model"],
        "workload_class": manifest["workload"].get("class"),
        "requested_backend": engagement["requested"],
        "effective_backend": engagement["effective"],
        "backend_counters": engagement["counters"],
        "dispatch_called": sorted(set(called)),
        "timings": {
            "definition": "perf_counter; launch->validated-completion is "
                          "launch_to_exit_s + post_hash_s",
            "child_wall_s": wall,
            "child_import_s": imported.get("t_import"),
            "child_compute_s": compute.get("t_compute"),
            "engagement_wall_s": wall_eng_total,
            "launch_to_exit_s": t_launch,
            "post_hash_s": None,  # filled by the caller after hashing
        },
        "cache_state": {
            "numba_cache": "cold",
            "definition": "fresh NUMBA_CACHE_DIR per leg; no cache attach",
        },
        "export_synced": bool(synced),
        "outputs": outputs.get("files", []),
        "output_dir": str(out_dir),
        "child_identity": {
            **(m1.get("identity") or {}),
            "python": identity_info.get("python"),
            "python_version": identity_info.get("python_version"),
            "installed_tree_sha256": identity_info.get(
                "installed_tree_sha256"),
            "distribution_version": identity_info.get(
                "distribution_version"),
            "installed_files_sha256": identity_info.get(
                "installed_files_sha256"),
        },
        "input_hashes": manifest.get("_verified_input_hashes"),
        "settings_fingerprint": manifest["settings_patch"],
        "resource_policy": manifest.get("resource_policy"),
        "memory": mem_stats,
        "counts": {"completed": 1, "failed": 0, "cancelled": 0},
        "fault_status": "none",
        "source_shadow_rejected": False,
    }
    return record


# --------------------------------------------------------------------------
# bounded batch runner (controls 6 and 10; later used by public RunBatch L3)
# --------------------------------------------------------------------------

def run_bounded_batch(job_factories: list, policy: dict, work_root: Path,
                      repo: str, *, deadline_s: float = 60.0,
                      on_child_start=None, enforce_policy: bool = True,
                      declared_policy: dict | None = None) -> dict:
    """Bounded-submission, concurrent-drain batch window over child jobs.

    job_factories: callables (index) -> Popen-ready command list; each must
    exit 0 for a validated success.  Submissions are bounded by
    policy['queue_depth']; at most policy['max_workers'] children run at
    once; results drain concurrently while submission continues; every exit
    code is watched.  A child death with the queue full fails the window
    within deadline_s (no deadlock, no unbounded retry), and remaining
    children are cleaned up.
    """
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    max_workers = int(policy["max_workers"])
    queue_depth = int(policy["queue_depth"])
    if enforce_policy and declared_policy is not None:
        if (max_workers != int(declared_policy["max_workers"])
                or queue_depth != int(declared_policy["queue_depth"])):
            raise ResourceLedgerError(
                f"enforced policy (workers={max_workers}, "
                f"queue={queue_depth}) diverges from declared "
                f"{declared_policy} (control 10)")

    running: dict[int, subprocess.Popen] = {}
    results: list[dict] = []
    next_job = 0
    failures: list[str] = []
    t0 = time.perf_counter()
    peak_inflight = 0
    aborted = False

    def _record_exit(idx: int, rc: int) -> None:
        results.append({"job": idx, "exitcode": rc,
                        "t": time.perf_counter() - t0})
        if rc != 0:
            failures.append(f"job {idx} exited {rc}")

    try:
        while (next_job < len(job_factories) or running) and \
                not aborted and \
                time.perf_counter() - t0 < deadline_s:
            # submit while in-flight < min(max_workers, queue_depth)
            limit = min(max_workers, queue_depth)
            peak_inflight = max(peak_inflight, len(running))
            while (next_job < len(job_factories) and len(running) < limit):
                idx = next_job
                cmd = job_factories[idx](idx)
                job_dir = work_root / f"job{idx}"
                job_dir.mkdir(parents=True, exist_ok=True)
                proc = subprocess.Popen(cmd, cwd=str(job_dir))
                running[idx] = proc
                if on_child_start is not None:
                    on_child_start(idx, proc)
                next_job += 1
            # drain concurrently: poll every running child
            for idx in list(running):
                rc = running[idx].poll()
                if rc is None:
                    continue
                del running[idx]
                _record_exit(idx, rc)
                if rc != 0:
                    # bounded fail: cancel the rest now, watch their exits
                    for j, p in running.items():
                        p.kill()
                    for j, p in list(running.items()):
                        try:
                            _record_exit(j, p.wait(timeout=5))
                        except subprocess.TimeoutExpired:
                            failures.append(f"job {j} unkillable")
                            p.kill()
                            _record_exit(j, p.wait())
                    running.clear()
                    aborted = True
        leftover_deadline = time.perf_counter() + 5.0
        while running and time.perf_counter() < leftover_deadline:
            for idx in list(running):
                rc = running[idx].poll()
                if rc is not None:
                    del running[idx]
                    results.append({"job": idx, "exitcode": rc,
                                    "t": time.perf_counter() - t0})
                    if rc != 0:
                        failures.append(f"job {idx} exited {rc}")
            time.sleep(0.02)
        for idx, p in running.items():
            p.kill()
            failures.append(f"job {idx} killed at deadline")
    finally:
        for idx, p in running.items():
            if p.poll() is None:
                p.kill()
        for idx, p in list(running.items()):
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    wall = time.perf_counter() - t0
    successes = [r for r in results if r["exitcode"] == 0]
    return {
        "wall_s": wall,
        "deadline_s": deadline_s,
        "submitted": next_job,
        "results": sorted(results, key=lambda r: r["job"]),
        "validated_successes": len(successes),
        "failures": failures,
        "peak_inflight": peak_inflight,
        "bounded": True,
        "status": "complete" if (not failures and
                                 len(successes) == len(job_factories))
                   else "failed",
    }


def assert_window_validated(window: dict) -> None:
    """Failed/cancelled/missing jobs invalidate the window (HARNESS.md)."""
    if window["status"] != "complete":
        raise BatchRunFailed(
            f"window not validated: {window['failures']} "
            f"({window['validated_successes']}/{window['submitted']} ok)")


def enforce_ledger(window: dict, declared_policy: dict,
                   sampler_stats: dict | None = None) -> None:
    """Declared vs enforced resource audit (control 10)."""
    if window["peak_inflight"] > int(declared_policy["max_workers"]):
        raise ResourceLedgerError(
            f"observed {window['peak_inflight']} concurrent workers > "
            f"declared {declared_policy['max_workers']}")
    if window["peak_inflight"] > int(declared_policy["queue_depth"]):
        raise ResourceLedgerError(
            f"observed {window['peak_inflight']} in-flight > declared queue "
            f"depth {declared_policy['queue_depth']}")
    limit = declared_policy.get("memory_limit_bytes")
    if limit is not None and sampler_stats is not None:
        peak = sampler_stats.get("peak_simultaneous_rss_bytes", 0)
        if peak > int(limit):
            raise ResourceLedgerError(
                f"simultaneous tree peak {peak} > memory_limit_bytes {limit}")


def sampler_for_child(proc_pid: int, interval_s: float = 0.2) -> ProcessTreeSampler:
    s = ProcessTreeSampler(proc_pid, interval_s=interval_s)
    s.start()
    return s


def run_engagement_leg(base_plan: dict, work_dir: str | Path, repo: str,
                       python_exe: str, timeout_s: float = 600.0,
                       child_env: dict | None = None) -> "_MarkerReader":
    """Run only the traced engagement leg (fast negative-control path).

    Returns the marker reader for _assert_dispatch/_assert_engagement.
    """
    work_dir = Path(work_dir)
    eng_dir = work_dir / "leg_engagement"
    plan = dict(base_plan, run={"out_dir": str(eng_dir),
                                "engagement_leg": True})
    if child_env:
        plan["child_env"] = child_env
    code, m, _wall, _stats = _run_child(plan, work_dir / "_work", Path(repo),
                                        python_exe, timeout_s,
                                        sample_memory=False)
    if code == 3:
        ident = m.get("identity") or {}
        raise IdentityError(
            f"child identity rejected: module_file={ident.get('module_file')} "
            "(source shadowing the installed wheel)")
    return m
