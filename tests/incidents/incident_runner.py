"""Bounded incident-attempt runner (FAILURES / dossier 03 diagnostic
protocol).

Each attempt executes in a spawned child process with:

* an INDEPENDENT parent-side watchdog on the wall clock (a worker stuck
  in C code cannot stop it),
* child heartbeats (phase / completed units / high-water RSS / last
  marker) at a bounded cadence on a child daemon thread,
* on deadline: faulthandler stack dump requested via SIGUSR1, a
  cooperative cancel flag, a bounded grace, SIGTERM to OWNED pids only,
  join/reap (SIGKILL only if the grace expires), result marked
  "incomplete",
* process-tree simultaneous-RSS sampling reused from the HARNESS
  sampler, plus RLIMIT_AS / RLIMIT_FSIZE safety nets in the child.

The attempt capsule (envelope, log, traceback, stack dump, heartbeat,
result) is preserved verbatim under the attempt directory.  A timeout is
recorded as censored data ("incomplete"), never as a root cause or a
finite runtime.
"""
from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmarks.platform.sampler import ProcessTreeSampler  # noqa: E402

DEFAULT_TIMEOUT_S = 180.0
DEFAULT_GRACE_S = 5.0
DEFAULT_HEARTBEAT_S = 2.0
DEFAULT_RLIMIT_AS_BYTES = 8 * 1024 ** 3      # safety net, not the primary bound
DEFAULT_RLIMIT_FSIZE_BYTES = 512 * 1024 ** 2  # per-file disk safety net

_CHILD_TEMPLATE = textwrap.dedent("""
    import faulthandler, json, sys, threading, time, os, resource
    from pathlib import Path
    repo, attempt_dir, payload_path, params_path = sys.argv[1:5]
    sys.path.insert(0, repo)
    attempt_dir = Path(attempt_dir)
    log = (attempt_dir / "child_log.txt").open("w")
    faulthandler.enable(log)
    def _dump(signum, frame):
        faulthandler.dump_traceback(file=log)
        log.flush()
    import signal as _signal
    _signal.signal(_signal.SIGUSR1, _dump)

    state = {"phase": "boot", "units_done": 0, "last_marker": "start"}
    hb_path = attempt_dir / "heartbeat.jsonl"
    stop_hb = threading.Event()
    def _hb():
        import resource
        while not stop_hb.is_set():
            try:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                rec = {"t": time.time(), **state, "ru_maxrss_bytes": rss}
                with hb_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\\n")
            except Exception:
                pass
            stop_hb.wait(float(os.environ.get("INCIDENT_HEARTBEAT_S", "2.0")))
    threading.Thread(target=_hb, daemon=True).start()

    import runpy
    payload_ns = runpy.run_path(payload_path)
    params = json.loads(Path(params_path).read_text())
    cancel_path = attempt_dir / "cancel.flag"
    params["cancel_file"] = str(cancel_path)
    state["phase"] = "payload"
    try:
        result = payload_ns["payload"](params, state)
        state["phase"] = "done"
        (attempt_dir / "result.json").write_text(json.dumps(
            {"status": "completed", "result": result}, default=str))
    except BaseException as exc:
        import traceback
        state["phase"] = "exception"
        (attempt_dir / "result.json").write_text(json.dumps(
            {"status": "failed",
             "error": f"{type(exc).__name__}: {exc}",
             "traceback": traceback.format_exc()}, default=str))
    finally:
        stop_hb.set()
        log.flush()
""")


def _cancelled(cancel_file: str) -> bool:
    return Path(cancel_file).exists()


class AttemptRecord(dict):
    """Envelope required by dossier 03's reproduction matrix."""


def run_attempt(payload_path: str, params: dict, attempt_dir: str | Path, *,
                timeout_s: float = DEFAULT_TIMEOUT_S,
                grace_s: float = DEFAULT_GRACE_S,
                heartbeat_s: float = DEFAULT_HEARTBEAT_S,
                rlimit_as_bytes: int = DEFAULT_RLIMIT_AS_BYTES,
                rlimit_fsize_bytes: int = DEFAULT_RLIMIT_FSIZE_BYTES,
                child_env: dict | None = None,
                python_exe: str | None = None,
                sample_interval_s: float = 0.2) -> AttemptRecord:
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    params_path = attempt_dir / "params.json"
    params_path.write_text(json.dumps(params, default=str))

    def _limits():  # child-side safety nets
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (rlimit_as_bytes,) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (rlimit_fsize_bytes,) * 2)

    child_script = attempt_dir / "_child_incident.py"
    child_script.write_text(_CHILD_TEMPLATE)
    env = dict(os.environ)
    env.pop("NUMBA_DISABLE_JIT", None)
    env["INCIDENT_HEARTBEAT_S"] = str(heartbeat_s)
    env["PYTHONHASHSEED"] = "0"
    if child_env:
        env.update(child_env)

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [python_exe or sys.executable, str(child_script), str(REPO),
         str(attempt_dir), str(payload_path), str(params_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env, preexec_fn=_limits, start_new_session=True)
    sampler = ProcessTreeSampler(root_pid=proc.pid,
                                 interval_s=sample_interval_s,
                                 label=attempt_dir.name)
    sampler.start()
    # watchdog: independent of anything the child does, including C-level
    # hangs; acts only on the parent's wall clock
    watchdog = threading.Timer(timeout_s, _on_deadline,
                               args=(proc, attempt_dir, grace_s, timeout_s))
    watchdog.daemon = True
    watchdog.start()
    try:
        exitcode = proc.wait()
    finally:
        watchdog.cancel()
        mem = sampler.stop()
    wall_s = time.perf_counter() - t0

    result_path = attempt_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() \
        else {"status": "no_result_file"}
    if result["status"] != "completed" and proc.poll() is None:
        pass  # unreachable; proc.wait already returned
    timed_out = (attempt_dir / "watchdog_fired.json").exists()
    status = result.get("status", "no_result_file")
    if timed_out:
        status = "incomplete"

    rec = AttemptRecord(
        attempt=str(attempt_dir),
        payload=str(payload_path),
        python_exe=python_exe or sys.executable,
        status=status,  # completed | failed | incomplete | no_result_file
        timed_out=timed_out,
        exitcode=exitcode,
        wall_s=wall_s,
        timeout_s=timeout_s,
        grace_s=grace_s,
        envelope=_envelope(params, python_exe or sys.executable),
        memory=mem,
        child_result=result,
    )
    (attempt_dir / "attempt_record.json").write_text(
        json.dumps(rec, indent=2, default=str))
    return rec


def _on_deadline(proc: subprocess.Popen, attempt_dir: Path,
                 grace_s: float, timeout_s: float) -> None:
    """Request a stack dump, set the cooperative cancel flag, wait the
    bounded grace, then terminate ONLY the owned child; reap happens in
    the main flow via proc.wait()."""
    (attempt_dir / "watchdog_fired.json").write_text(json.dumps(
        {"deadline_s": timeout_s, "grace_s": grace_s,
         "t": time.time()}))
    try:  # stack dump where safe (child handler -> faulthandler)
        proc.send_signal(signal.SIGUSR1)
    except Exception:
        pass
    (attempt_dir / "cancel.flag").write_text("watchdog deadline\n")
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.terminate()  # SIGTERM: owned child only
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()          # last resort after the bounded grace
    proc.wait(timeout=grace_s)


def _envelope(params: dict, python_exe: str) -> dict:
    """Versions / OS / start method / dataset identity / parameters —
    the reproduction-matrix fields dossier 03 requires per attempt."""
    try:
        exe_ver = subprocess.run(
            [python_exe, "-c", "import sys;print(sys.version.split()[0])"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        exe_ver = ""
    return {
        "python": exe_ver or sys.version.split()[0],
        "python_exe": python_exe,
        "os": f"{platform.system()} {platform.release()}",
        "start_method": "spawn (subprocess, start_new_session=True)",
        "machine": platform.machine(),
        "dataset": params.get("dataset", {}),
        "parameters": params.get("parameters", {}),
        "child_env": params.get("child_env", {}),
    }


def assert_attempt_ok(rec: AttemptRecord, *, allow_failed: bool = False):
    """Dossier rule: a watchdog timeout is censored data, never a pass;
    failures must carry an executed traceback."""
    if rec["status"] == "incomplete":
        raise AssertionError(
            f"attempt timed out (censored): {rec['attempt']}")
    if rec["status"] == "failed" and not allow_failed:
        raise AssertionError(
            f"attempt failed: {rec['child_result'].get('error')}")
    if rec["status"] == "no_result_file":
        raise AssertionError(f"child died without result: {rec['attempt']}")
