"""N00 C-arm leg driver (una_10x, designs/07 matched-window law).

Orchestrates the AB/AB interleaved leg matrix ruled by the team-lead:

  L2A1 numba -> L1B1 native -> L2A2 numba -> L1B2 native

Both arms run on the CURRENT tree under the SAME interpreter (venv-n00)
with PYTHONPATH-resolved repo src; the ONLY treatment delta is the
opt-in install() hook.  FRESH numba control legs (p05s receipts are
historical context, never the B arm).  Every timed wall is read back
from the per-leg JSON produced by leg_runner.py (never transcribed).

Fail-closed driver asserts per leg: exit 0, tree_head == HEAD, zero
failed rows, required census (C30 law: dot-prefixed cohort-store
residue excluded) == 64 files / 147829892 B, cache off, w4/t1/q64,
engagement law per arm.  Cross-leg: the timestamp-normalized digest
census must be IDENTICAL across all four legs (native vs numba
included) - any mismatch voids the leg set (bitwise determinism).

Pre-leg loadavg gate (team-lead ruling): 1-min loadavg must be < 8 at
leg start; otherwise the driver waits (bounded) instead of burning the
leg on contaminated walls.  The pre-leg loadavg is sampled while NO leg
process runs and is recorded in the leg console line.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WS = Path("/Users/alansynn/orca/workspaces/una-x")
MAIN = WS / "main"
RUNROOT = MAIN / "campaign_runs/una_10x/tenx-20261005T2331Z"
HERE = RUNROOT / "n00"
RUNNER = HERE / "instruments" / "leg_runner.py"
LEGS = HERE / "legs"
SCRATCH = Path("/tmp/n00_legs")
VENV = WS / "venv-n00"

PAIRING = MAIN / "campaign_data/inputs/upstream_Boston/Boston/" \
               "local_audit_manifest.csv"
DATA_FOLDER = MAIN / "campaign_data/inputs/upstream_Boston/Boston"

# Pinned control-cohort census (P01F registration; C30 residue law).
EXPECT_FILES = 64
EXPECT_BYTES = 147829892

LOADAVG_GATE = 8.0
LOADAVG_WAIT_S = 60
LOADAVG_MAX_WAIT_S = 90 * 60

# Fail-closed leg cap: a healthy leg is ~20-40 s (p05s walls ~18.6 s).
# The first shakedown wedged for 13+ min inside the leg process (opaque
# write-block, 2026-10-07) and silently blocked the whole pipeline -
# the driver now kills the leg's whole process group at the cap and
# relays the leg's faulthandler stack sidecar (leg_runner.py) so a
# wedge lands as a NAMED artifact, never an indefinite hang.
LEG_TIMEOUT_S = 45 * 60
SHAKEDOWN_TIMEOUT_S = 30 * 60

# (tag, arm) - AB/AB interleaved fresh-control matrix.
LEG_MATRIX = [
    ("L2A1_numba", "numba"),
    ("L1B1_native", "native"),
    ("L2A2_numba", "numba"),
    ("L1B2_native", "native"),
]

TS = re.compile(r"accessibility_\d{4}-\d{2}-\d{2}_\d{4}")


def _census(leg):
    """Timestamp-normalized digest census, hidden residue excluded."""
    return {TS.sub("accessibility_TS", f["rel"]): f["sha256"]
            for f in leg["output_files"]
            if not f["rel"].split("/")[0].startswith(".")}


def _loadavg():
    return os.getloadavg()[0]


def _wait_quiet(tag):
    """Block until the pre-leg gate passes (bounded); record the wait."""
    start = time.time()
    waits = []
    while _loadavg() >= LOADAVG_GATE:
        if time.time() - start > LOADAVG_MAX_WAIT_S:
            raise SystemExit(
                "LOADAVG GATE TIMEOUT after %.0f min (last %.2f) - legs "
                "NOT run, budget unburned" % ((time.time() - start) / 60,
                                              _loadavg()))
        waits.append(_loadavg())
        time.sleep(LOADAVG_WAIT_S)
    print("GATE %s loadavg1m=%.2f (waited %d samples, %.1f min)"
          % (tag, _loadavg(), len(waits), (time.time() - start) / 60))
    sys.stdout.flush()
    return waits


def _run_leg(tag, arm, gate=True, timeout_s=LEG_TIMEOUT_S):
    pre_loadavg = _loadavg()
    waits = _wait_quiet(tag) if gate else []
    numba_dir = SCRATCH / "numba" / tag
    outdir = SCRATCH / "out" / tag
    for d in (numba_dir, outdir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    leg_json = LEGS / f"{tag}.json"
    if leg_json.exists():
        leg_json.unlink()
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(numba_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("UNA_BATCH_PHASES", None)   # diagnostics off in timed legs
    env["PYTHONPATH"] = "%s/src:%s/native" % (MAIN, MAIN)
    # Spawn-worker native opt-in (native/sitecustomize.py): "1" installs
    # the cohort hook in each worker (native legs); "0" loads the module
    # WITHOUT installing - matched module-load wiring for the numba
    # control, hook never installed, engagement untouched.
    env["UNA_NATIVE_TRACE"] = "1" if arm == "native" else "0"
    cmd = [str(VENV / "bin" / "python"), str(RUNNER),
           "--tree", str(MAIN),
           "--pairing", str(PAIRING),
           "--data-folder", str(DATA_FOLDER),
           "--outdir", str(outdir),
           "--tag", tag,
           "--arm", arm,
           "--out", str(leg_json),
           "--parallel", "--workers", "4",
           "--threads", "1",
           "--queue-depth", "64"]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    wedge = False
    try:
        proc_stdout, proc_stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        wedge = True
        os.killpg(os.getpgid(proc.pid), 9)
        proc_stdout, proc_stderr = proc.communicate()
    host_wall = time.perf_counter() - t0
    if wedge:
        stacks = LEGS / f"{tag}.json.stacks.txt"
        print(f"LEG WEDGE {tag}: leg process group killed at "
              f"{timeout_s:.0f}s (rc/applied SIGKILL)")
        if stacks.exists():
            body = stacks.read_text()
            print(f"stacks sidecar ({len(body)} bytes, "
                  f"last 4000):\n{body[-4000:]}")
        else:
            print("  (no stacks sidecar - faulthandler never fired?)")
        print(f"leg stderr tail:\n{proc_stderr[-3000:]}")
        sys.exit(21)
    if proc.returncode != 0:
        print(f"LEG FAIL {tag} rc={proc.returncode}\n"
              f"stdout:\n{proc_stdout[-3000:]}\n"
              f"stderr:\n{proc_stderr[-3000:]}")
        sys.exit(20)
    leg = json.loads(leg_json.read_text())
    # Per-worker engagement evidence (native/sitecustomize.py atexit
    # reports): parse the leg's captured stderr lines into the leg JSON
    # so the dispatch counters are recorded live per leg.
    worker_engagement = []
    for line in (proc_stderr or "").splitlines():
        if line.startswith("[n00-native-engagement] "):
            worker_engagement.append(
                json.loads(line.split("] ", 1)[1]))
    leg["engagement"]["workers"] = worker_engagement
    leg["config"]["una_native_trace_env"] = env["UNA_NATIVE_TRACE"]
    required = [f for f in leg["output_files"]
                if not f["rel"].split("/")[0].startswith(".")]
    hidden = [f for f in leg["output_files"]
              if f["rel"].split("/")[0].startswith(".")]
    leg["required_census"] = {"files": len(required),
                              "bytes": sum(f["bytes"] for f in required)}
    leg["hidden_residue"] = {"files": len(hidden),
                             "bytes": sum(f["bytes"] for f in hidden),
                             "law": "cohort store root - the one allowed "
                                    "hidden residue (C30); excluded from "
                                    "the required census"}
    leg["driver_gate"] = {
        "pre_leg_loadavg_1m": round(pre_loadavg, 2),
        "gate_threshold": LOADAVG_GATE if gate else None,
        "gate_enforced": gate,
        "wait_samples": len(waits),
        "waited_s": len(waits) * LOADAVG_WAIT_S,
    }
    leg_json.write_text(json.dumps(leg, indent=1) + "\n")

    assert leg["tag"] == tag, tag
    assert leg["zero_failed_jobs"], tag
    assert leg["wedge_watchdog"]["clean"], \
        (tag, "wedge dumps fired during a completing leg")
    assert leg["required_census"]["files"] == EXPECT_FILES, \
        (tag, leg["required_census"]["files"])
    assert leg["required_census"]["bytes"] == EXPECT_BYTES, \
        (tag, leg["required_census"]["bytes"])
    assert leg["config"]["cache_mode"] == "off", tag
    assert leg["config"]["workers"] == 4, tag
    assert leg["config"]["threads_per_worker"] == 1, tag
    assert leg["config"]["queue_depth"] == 64, tag
    if arm == "native":
        eng = leg["engagement"]
        worker_dispatches = sum(w.get("dispatches", 0)
                                for w in eng.get("workers", []))
        assert worker_dispatches > 0, \
            (tag, "no worker reported native dispatches - "
                  "sitecustomize opt-in wiring dead")
    else:
        assert not leg["engagement"].get("workers"), \
            (tag, "numba leg must carry no worker engagement reports")
    print(f"LEG OK {tag:14s} arm={arm:6s} "
          f"wall={leg['timing']['public_call_wall_s']:8.4f}s "
          f"(host {host_wall:6.1f}s) "
          f"rss_tree={leg['sampler']['peak_tree_rss_bytes']/2**20:7.1f}MiB "
          f"preLA={pre_loadavg:.2f} "
          f"maxLA={leg['sampler']['loadavg_1m_max']} "
          f"residue={leg['hidden_residue']['files']}f/"
          f"{leg['hidden_residue']['bytes']}B "
          f"wdisp={sum(w.get('dispatches', 0) for w in leg['engagement'].get('workers', []))}")
    sys.stdout.flush()
    return leg


def shakedown(arm):
    # Shakedown walls are never claimed; the quiet-window gate is for
    # the measured matrix only.
    tag = f"sd_n00_{arm}"
    leg = _run_leg(tag, arm, gate=False, timeout_s=SHAKEDOWN_TIMEOUT_S)
    return leg


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shakedowns-only", action="store_true")
    ap.add_argument("--legs-only", action="store_true",
                    help="skip shakedowns; run the AB/AB matrix")
    args = ap.parse_args()

    LEGS.mkdir(parents=True, exist_ok=True)

    if not args.legs_only:
        print("== shakedowns (labelled; consumed, never claimed) ==")
        shakedown("numba")
        shakedown("native")
        if args.shakedowns_only:
            return

    print("== measured AB/AB matrix (fresh controls) ==")
    legs = {}
    for tag, arm in LEG_MATRIX:
        legs[tag] = _run_leg(tag, arm)

    print("== bitwise cross-leg determinism (native vs numba included) ==")
    censuses = {tag: _census(leg) for tag, leg in legs.items()}
    ref_tag, ref = next(iter(censuses.items()))
    for tag, c in censuses.items():
        assert c == ref, \
            f"DETERMINISM VOID: {tag} != {ref_tag} " \
            f"({len(set(c.items()) ^ set(ref.items()))} differing files)"
    print(f"DETERMINISM OK: {len(censuses)} legs, {len(ref)} files, "
          f"digest-identical (timestamp-normalized, residue-excluded)")

    summary = {
        "schema": "una-10x-n00-leg-matrix-1",
        "matrix": LEG_MATRIX,
        "walls_s": {t: legs[t]["timing"]["public_call_wall_s"]
                    for t in legs},
        "pre_leg_loadavg": {t: legs[t]["driver_gate"]["pre_leg_loadavg_1m"]
                            for t in legs},
        "in_run_loadavg_max": {t: legs[t]["sampler"]["loadavg_1m_max"]
                               for t in legs},
        "peak_tree_rss_bytes": {t: legs[t]["sampler"]["peak_tree_rss_bytes"]
                                for t in legs},
        "engagement_native": {t: legs[t]["engagement"] for t in legs
                              if legs[t]["arm"] == "native"},
        "determinism": {"files": len(ref),
                        "verdict": "identical across all legs"},
        "required_census": {"files": EXPECT_FILES,
                            "bytes": EXPECT_BYTES},
    }
    (LEGS / "matrix_summary.json").write_text(json.dumps(summary,
                                                         indent=1) + "\n")
    print("MATRIX SUMMARY -> legs/matrix_summary.json")


if __name__ == "__main__":
    main()
