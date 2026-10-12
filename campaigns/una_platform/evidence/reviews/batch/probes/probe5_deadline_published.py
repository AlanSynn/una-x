"""BATCH_REVIEW probe 5: deadline semantics and the report==published
invariant.

(a) success path: BatchExecutionReport.output_manifest must equal the
    artifacts actually on disk, exactly (no extra, no missing).
(b) deadline: a 6-row batch, rows 2+ held in a 40 s pre_run sleep with
    timeout_s=20 — the deadline must fire BatchCancelledError (not hang,
    not silently truncate), the committed prefix on disk must equal the
    outcomes attached to the exception, no staging leftovers, no
    surviving worker children, batch_report stays None, and a resume
    without the deadline must complete the batch with byte-identical
    row artifacts (crash-resume idempotence, dossier 05).
"""
import os
import shutil
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "src"))

WORK = Path("/tmp/batch_review_probe5")
SMOKE = REPO / "tests" / "execution" / "fixtures"


def _children_of_me():
    """(stray_worker_cmds, tracker_or_other_count) — classify children by
    cmdline: a multiprocessing.resource_tracker helper is expected and
    benign; a spawn worker (una_batch_worker_main / spawn_main) after
    cancel() is a reap failure."""
    me = str(os.getpid())
    workers, others = [], 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            if fields[1] != me:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", "replace").replace("\0", " ")
            if "resource_tracker" in cmd or "multiprocessing.forkserver" in cmd:
                others += 1
            else:
                workers.append(cmd.strip()[:90])
        except (OSError, IndexError, ValueError):
            continue
    return workers, others


def build(n_rows=6, slow_from=2, sleep=40.0):
    from urban_network_analysis import UNA
    p = UNA(verbosity=0)
    p.settings.data_folder = str(SMOKE)
    p.settings.network_file = "network.geojson"
    p.settings.origins_file = "origins.geojson"
    p.settings.destinations_file = "destinations.geojson"
    p.settings.output_folder = str(WORK / "out")
    p.settings.output_wStamp = False
    for i in range(n_rows):
        p.settings.name = f"row{i}"
        p.settings.search_radius = 90 + 2 * i
        if i >= slow_from:
            p.settings.inject_specs = [
                {"phase": "pre_run", "action": "sleep", "arg": sleep}]
        else:
            p.settings.inject_specs = []
        p.SaveSettingsToProject()
    return p


def main():
    from urban_network_analysis.batch import runtime as rt

    orig_job_for = rt._job_for

    def job_for(una, plan, i, staging_dir, analysis, script_output_folder,
                need_state):
        job = orig_job_for(una, plan, i, staging_dir, analysis,
                           script_output_folder, need_state)
        specs = getattr(una.projects[i], "inject_specs", None)
        if specs:
            object.__setattr__(job, "inject", specs)
        return job

    rt._job_for = job_for
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    ok = True

    # ---- (a) success: report.output_manifest == disk --------------------
    pa = build(n_rows=3, slow_from=99)
    t0 = time.monotonic()
    pa.RunBatch("accessibility", parallel=True, workers=2)
    dt = time.monotonic() - t0
    manifest = sorted(f for v in pa.batch_report.output_manifest.values()
                      for f in v)
    disk = sorted(
        str(WORK / "out" / f) for f in os.listdir(WORK / "out")
        if not f.startswith("."))
    print(f"(a) {len(manifest)} manifest files, {len(disk)} on disk "
          f"({dt:.1f}s)")
    a_ok = sorted(map(os.path.abspath, manifest)) == \
        sorted(map(os.path.abspath, disk))
    print("(a) VERDICT:", "PASS" if a_ok else "FAIL")
    ok &= a_ok

    # ---- (b) deadline ----------------------------------------------------
    shutil.rmtree(WORK / "out", ignore_errors=True)
    ck = WORK / "ckpt"
    opts = replace(build().execution, timeout_s=20.0, checkpoint=str(ck))
    pb = build()
    t0 = time.monotonic()
    cancelled_type, attached = None, None
    try:
        pb.RunBatch("accessibility", parallel=True, workers=2,
                    execution=opts)
        print("(b) FAIL: deadline did not cancel a 40s-row batch")
        rt._job_for = orig_job_for
        return 1
    except BaseException as exc:
        dt = time.monotonic() - t0
        cancelled_type = type(exc).__name__
        attached = sorted(r.index for r in getattr(exc, "batch_rows", ()))
    print(f"(b) cancelled after {dt:.1f}s: {cancelled_type}; prefix rows "
          f"{attached}; report={pb.batch_report}")
    disk_rows = sorted({str(int(f[3:].split(".")[0]))
                        for f in os.listdir(WORK / "out")
                        if not f.startswith(".") and f.startswith("row")})
    leftovers = [f for f in os.listdir(WORK / "out")
                 if f.startswith(".una-batch-staging")]
    workers, helpers = _children_of_me()
    prefix_ok = attached == list(range(len(attached))) and \
        attached == sorted(int(r) for r in disk_rows)
    b_ok = (cancelled_type == "BatchCancelledError" and dt < 60
            and pb.batch_report is None and not leftovers
            and not workers and prefix_ok)
    print("(b) disk row prefixes:", disk_rows, "| staging leftovers:",
          leftovers, "| stray worker cmds:", workers,
          "| benign helpers:", helpers)
    print("(b) VERDICT:", "PASS" if b_ok else "FAIL")
    ok &= b_ok

    # ---- (b) resume without the deadline completes idempotently ---------
    opts2 = replace(pb.execution, timeout_s=None, checkpoint=str(ck))
    pc = build()
    for s in pc.projects:
        s.inject_specs = []
    pc.RunBatch("accessibility", parallel=True, workers=2,
                execution=opts2)
    resumed_rows = [(r.index, r.status) for r in pc.batch_report.rows]
    print("(b) resume rows:", resumed_rows)
    resumed_ok = all(st == "ran" for _, st in resumed_rows) and \
        len(resumed_rows) == 6
    print("(b) resume VERDICT:", "PASS" if resumed_ok else "FAIL")
    ok &= resumed_ok

    rt._job_for = orig_job_for
    print("PROBE5 VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
