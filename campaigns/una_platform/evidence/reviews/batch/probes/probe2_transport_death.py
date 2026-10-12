"""BATCH_REVIEW probe 2: synchronous result transport under abrupt death.

(a) unit: _ResultChannel (SimpleQueue) with 48 MiB payloads; producer A
    SIGKILLs itself right after its second put() returns; every put()
    that returned before the kill must arrive intact (this is the
    mp.Queue feeder-buffer truncation BATCH_EXEC's fix round removed).
(b) integration: 5-row public RunBatch, workers=2, row4 dies at its
    post_run seam via the documented inject_specs seam (same
    monkeypatch of runtime._job_for as tests/batch/runtime/conftest.py).
    Required: contiguous committed prefix from row 0 (exact extent is
    timing-dependent — the docstring promises "every RECEIVABLE
    result"), row 4 never committed, loud BatchCancelledError, no hang,
    batch_report stays None, staging removed, no surviving worker child.
"""
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "src"))

WORK = Path("/tmp/batch_review_probe2")
BIG = 48 * (1 << 20)


def _children_of_me():
    """(stray_worker_cmds, benign_helper_count): resource_tracker and
    forkserver helpers are expected multiprocessing plumbing; a spawn
    worker after cancel() would be a reap failure."""
    me = str(os.getpid())
    workers, helpers = [], 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            if fields[1] != me:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", "replace").replace("\0", " ")
            if "resource_tracker" in cmd or "forkserver" in cmd:
                helpers += 1
            else:
                workers.append(cmd.strip()[:90])
        except (OSError, IndexError, ValueError):
            continue
    return workers, helpers


def _producer(ch, tag, payloads, die_after):
    for i, blob in enumerate(payloads):
        ch.put((tag, i, blob))
        if die_after is not None and i == die_after:
            os.kill(os.getpid(), signal.SIGKILL)


def unit_channel():
    from urban_network_analysis.batch.runtime import _ResultChannel

    ctx = mp.get_context("spawn")
    ch = _ResultChannel(ctx)

    payloads_a = [bytes([0xA0 + k]) * BIG for k in range(3)]
    payloads_b = [bytes([0xB0 + k]) * BIG for k in range(3)]
    pa = ctx.Process(target=_producer, args=(ch, "A", payloads_a, 1))
    pb = ctx.Process(target=_producer, args=(ch, "B", payloads_b, None))
    pa.start()
    pb.start()
    got = []
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline and len(got) < 5:
        try:
            got.append(ch.get(timeout=5))
        except Exception:
            break
    pb.join(10)
    if pb.is_alive():
        pb.terminate()
        pb.join(5)
    pa.join(5)
    ok = True
    for tag, i, blob in got:
        if len(blob) != BIG or blob != bytes([blob[0]]) * BIG:
            print(f"  CORRUPT result ({tag},{i}) len={len(blob)}")
            ok = False
    seq = [(t, i) for t, i, _ in got]
    print("(a) received:", seq)
    if not {("A", 0), ("A", 1)}.issubset(set(seq)):
        print("  FAIL: a put() that returned before SIGKILL was lost")
        ok = False
    print("(a) VERDICT:", "PASS" if ok else "FAIL")
    return ok


def integration():
    from dataclasses import replace

    from urban_network_analysis import UNA
    from urban_network_analysis.batch import runtime as rt

    # conftest-equivalent: honour Settings.inject_specs on the job
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
    out = WORK / "out"
    SMOKE = REPO / "tests" / "execution" / "fixtures"

    p = UNA(verbosity=0)
    p.settings.data_folder = str(SMOKE)
    p.settings.network_file = "network.geojson"
    p.settings.origins_file = "origins.geojson"
    p.settings.destinations_file = "destinations.geojson"
    p.settings.output_folder = str(out)
    p.settings.output_wStamp = False
    die = {"phase": "post_run", "action": "exit", "arg": 9}
    for i in range(5):
        p.settings.name = f"row{i}"
        p.settings.search_radius = 90 + 5 * i
        if i == 4:
            p.settings.inject_specs = [die]
        p.SaveSettingsToProject()
    opts = replace(p.execution)

    t0 = time.monotonic()
    try:
        p.RunBatch("accessibility", parallel=True, workers=2,
                   execution=opts)
        print("(b) FAIL: batch completed although row4's worker died")
        rt._job_for = orig_job_for
        return False
    except BaseException as exc:
        dt = time.monotonic() - t0
        rows = getattr(exc, "batch_rows", ())
        committed = sorted(r.index for r in rows)
        print(f"(b) raised in {dt:.1f}s:", type(exc).__name__,
              "-", str(exc)[:90])
        print("(b) attached committed rows:",
              [(r.index, r.status) for r in rows])
    rt._job_for = orig_job_for

    leftovers = os.listdir(out) if out.is_dir() else []
    report_ok = p.batch_report is None
    workers, helpers = _children_of_me()
    print("(b) files on disk:", sorted(leftovers))
    print("(b) report is None:", report_ok,
          "| staging leftovers:",
          [f for f in leftovers if f.startswith(".")],
          "| stray worker cmds:", workers,
          "| benign helpers:", helpers)
    contiguous = committed == list(range(len(committed))) and \
        4 not in committed and len(committed) >= 1
    disk_ok = all(f.startswith("row") for f in leftovers)
    ok = contiguous and report_ok and not workers and disk_ok
    print("(b) VERDICT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ok_a = unit_channel()
    ok_b = integration()
    sys.exit(0 if (ok_a and ok_b) else 1)
