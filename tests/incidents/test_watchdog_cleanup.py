"""Watchdog and failure-cleanup verification (dossier 03 diagnostic
protocol): an independent parent-side watchdog must bound a runaway
child that ignores the cooperative cancel flag — stack dump requested,
bounded grace, SIGTERM to the OWNED child only, reap, capsule preserved,
result marked incomplete, and no orphaned processes left behind.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from incident_runner import (DEFAULT_GRACE_S, assert_attempt_ok,
                             run_attempt)  # noqa: E402

PAYLOAD = str(HERE / "payloads" / "payload_runaway.py")

pytestmark = pytest.mark.incidents


def _descendant_pids(root_pid: int) -> set[int]:
    """All live pids in the tree rooted at root_pid (ppp closure over /proc)."""
    out = {root_pid}
    changed = True
    while changed:
        changed = False
        for pdir in Path("/proc").iterdir():
            if not pdir.name.isdigit():
                continue
            try:
                stat = (pdir / "stat").read_text()
                ppid = int(stat[stat.rindex(")") + 2:].split()[1])
            except (OSError, ValueError):
                continue
            if ppid in out and int(pdir.name) not in out:
                out.add(int(pdir.name))
                changed = True
    return out


def test_runaway_child_bounded_and_reaped(attempts_root):
    d = attempts_root / "watchdog_runaway"
    timeout_s = 6.0
    t0 = time.perf_counter()
    rec = run_attempt(PAYLOAD, {"run_for_s": 600}, d,
                      timeout_s=timeout_s, grace_s=DEFAULT_GRACE_S)
    wall = time.perf_counter() - t0

    # bounded: total wall ~= timeout + grace, far below the 600 s runaway
    assert wall < 60.0, f"watchdog did not bound the child: {wall:.1f}s"
    assert rec["timed_out"] is True
    assert rec["status"] == "incomplete", (
        "watchdog timeout must be recorded as censored 'incomplete', "
        f"got {rec['status']}")
    assert rec["exitcode"] != 0, "terminated child must not report success"
    assert rec["memory"]["samples_taken"] > 0

    # capsule preserved: watchdog note + heartbeat + attempt record
    assert (d / "watchdog_fired.json").is_file()
    assert (d / "attempt_record.json").is_file()
    assert (d / "heartbeat.jsonl").is_file()
    fired = json.loads((d / "watchdog_fired.json").read_text())
    assert fired["deadline_s"] == timeout_s

    # cleanup: no owned process (child or descendant) survives the reap
    time.sleep(0.3)
    per_sample = rec["memory"]["per_sample"]
    child_pid = per_sample[-1]["root_pid"] if per_sample else None
    owned = _descendant_pids(child_pid) if child_pid else set()
    live_owned = [p for p in owned if Path(f"/proc/{p}").exists()
                  and _cmdline_mentions(str(d), p)]
    assert not live_owned, f"orphaned owned workers: {live_owned}"

    # a timeout must NOT be usable as a runtime measurement
    with pytest.raises(AssertionError, match="censored"):
        assert_attempt_ok(rec)


def _cmdline_mentions(needle: str, pid: int) -> bool:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
            "utf-8", "replace")
        return needle in cmd
    except OSError:
        return False


def test_healthy_child_untouched_by_watchdog(attempts_root):
    """Control: a completing child is never signalled and reports
    completed with its sampler memory attached."""
    rec = run_attempt(PAYLOAD, {"run_for_s": 0.2}, attempts_root / "watchdog_ok",
                      timeout_s=30, grace_s=DEFAULT_GRACE_S)
    assert rec["timed_out"] is False
    assert rec["status"] == "completed"
    assert_attempt_ok(rec)
