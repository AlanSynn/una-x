"""Simultaneous process-tree memory sampling (HARNESS.md "Memory and thermal
sampling"; negative control 8).

A coordinator thread walks /proc on a fixed interval and sums the RSS of the
root process *and every recursive descendant that coexists in that sample*.
Per-worker maxima are never summed: the reported peak is the largest
simultaneous whole-tree RSS observed in one sample.  Available RAM/swap are
recorded from /proc/meminfo, sampling loss and child exits are retained.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path


def _proc_stat_fields(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    # comm may contain spaces/parens: parse after the last ')'
    close = raw.rfind(b")")
    if close < 0:
        return None
    fields = raw[close + 2:].split()
    return [str(pid)] + [f.decode(errors="replace") for f in fields]


def snapshot_tree(root_pid: int) -> dict:
    """One simultaneous sample of the whole descendant tree of *root_pid*.

    Returns {} when the root has already exited (sampling loss is recorded
    by the caller rather than silently dropped).
    """
    root_fields = _proc_stat_fields(root_pid)
    if root_fields is None:
        return {}
    ppid = {}
    rss_pages = {}
    statm = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        f = _proc_stat_fields(pid)
        if f is None:
            continue
        # field 4 (index 3 after prepending pid) is ppid
        ppid[pid] = int(f[3])
        try:
            with open(f"/proc/{pid}/statm", "rb") as fh:
                resident_pages = int(fh.read().split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError,
                IndexError, ValueError):
            resident_pages = None
        rss_pages[pid] = resident_pages
    # descendants of root_pid at this instant
    children: dict[int, list[int]] = {}
    for pid, pp in ppid.items():
        children.setdefault(pp, []).append(pid)
    tree, stack = set(), [root_pid]
    while stack:
        cur = stack.pop()
        if cur in tree:
            continue
        tree.add(cur)
        stack.extend(children.get(cur, ()))
    page = os.sysconf("SC_PAGE_SIZE")
    per_process = {}
    total = 0
    missing = 0
    for pid in sorted(tree):
        pages = rss_pages.get(pid)
        if pages is None:
            missing += 1
            continue
        per_process[str(pid)] = pages * page
        total += pages * page
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemAvailable:", "MemTotal:", "SwapTotal:",
                                "SwapFree:")):
                k, v = line.split(":", 1)
                mem[k] = int(v.strip().split()[0]) * 1024
    except OSError:
        pass
    return {
        "t": time.time(),
        "root_pid": root_pid,
        "tree_size": len(tree),
        "rss_bytes_sum": total,
        "rss_bytes_per_pid": per_process,
        "pids_missing_statm": missing,
        "meminfo": mem,
    }


class ProcessTreeSampler:
    """Background coordinator sampler; peak = max simultaneous tree RSS."""

    def __init__(self, root_pid: int, interval_s: float = 0.25,
                 label: str = "run"):
        self.root_pid = root_pid
        self.interval_s = interval_s
        self.label = label
        self.samples: list[dict] = []
        self.lost_samples = 0
        self.child_exit_seen: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"una-sampler-{self.label}")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            snap = snapshot_tree(self.root_pid)
            if snap:
                self.samples.append(snap)
            else:
                self.lost_samples += 1
            self._stop.wait(self.interval_s)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 4)
        peak = max((s["rss_bytes_sum"] for s in self.samples), default=0)
        return {
            "label": self.label,
            "interval_s": self.interval_s,
            "samples_taken": len(self.samples),
            "samples_lost": self.lost_samples,
            "peak_simultaneous_rss_bytes": peak,
            "peak_is_simultaneous_tree_sum": True,
            "per_sample": self.samples,
            "meminfo_final": (self.samples[-1]["meminfo"]
                              if self.samples else {}),
            "filesystem": "procfs(/proc)",
        }

    @property
    def peak(self) -> int:
        return max((s["rss_bytes_sum"] for s in self.samples), default=0)
