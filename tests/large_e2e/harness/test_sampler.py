"""Process-tree sampler tests (HARNESS.md behavior 7).

The sampler must take ONE simultaneous snapshot of the coordinator plus ALL
descendants (children and grandchildren), label the summed RSS as
conservative, record its own overhead, hold its declared interval, and trip
its watchdogs.  Real child/grandchild processes are used so the tree walk is
proven against actual descendants, not mocks.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from harness import sampler as sampler_module
from harness.sampler import ProcessTreeSampler

def _touch_pages(n: int) -> str:
    """A -c script that makes n bytes RESIDENT (calloc maps lazily on macOS)."""
    return (
        "n = %d\n"
        "b = bytearray(n)\n"
        "b[::4096] = b'\\x01' * ((n + 4095) // 4096)\n"
        % n)


GRANDCHILD_SCRIPT = (
    _touch_pages(48 * 1024 * 1024)
    + "import time; time.sleep(6)\n")

ALLOC_CHILD_SCRIPT = (
    "import subprocess, sys, time\n"
    "grandchild = subprocess.Popen(\n"
    "    [sys.executable, '-c', %r])\n"
    "time.sleep(0.5)              # grandchild reaches its allocation\n"
    "n = %d\n"
    "b = bytearray(n)\n"
    "b[::4096] = b'\\x01' * ((n + 4095) // 4096)\n"
    "time.sleep(4)                # window for the sampler to see both\n"
    "grandchild.kill()\n"
    "grandchild.wait()\n"
    % (GRANDCHILD_SCRIPT, 96 * 1024 * 1024))

MIB = 1024 * 1024


def test_summary_labels_conservative_sampled_semantics():
    sampler = ProcessTreeSampler(interval_s=0.05)
    sampler.start()
    time.sleep(0.35)
    summary = sampler.stop()
    # Declared metric matches the implementation: conservative summed RSS.
    assert "double-count" in summary["metric_semantics"]
    assert "summed resident set" in summary["metric_semantics"]
    # Sampled peaks are never advertised as caps.
    assert summary["guarantee"].startswith("sampled peaks only")
    assert summary["trip"] is None
    assert summary["samples_recorded"] >= 3
    assert summary["sampler_overhead_seconds"] > 0
    assert summary["avg_walk_seconds"] > 0
    assert summary["peak"]["n_processes"] >= 1


def test_interval_is_held_between_ticks():
    interval = 0.1
    sampler = ProcessTreeSampler(interval_s=interval)
    sampler.start()
    time.sleep(1.0)
    summary = sampler.stop()
    samples = summary["samples"]
    assert len(samples) >= 5
    gaps = [b["t"] - a["t"] for a, b in zip(samples, samples[1:])]
    # Held from tick START (walk time included): the median gap must sit
    # close to the declared interval, never far below it.
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    assert interval * 0.8 <= median_gap <= interval * 3.0, median_gap


def test_child_and_grandchild_in_same_simultaneous_snapshot(tmp_path):
    """A resident child AND its resident grandchild appear together in one
    per-sample per_pid list, and the summed RSS peak reflects them."""
    child = subprocess.Popen(
        [sys.executable, "-c", ALLOC_CHILD_SCRIPT])
    try:
        sampler = ProcessTreeSampler(interval_s=0.1)
        sampler.start()
        child.wait(timeout=30)
        time.sleep(0.3)      # allow a final tick to land
        summary = sampler.stop()
    finally:
        child.kill()
        child.wait()

    assert summary["peak"]["tree_rss_bytes_sum"] > 96 * MIB
    pids_seen = {entry["pid"] for s in summary["samples"]
                 for entry in s["per_pid"]}
    assert child.pid in pids_seen
    # SIMULTANEITY: some single snapshot contains child AND grandchild.
    together = any(child.pid in {e["pid"] for e in s["per_pid"]} and
                   len(s["per_pid"]) >= 3
                   for s in summary["samples"])
    assert together, "no single snapshot covered child + grandchild"
    # Per-sample process counts include self + child + grandchild.
    assert summary["peak"]["n_processes"] >= 3


def test_memory_budget_watchdog_trips_and_stops_sampler():
    sampler = ProcessTreeSampler(interval_s=0.05, memory_budget_bytes=MIB)
    sampler.start()
    deadline = time.monotonic() + 10
    while not sampler.trip_event.is_set() and time.monotonic() < deadline:
        time.sleep(0.05)
    summary = sampler.stop()
    assert sampler.trip_event.is_set()
    assert summary["trip"] is not None
    assert summary["trip"]["reason"] == "memory_budget_exceeded"
    assert summary["trip"]["detail"]["budget_bytes"] == MIB
    assert "double-count" in summary["trip"]["detail"]["note"]
    # The sampler thread actually stopped after tripping.
    time.sleep(0.2)
    samples_before = summary["samples_total"]
    assert not sampler._thread.is_alive()
    assert summary["samples_total"] == samples_before


def test_external_pressure_stop_trips_on_tiny_available(monkeypatch):
    """The pressure watchdog fires when system available RAM falls below
    max(1 GiB, 0.10 * physical) — simulated with a fake vm metric."""
    class FakeVM:
        total = 8 * 1024 * 1024 * 1024
        available = 64 * MIB   # far below max(1 GiB, 0.10 * 8 GiB)

    sampler = ProcessTreeSampler(interval_s=0.05)
    monkeypatch.setattr(sampler, "_vm", lambda: FakeVM())
    sampler.start()
    deadline = time.monotonic() + 10
    while not sampler.trip_event.is_set() and time.monotonic() < deadline:
        time.sleep(0.05)
    summary = sampler.stop()
    assert summary["trip"] is not None
    assert summary["trip"]["reason"] == "external_pressure_stop"
    assert summary["trip"]["detail"]["available_bytes"] == 64 * MIB


def test_decimation_keeps_stored_samples_bounded(monkeypatch):
    """Long runs must not accumulate unbounded samples: past the bound the
    sampler decimates deterministically and reports it."""
    monkeypatch.setattr(sampler_module, "MAX_STORED_SAMPLES", 8)
    sampler = ProcessTreeSampler(interval_s=0.01)
    sampler.start()
    time.sleep(0.5)
    summary = sampler.stop()
    assert summary["samples_total"] > summary["samples_recorded"]
    assert summary["decimated"] is True
    assert summary["samples_recorded"] <= 16  # bound + bounded slop
