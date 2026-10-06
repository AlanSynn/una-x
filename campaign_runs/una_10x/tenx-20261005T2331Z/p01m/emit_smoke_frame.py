"""P01M smoke frame: exercise the phase-telemetry emit path end to end.

Produces phase_events.jsonl + phase_summary.json from the PUBLIC parallel
RunBatch over the frozen 3x3 smoke grid on THIS local machine, plus a
default-off control run proving the disabled schema stays empty.

STATUS: SMOKE_FRAME_LOCAL — plumbing demonstration only.  NOT H1/H2/H3
evidence (different machine, tiny workload, no observed cohort), not
primary timing, not chargeable against the heavy budget.  The observed
H1/H2/H3 + sampler-ablation runs remain blocked_pending_cluster (PACE
maintenance 2026-10-05..08).
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

SMOKE = REPO / "tests" / "execution" / "fixtures"
OUT = Path(__file__).resolve().parent


def build(n=4):
    from urban_network_analysis import UNA
    p = UNA(verbosity=0)
    p.settings.data_folder = str(SMOKE)
    p.settings.network_file = "network.geojson"
    p.settings.origins_file = "origins.geojson"
    p.settings.destinations_file = "destinations.geojson"
    p.settings.output_folder = str(tempfile.mkdtemp(prefix="p01m-smoke-"))
    p.settings.output_wStamp = False
    for i in range(n):
        p.settings.name = f"row{i}"
        p.settings.search_radius = 90 + 10 * i
        p.SaveSettingsToProject()
    return p


def main():
    t0 = time.monotonic_ns()
    # --- control: telemetry OFF ------------------------------------------
    os.environ.pop("UNA_BATCH_PHASES", None)
    p0 = build()
    p0.RunBatch("accessibility", parallel=True, workers=2)
    rep0 = p0.batch_report
    control = {
        "telemetry_keys_in_extra": sorted(
            k for k in rep0.extra
            if k in ("coordinator_phases", "row_transport",
                     "phase_totals", "telemetry_dropped_events")),
        "rows_phases_all_default": all(r.phases == () for r in rep0.rows),
        "rows_dropped_all_zero": all(r.dropped_events == 0
                                     for r in rep0.rows),
    }
    assert control["telemetry_keys_in_extra"] == [], control
    assert control["rows_phases_all_default"] and \
        control["rows_dropped_all_zero"]

    # --- instrumented: telemetry ON --------------------------------------
    os.environ["UNA_BATCH_PHASES"] = "1"
    t_on = time.monotonic_ns()
    p = build()
    p.RunBatch("accessibility", parallel=True, workers=2)
    rep = p.batch_report
    wall_ns = time.monotonic_ns() - t_on

    events = []
    host_pid = os.getpid()
    for r in rep.rows:
        for s in r.phases:
            events.append({"arm": "smoke_local_on", "side": "worker",
                           "row_index": r.index, "worker_pid": r.worker_pid,
                           "status": r.status, **s})
    for s in rep.extra["coordinator_phases"]:
        events.append({"arm": "smoke_local_on", "side": "coordinator",
                       "row_index": None, "worker_pid": host_pid,
                       "status": "COMMITTED", **s})
    events.sort(key=lambda e: (e["start_mono_ns"], e["side"]))

    with open(OUT / "phase_events.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "schema": "una-10x-p01m-phase-events-1",
            "frame": "SMOKE_FRAME_LOCAL",
            "note": "plumbing demonstration; not H1/H2/H3 evidence; "
                    "not primary timing",
            "host": sys.platform, "coordinator_pid": host_pid,
            "workers_effective": rep.workers_effective,
            "rows": len(rep.rows)}) + "\n")
        for e in events:
            f.write(json.dumps(e) + "\n")

    # Summary: per (side, phase) count + wall totals.
    agg = {}
    for e in events:
        k = (e["side"], e["phase"])
        a = agg.setdefault(k, {"count": 0, "wall_ns": 0})
        a["count"] += 1
        a["wall_ns"] += e["end_mono_ns"] - e["start_mono_ns"]
    summary = {
        "schema": "una-10x-p01m-phase-summary-1",
        "frame": "SMOKE_FRAME_LOCAL",
        "monotonic_epoch_note": "monotonic ns, single host, single run; "
                                "spans from one recorder share one epoch, "
                                "worker and coordinator epochs are "
                                "independent (do not subtract across sides)",
        "control_off": control,
        "run_wall_ns_instrumented": wall_ns,
        "telemetry_dropped_events":
            rep.extra["telemetry_dropped_events"],
        "phase_totals": rep.extra.get("phase_totals", {}),
        "row_transport_rows": sorted(rep.extra["row_transport"]),
        "phases": [{"side": s, "phase": ph, **v}
                   for (s, ph), v in sorted(agg.items())],
        "provenance": {
            "generator": str(Path(__file__).resolve()),
            "workload": "frozen 3x3 smoke grid, 4 rows, workers=2",
            "heavy_budget_charged": False,
        },
    }
    with open(OUT / "phase_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    total_s = (time.monotonic_ns() - t0) / 1e9
    print(f"SMOKE OK: {len(events)} events, "
          f"{len(agg)} distinct (side, phase); control clean; "
          f"wall {total_s:.2f}s")


if __name__ == "__main__":
    main()
