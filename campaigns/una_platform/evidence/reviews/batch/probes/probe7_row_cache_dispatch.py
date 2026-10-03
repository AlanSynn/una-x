import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

REPO = "/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/agent-a0b5549660ecd8e5b"
sys.path.insert(0, REPO + "/src")

from urban_network_analysis import UNA  # noqa: E402
from urban_network_analysis.Execution import CacheOptions  # noqa: E402
from urban_network_analysis.batch import runtime as rt  # noqa: E402

SMOKE = REPO + "/tests/execution/fixtures"
WORK = Path("/tmp/bpr7_instr")
STORE = WORK / "cache"
CKPT = WORK / "ckpt"


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    orig = rt._job_for

    def job_for(una, plan, i, staging_dir, analysis, sof, need_state):
        job = orig(una, plan, i, staging_dir, analysis, sof, need_state)
        ex = job.settings_state.get("execution")
        print(f"[dispatch row {i}] cache.mode="
              f"{getattr(getattr(ex, 'cache', None), 'mode', '?')}")
        return job

    rt._job_for = job_for
    try:
        p = UNA(verbosity=0)
        p.settings.data_folder = SMOKE
        p.settings.network_file = "network.geojson"
        p.settings.origins_file = "origins.geojson"
        p.settings.destinations_file = "destinations.geojson"
        p.settings.output_folder = str(WORK / "out")
        p.settings.output_wStamp = False
        for i in range(2):
            p.settings.name = f"row{i}"
            p.settings.search_radius = 95.0
            p.settings.execution = replace(
                p.execution,
                cache=CacheOptions(mode="disk", directory=str(STORE)))
            p.SaveSettingsToProject()
        p.RunBatch("accessibility", parallel=True, workers=2)
    finally:
        rt._job_for = orig

    n = sum(len(fs) for _, _, fs in os.walk(STORE)) if STORE.exists() else 0
    row0 = p.projects[0].execution.cache
    print("store files:", n)
    print("row0 post-run cache:", row0.mode, row0.directory)
    print("report notes_runtime:")
    for note in p.batch_report.notes_runtime:
        print("   ", note)


if __name__ == "__main__":
    main()
