"""BATCH_REVIEW probe 1: composite re-finalize after resume-finalize.

Adversarial question (cache/checkpoint interaction class): when a
checkpointed, composite-output batch COMPLETES (journal composite phase
finalized) and a later resume re-runs exactly one row whose committed
artifact was damaged on disk, what does the republished composite file
contain?

Code path under test (reviewed bytes e2d790b):
  runtime.run_batch_parallel -> the reuse branch skips re-folding
  committed rows' captures when composite_already_finalized is True
  (src/urban_network_analysis/batch/runtime.py:1038-1047); the re-run
  row folds its capture (folded_this_call=True); the trailing condition
  (runtime.py:1113-1115) then calls una._finalize_batch_composite(),
  which rebuilds composite_results from self._composite_captured ONLY
  (src/urban_network_analysis/UNA.py::_compute_batch_composite) and
  overwrites the composite files (UNA.py::_export_batch_composite).

Expected if correct: composite after resume carries ALL THREE row
columns.  Expected per code reading: only the re-run row's column.
"""
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "src"))

import pyarrow.feather as feather  # noqa: E402
from urban_network_analysis import UNA  # noqa: E402

SMOKE = REPO / "tests" / "execution" / "fixtures"
WORK = Path("/tmp/batch_review_probe1")


def make(n, out, ckpt):
    p = UNA(verbosity=0)
    p.settings.data_folder = str(SMOKE)
    p.settings.network_file = "network.geojson"
    p.settings.origins_file = "origins.geojson"
    p.settings.destinations_file = "destinations.geojson"
    p.settings.output_folder = str(out)
    p.settings.output_wStamp = False
    p.settings.batch_composite_output = True
    p.settings.batch_composite_result_column = "reach"
    for i in range(n):
        p.settings.name = f"row{i}"
        p.settings.search_radius = 90 + 10 * i
        p.SaveSettingsToProject()
    return p, replace(p.execution, checkpoint=str(ckpt))


def composite_files(out):
    return sorted(f for f in os.listdir(out) if f.startswith("composite"))


def composite_columns(out):
    """Column names of the composite output (feather preferred)."""
    files = composite_files(out)
    if not files:
        return None, "NO COMPOSITE FILE"
    feather_file = next((f for f in files if f.endswith(".feather")), None)
    if feather_file:
        path = out / feather_file
        return feather.read_table(path).column_names, str(path)
    path = out / files[0]
    import geopandas as gpd
    return list(gpd.read_file(path).columns), str(path)


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    out = WORK / "out"
    ckpt = WORK / "ckpt"

    p, opts = make(3, out, ckpt)
    p.RunBatch("accessibility", parallel=True, workers=2, execution=opts)
    cols1, path1 = composite_columns(out)
    print("run1 composite columns:", cols1)
    print("run1 generation:", p.batch_report.checkpoint_generation)
    print("run1 composite files:", composite_files(out))

    # Damage one committed row artifact (same fault class as
    # tests/batch/runtime/test_runtime_checkpoint.py::
    # test_tampered_artifact_forces_row_rerun).
    victims = sorted(f for f in os.listdir(out) if f.startswith("row0"))
    victim = out / victims[0]
    before = victim.read_bytes()
    with open(victim, "ab") as f:
        f.write(b"TAMPERED-BY-REVIEW-PROBE")
    print("tampered:", victim.name, len(before), "->",
          victim.stat().st_size, "bytes")

    p2, opts2 = make(3, out, ckpt)
    p2.RunBatch("accessibility", parallel=True, workers=2, execution=opts2)
    cols2, path2 = composite_columns(out)
    print("run2 composite columns:", cols2)
    rep = p2.batch_report
    print("run2 rows:", [(r.index, r.status) for r in rep.rows])
    print("run2 notes_runtime:")
    for note in rep.notes_runtime:
        print("   ", note)

    if cols1 == cols2:
        print("VERDICT: PASS (composite unchanged after resume)")
        return 0
    print(f"VERDICT: FAIL (composite shrank: {cols1} -> {cols2})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
