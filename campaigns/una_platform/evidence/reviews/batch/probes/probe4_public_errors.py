"""BATCH_REVIEW probe 4: public error semantics.

(a) serial-exact in-row failure: a mid-batch row whose input file is
    missing must raise the SAME exception type/message the serial route
    raises, AFTER the earlier rows have committed and published; the
    exception must carry the committed prefix (exc.batch_rows);
    una.batch_report stays None; no staging leftovers.
(b) skipped-row parity: a row with a missing required field is reported
    skipped with its bind transition applied (output_folder defaulted),
    exactly as the serial loop would leave it.
"""
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "src"))

WORK = Path("/tmp/batch_review_probe4")
SMOKE = REPO / "tests" / "execution" / "fixtures"


def base_una():
    from urban_network_analysis import UNA
    p = UNA(verbosity=0)
    p.settings.data_folder = str(SMOKE)
    p.settings.network_file = "network.geojson"
    p.settings.origins_file = "origins.geojson"
    p.settings.destinations_file = "destinations.geojson"
    p.settings.output_folder = str(WORK / "out")
    p.settings.output_wStamp = False
    return p


def add_rows(p, bad_at=None, skip_at=None):
    for i in range(3):
        p.settings.name = f"row{i}"
        p.settings.search_radius = 90 + 5 * i
        p.settings.origins_file = (
            "void.geojson" if i == bad_at else "origins.geojson")
        if i == skip_at:
            p.settings.origins_file = ""
        p.SaveSettingsToProject()
        p.settings.origins_file = "origins.geojson"


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    ok = True

    # ---- (a) serial exception reference ---------------------------------
    ps = base_una()
    add_rows(ps, bad_at=1)
    serial_type, serial_msg = None, None
    try:
        ps.RunBatch("accessibility")
    except BaseException as exc:
        serial_type, serial_msg = type(exc).__name__, str(exc)
    serial_report = ps.batch_report
    print("(a) serial raised:", serial_type, "-", serial_msg[:80])
    print("(a) serial report:", serial_report)

    # ---- (a) parallel route with the same broken row --------------------
    shutil.rmtree(WORK / "out", ignore_errors=True)
    pp = base_una()
    add_rows(pp, bad_at=1)
    from dataclasses import replace
    pp_type, pp_msg, pp_rows = None, None, None
    try:
        pp.RunBatch("accessibility", parallel=True, workers=2)
    except BaseException as exc:
        pp_type, pp_msg = type(exc).__name__, str(exc)
        pp_rows = sorted(getattr(exc, "batch_rows", ()) and
                         [r.index for r in getattr(exc, "batch_rows", [])])
    print("(a) parallel raised:", pp_type, "-", pp_msg[:80])
    print("(a) committed prefix on the exception:",
          pp_rows, "| report:", pp.batch_report)
    disk = sorted(os.listdir(WORK / "out")) if (WORK / "out").exists() else []
    print("(a) published files:", disk)
    a_ok = (pp_type == serial_type and pp_msg == serial_msg
            and pp.batch_report is None
            and pp_rows == [0]
            and all(f.startswith("row0") for f in disk)
            and not any(f.startswith(".") for f in disk))
    # serial parity of the prefix state: row0's settings post-state matches
    a_ok &= (ps.projects[0].output_folder == pp.projects[0].output_folder
             and ps.projects[0].output_file_name ==
             pp.projects[0].output_file_name)
    print("(a) VERDICT:", "PASS" if a_ok else "FAIL")
    ok &= a_ok

    # ---- (b) skipped row -------------------------------------------------
    shutil.rmtree(WORK / "out", ignore_errors=True)
    sk_s = base_una()
    add_rows(sk_s, skip_at=1)
    sk_s.RunBatch("accessibility")
    serial_skipped = [(r.index, r.status) for r in sk_s.batch_report.rows]
    shutil.rmtree(WORK / "out", ignore_errors=True)
    sk_p = base_una()
    add_rows(sk_p, skip_at=1)
    sk_p.RunBatch("accessibility", parallel=True, workers=2)
    par_skipped = [(r.index, r.status) for r in sk_p.batch_report.rows]
    bind_ok = (sk_s.projects[1].output_folder ==
               sk_p.projects[1].output_folder)
    print("(b) serial skipped rows:", serial_skipped,
          "| parallel:", par_skipped,
          "| bound output_folder equal:", bind_ok)
    b_ok = serial_skipped == par_skipped and bind_ok
    print("(b) VERDICT:", "PASS" if b_ok else "FAIL")
    ok &= b_ok

    print("PROBE4 VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
