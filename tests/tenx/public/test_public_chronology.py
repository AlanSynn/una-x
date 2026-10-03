"""H01 chronology tests: public-surface reference behaviors measured on
the observed Boston cohort (receipts under campaign_runs/una_10x), pinned
on the hermetic smoke grid.

Pinned here:
1. Public RunAccessibility twice -> artifacts bitwise-equal keyed by the
   per-job file name; every published file lives under a minute-stamped
   export directory (wall-clock metadata by design).
2. Public RunBatch (serial default) twice into separate folders -> equal
   ordered commit sequences (index/name/status), zero failed rows, and
   bitwise-equal per-job artifacts across the two runs.
3. First in-process call pays JIT trace construction inside the public
   boundary; cold vs warm outputs are bitwise-equal per job name.

Parallel-runtime fault behaviors (worker death, deadline cancel,
checkpoint resume, tamper rejection) are pinned by tests/batch/runtime;
the campaign fault_matrix.json records the same surface exercised
through the public API on the observed cohort.
"""
from __future__ import annotations

import hashlib
import os
import re

import pytest

from urban_network_analysis.Settings import Settings
# UNA imports call-time inside tests (whole-repo collection safety; see
# the note in tests/execution/test_runbatch_serial.py).


PAIRING_HEADER = (
    "name,data_folder,network_file,origins_file,destinations_file,"
    "search_radius,output_folder,output_wStamp,gravity_beta\n")

EXPORT_DIR = re.compile(r"^accessibility_\d{4}-\d{2}-\d{2}_\d{4}$")

CONFIG = {
    "network_file": "network.geojson",
    "origins_file": "origins.geojson",
    "destinations_file": "destinations.geojson",
    "search_radius": 400,
    "calculate_reach": True,
    "calculate_exponential_gravity": True,
    "gravity_beta": 0.001,
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _tree_by_name(folder):
    """Artifact digests keyed by per-job file name (timestamped export
    directories are wall-clock metadata, not content)."""
    rec = {}
    for dp, _, files in os.walk(folder):
        for fn in files:
            rec[fn] = _sha256(os.path.join(dp, fn))
    return rec


def _run_once(fixture, out, name):
    from urban_network_analysis import UNA
    s = Settings()
    row = dict(CONFIG)
    row["name"] = name
    row["output_file_name"] = name
    s.ApplyRow(row)
    s.data_folder = str(fixture)
    s.output_folder = str(out)
    una = UNA(verbosity=0)
    una.settings = s
    una.RunAccessibility()
    return una


def _pairing(tmp_path, fixture, out, betas=(0.001, 0.002)):
    lines = [PAIRING_HEADER]
    for i, beta in enumerate(betas):
        lines.append(
            f"beta_{i},{fixture},network.geojson,origins.geojson,"
            f"destinations.geojson,400,{out},False,{beta!r}\n")
    stem = "pairing_" + os.path.basename(str(out)) + ".csv"
    csv = tmp_path / stem
    csv.write_text("".join(lines))
    return str(csv)


def test_runaccessibility_twice_bitwise_by_job_name(smoke_fixture,
                                                    tmp_path):
    trees = []
    for i in (1, 2):
        out = tmp_path / f"ra{i}"
        _run_once(smoke_fixture, out, "chron_ra")
        trees.append(_tree_by_name(out))
        assert trees[-1], "no artifacts published"
    names = sorted(trees[0])
    assert names == sorted(trees[1])
    for name in names:
        assert trees[0][name] == trees[1][name], name
    for i in (1, 2):
        for dp, _, files in os.walk(tmp_path / f"ra{i}"):
            for fn in files:
                rel = os.path.relpath(os.path.join(dp, fn),
                                      tmp_path / f"ra{i}")
                assert EXPORT_DIR.match(rel.split(os.sep)[0]), rel


def test_runbatch_twice_ordered_commits_and_bitwise(tmp_path,
                                                    smoke_fixture):
    from urban_network_analysis import UNA
    pair1 = _pairing(tmp_path, smoke_fixture, str(tmp_path / "batch1"))
    pair2 = _pairing(tmp_path, smoke_fixture, str(tmp_path / "batch2"))
    reports = []
    for pair, out in ((pair1, tmp_path / "batch1"),
                      (pair2, tmp_path / "batch2")):
        una = UNA(verbosity=0)
        una.settings.data_folder = str(smoke_fixture)
        una.settings.output_folder = str(out)
        una.RunBatch("accessibility", pair)
        assert una.batch_report is not None
        reports.append(una.batch_report)
    r1, r2 = reports
    seq1 = [(r.index, r.name, r.status) for r in r1.rows]
    seq2 = [(r.index, r.name, r.status) for r in r2.rows]
    assert seq1 == seq2
    assert seq1 == [(0, "beta_0", "ran"), (1, "beta_1", "ran")]
    assert r1.parallel_requested is False
    assert r1.workers_requested is None
    t1 = _tree_by_name(tmp_path / "batch1")
    t2 = _tree_by_name(tmp_path / "batch2")
    assert t1 and sorted(t1) == sorted(t2)
    for name in sorted(t1):
        assert t1[name] == t2[name], name


def test_first_trace_inside_public_boundary_outputs_unchanged(
        tmp_path, smoke_fixture):
    cold = tmp_path / "cold"
    warm = tmp_path / "warm"
    for tag, out in (("cold", cold), ("warm", warm)):
        _run_once(smoke_fixture, out, "chron")
    t1 = _tree_by_name(cold)
    t2 = _tree_by_name(warm)
    assert t1 and sorted(t1) == sorted(t2)
    for name in sorted(t1):
        assert t1[name] == t2[name], name
