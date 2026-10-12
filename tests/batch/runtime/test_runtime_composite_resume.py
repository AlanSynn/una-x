"""Composite integrity across resume of a FINALIZED composite batch
(BATCH_REVIEW rev-20261003T0105Z MAJOR-1 + MINOR-1).

A checkpointed composite batch that completed has the journal's composite
phase finalized.  On resume, committed rows' captures are NOT re-folded
into the live accumulator — so finalizing from that accumulator alone
used to republish the composite built from ONLY the rows that re-ran
(a silent single-column rebuild over the full published artifact,
MAJOR-1), and a corrupted composite file on disk survived a resume
cycle with a success report (MINOR-1).  The contract here: the resumed
batch either leaves the published composite exactly as published, or
rebuilds it in full from the journal's digest-verified capture payloads
— never a subset.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from conftest import tree_bytes

pytestmark = pytest.mark.batch_runtime

FIXTURE_FILES = ("network.geojson", "origins.geojson", "destinations.geojson")


def _copied_data(smoke_fixture, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    for f in FIXTURE_FILES:
        shutil.copy(smoke_fixture / f, data_dir / f)
    return data_dir


def _composite(i, s):
    s.batch_composite_output = True


def _caps(gdf):
    return [c for c in gdf.columns if c.startswith("knn_access_")]


def _composite_files(out):
    return sorted(f for f in os.listdir(out) if f.startswith("composite"))


def test_resume_after_finalize_rebuilds_full_composite(
        make_batch, smoke_fixture, tmp_path):
    """MAJOR-1: a resume that re-runs one row (tampered artifact) must
    republish the FULL composite — every row's column in caller order,
    byte-identical to the published artifact — never a single-row
    rebuild over it."""
    from dataclasses import replace

    ckpt = str(tmp_path / "ckpt")
    out = tmp_path / "out"

    # run 1 completes and finalizes the composite
    p1 = make_batch(3, out_folder=out, extra=_composite,
                    data_dir=_copied_data(smoke_fixture, tmp_path))
    p1.RunBatch("accessibility", parallel=True, workers=2,
                execution=replace(p1.execution, checkpoint=ckpt))
    assert p1.batch_report is not None
    cols1 = _caps(p1.composite_result)
    assert cols1 == ["knn_access_row0", "knn_access_row1",
                     "knn_access_row2"]
    values1 = {c: np.asarray(p1.composite_result[c]).copy()
               for c in cols1}
    comp_files = _composite_files(out)
    assert comp_files
    bytes1 = {f: open(os.path.join(out, f), "rb").read() for f in comp_files}
    gen1 = p1.batch_report.checkpoint_generation

    # damage one committed row artifact -> resume re-runs row 0
    victims = sorted(f for f in os.listdir(out) if f.startswith("row0"))
    assert victims
    with open(os.path.join(out, victims[0]), "ab") as f:
        f.write(b"TAMPERED-BY-REGRESSION-TEST")

    p2 = make_batch(3, out_folder=out, extra=_composite,
                    data_dir=_copied_data(smoke_fixture, tmp_path))
    p2.RunBatch("accessibility", parallel=True, workers=2,
                execution=replace(p2.execution, checkpoint=ckpt))
    rep = p2.batch_report
    assert rep is not None
    assert any("2 committed row(s) reused" in n for n in rep.notes_runtime)

    # the republished composite is the FULL composite, in caller order
    assert _caps(p2.composite_result) == cols1
    for c in cols1:
        assert np.array_equal(np.asarray(p2.composite_result[c]), values1[c])
    bytes2 = {f: open(os.path.join(out, f), "rb").read() for f in comp_files}
    assert bytes2 == bytes1
    # a republication is a new finalized generation
    assert rep.checkpoint_generation == gen1 + 1


def test_resume_restores_corrupted_composite_files(
        make_batch, smoke_fixture, tmp_path):
    """MINOR-1: post-run corruption of a published composite file must
    not survive a resume cycle silently — when no row re-runs, resume
    verifies the recorded composite artifacts and rebuilds them from the
    journal's verified captures when they fail."""
    from dataclasses import replace

    ckpt = str(tmp_path / "ckpt")
    out = tmp_path / "out"

    p1 = make_batch(3, out_folder=out, extra=_composite,
                    data_dir=_copied_data(smoke_fixture, tmp_path))
    p1.RunBatch("accessibility", parallel=True, workers=2,
                execution=replace(p1.execution, checkpoint=ckpt))
    cols1 = _caps(p1.composite_result)
    comp_files = _composite_files(out)
    assert comp_files
    bytes1 = {f: open(os.path.join(out, f), "rb").read() for f in comp_files}

    # corrupt one published composite file (no row artifact touched)
    feather = next(f for f in comp_files if f.endswith(".feather"))
    with open(os.path.join(out, feather), "wb") as f:
        f.write(b"CORRUPTED-AFTER-PUBLICATION")

    p2 = make_batch(3, out_folder=out, extra=_composite,
                    data_dir=_copied_data(smoke_fixture, tmp_path))
    p2.RunBatch("accessibility", parallel=True, workers=2,
                execution=replace(p2.execution, checkpoint=ckpt))
    rep = p2.batch_report
    assert rep is not None
    # no row re-ran: every committed row was reused
    assert any("3 committed row(s) reused" in n for n in rep.notes_runtime)

    # the corrupted file was restored to its published bytes
    bytes2 = {f: open(os.path.join(out, f), "rb").read() for f in comp_files}
    assert bytes2 == bytes1
    assert _caps(p2.composite_result) == cols1
