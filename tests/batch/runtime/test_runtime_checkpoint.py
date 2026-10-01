"""Checkpoint / crash-recovery / resume invariants (dossier 05).

A checkpointed batch records committed progress (identity = code/model/
profile/input digests, per-row artifacts with digests, composite capture
payloads as raw .npy).  Resume reuses only verified COMMITTED units,
rejects changed identity rather than mixing generations, re-runs rows
whose artifacts fail verification, and re-running a COMPLETED batch is
idempotent — no double-counted composites, no rewritten artifacts.
"""
from __future__ import annotations

import os
import shutil

import pytest

from urban_network_analysis.batch.report import (BatchCancelledError,
                                                 BatchCheckpointError)

pytestmark = pytest.mark.batch_runtime

FIXTURE_FILES = ("network.geojson", "origins.geojson", "destinations.geojson")


def _copied_data(smoke_fixture, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    for f in FIXTURE_FILES:
        shutil.copy(smoke_fixture / f, data_dir / f)
    return data_dir


def _mtime_ns(paths):
    return {p: os.stat(p).st_mtime_ns for p in paths}


def test_crash_resume_completes_and_reuses_committed_rows(
        make_batch, smoke_fixture, tmp_path):
    ckpt = str(tmp_path / "ckpt")
    out = tmp_path / "out"

    def extra_kill(i, s):
        s.inject_specs = ([{"phase": "startup", "action": "exit",
                            "arg": 1}] if i == 1 else [])

    # run 1: row 0 commits (workers=1 makes that deterministic), row 1's
    # worker dies at startup -> cancellation, journal keeps row 0.
    p1 = make_batch(3, out_folder=out, extra=extra_kill,
                    data_dir=_copied_data(smoke_fixture, tmp_path))
    from dataclasses import replace
    with pytest.raises(BatchCancelledError):
        p1.RunBatch("accessibility", parallel=True, workers=1,
                    execution=replace(p1.execution, checkpoint=ckpt))
    row0_files = [f for f in os.listdir(out) if f.startswith("row0")]
    assert row0_files
    row0_before = _mtime_ns([os.path.join(out, f) for f in row0_files])

    # run 2: same batch, no fault — row 0 is reused, 1..2 run to completion
    p2 = make_batch(3, out_folder=out,
                    data_dir=_copied_data(smoke_fixture, tmp_path))
    p2.RunBatch("accessibility", parallel=True, workers=2,
                execution=replace(p2.execution, checkpoint=ckpt))

    rep = p2.batch_report
    assert [r.status for r in rep.rows] == ["ran", "ran", "ran"]
    assert rep.commit_order == (0, 1, 2)
    assert rep.checkpoint_generation == 2     # row records -> finalize bump
    assert any("reused" in n for n in rep.notes_runtime)
    # reused row 0 was NOT re-executed: artifacts untouched
    assert _mtime_ns([os.path.join(out, f) for f in row0_files]) == \
        row0_before
    # report completeness: the reused row's outcome carries its committed
    # artifacts (verified against the journal record), like any other row
    assert sorted(os.path.basename(f)
                  for f in rep.rows[0].output_files) == sorted(row0_files)
    assert all(os.path.exists(f) for f in rep.rows[0].output_files)
    assert rep.output_manifest["0"] == rep.rows[0].output_files
    for i in (1, 2):
        assert [f for f in os.listdir(out) if f.startswith(f"row{i}")]


def test_settings_change_rejects_checkpoint_resume(make_batch, tmp_path):
    """The identity's ``model`` quarter: a changed Settings closure (here a
    row's search_radius) must reject resume instead of silently reusing
    the stale committed row under the new parameters (review MAJOR-1)."""
    ckpt = str(tmp_path / "ckpt")

    from dataclasses import replace
    p = make_batch(2, out_folder=tmp_path / "out")
    p.RunBatch("accessibility", parallel=True, workers=2,
               execution=replace(p.execution, checkpoint=ckpt))

    journal_path = os.path.join(ckpt, "journal.json")
    before = open(journal_path, "rb").read()

    p2 = make_batch(2, out_folder=tmp_path / "out")
    p2.projects[1].search_radius = 123        # row 1 ran with 100
    with pytest.raises(BatchCheckpointError, match="identity mismatch"):
        p2.RunBatch("accessibility", parallel=True, workers=2,
                    execution=replace(p2.execution, checkpoint=ckpt))
    # the recorded generation was neither overwritten nor quarantined
    assert open(journal_path, "rb").read() == before
    assert not [n for n in os.listdir(ckpt) if "quarantin" in n]


def test_tampered_capture_payload_forces_row_rerun(make_batch, smoke_fixture,
                                                   tmp_path):
    """A tampered composite capture payload must drop the row from the
    generation at resume verification (re-run from scratch) — never fold
    an unverified payload or fail mid-cursor after other rows ran
    (review MINOR-1)."""
    ckpt = str(tmp_path / "ckpt")
    out = tmp_path / "out"

    def composite(i, s):
        s.batch_composite_output = True

    def extra_kill(i, s):
        composite(i, s)
        s.inject_specs = ([{"phase": "startup", "action": "exit",
                            "arg": 1}] if i == 1 else [])

    from dataclasses import replace
    p1 = make_batch(3, out_folder=out, extra=extra_kill,
                    data_dir=_copied_data(smoke_fixture, tmp_path))
    with pytest.raises(BatchCancelledError):
        p1.RunBatch("accessibility", parallel=True, workers=1,
                    execution=replace(p1.execution, checkpoint=ckpt))

    # row 0 committed with a capture payload; corrupt it on disk
    caps = [f for f in os.listdir(os.path.join(ckpt, "captures"))
            if f.startswith("row0")]
    assert caps
    with open(os.path.join(ckpt, "captures", caps[0]), "ab") as f:
        f.write(b"TAMPERED")

    # serial reference for the same composite batch
    ser = make_batch(3, out_folder=tmp_path / "ser", extra=composite)
    ser.RunBatch("accessibility")

    p2 = make_batch(3, out_folder=out, extra=composite,
                    data_dir=_copied_data(smoke_fixture, tmp_path))
    p2.RunBatch("accessibility", parallel=True, workers=2,
                execution=replace(p2.execution, checkpoint=ckpt))

    rep = p2.batch_report
    # the tampered row was re-run, not folded from unverified bytes
    assert not any("reused" in n for n in rep.notes_runtime)
    cols = [c for c in p2.composite_result.columns
            if c.startswith("knn_access_")]
    ser_cols = [c for c in ser.composite_result.columns
                if c.startswith("knn_access_")]
    assert cols == ser_cols
    # the fresh capture republished its verified digest in the journal
    import hashlib
    import json
    with open(os.path.join(ckpt, "journal.json"), encoding="utf-8") as f:
        doc = json.load(f)
    cap = doc["rows"]["0"]["capture"]
    with open(os.path.join(ckpt, "captures", cap["path"]), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == cap["sha256"]


def test_identity_mismatch_rejected_not_mixed(make_batch, smoke_fixture,
                                              tmp_path):
    ckpt = str(tmp_path / "ckpt")
    data_dir = _copied_data(smoke_fixture, tmp_path)

    p = make_batch(2, out_folder=tmp_path / "out", data_dir=data_dir)
    from dataclasses import replace
    p.RunBatch("accessibility", parallel=True, workers=2,
               execution=replace(p.execution, checkpoint=ckpt))

    # change a batch input: the recorded identity no longer matches
    origins = data_dir / "origins.geojson"
    with open(origins, "a", encoding="utf-8") as f:
        f.write("\n")

    journal_path = os.path.join(ckpt, "journal.json")
    before = open(journal_path, "rb").read()
    with pytest.raises(BatchCheckpointError, match="identity mismatch"):
        p.RunBatch("accessibility", parallel=True, workers=2,
                   execution=replace(p.execution, checkpoint=ckpt))
    # the recorded generation was neither overwritten nor quarantined
    assert open(journal_path, "rb").read() == before
    assert not [n for n in os.listdir(ckpt) if "quarantin" in n]


def test_tampered_artifact_forces_row_rerun(make_batch, tmp_path):
    ckpt = str(tmp_path / "ckpt")
    out = tmp_path / "out"

    from dataclasses import replace
    p = make_batch(2, out_folder=out)
    p.RunBatch("accessibility", parallel=True, workers=2,
               execution=replace(p.execution, checkpoint=ckpt))
    good = {n: open(os.path.join(out, n), "rb").read()
            for n in os.listdir(out)}

    victim = sorted(n for n in good if n.startswith("row0"))[0]
    with open(os.path.join(out, victim), "ab") as f:
        f.write(b"TAMPERED")

    p2 = make_batch(2, out_folder=out)
    p2.RunBatch("accessibility", parallel=True, workers=2,
                execution=replace(p2.execution, checkpoint=ckpt))
    # the tampered row was re-run and republished its verified bytes; the
    # untouched row stayed reused
    assert open(os.path.join(out, victim), "rb").read() == good[victim]
    assert {n: open(os.path.join(out, n), "rb").read()
            for n in os.listdir(out)} == good
    assert [r.status for r in p2.batch_report.rows] == ["ran", "ran"]


def test_completed_batch_rerun_is_idempotent(make_batch, tmp_path):
    """Re-running a completed batch reuses every unit: no artifact is
    rewritten, no worker re-runs a committed row, and the final state is
    still rehydrated — here through a state-only rerun of a SERIALIZED
    (parent-class) last row, since a fresh process holds no live state."""
    ckpt = str(tmp_path / "ckpt")
    out = tmp_path / "out"

    def dup_last_pair(i, s):
        if i >= 2:
            s.name = "dup"      # rows 2,3 conflict -> serialized (parent)

    from dataclasses import replace
    opts = replace(make_batch(4, out_folder=out).execution, checkpoint=ckpt)

    p = make_batch(4, out_folder=out, extra=dup_last_pair)
    p.RunBatch("accessibility", parallel=True, workers=2, execution=opts)
    tree1 = {n: open(os.path.join(out, n), "rb").read()
             for n in os.listdir(out)}
    gen1 = p.batch_report.checkpoint_generation
    assert [idx for idx, _ in p.batch_report.serialized] == [2, 3]

    p2 = make_batch(4, out_folder=out, extra=dup_last_pair)
    p2.RunBatch("accessibility", parallel=True, workers=2, execution=opts)
    rep = p2.batch_report
    # nothing re-executed, nothing rewritten, nothing duplicated
    tree2 = {n: open(os.path.join(out, n), "rb").read()
             for n in os.listdir(out)}
    assert tree1 == tree2
    assert rep.workers_effective == 0 and rep.pools == ()
    assert all(r.phase == "COMMITTED" and r.worker_pid == 0
               for r in rep.rows)
    assert rep.checkpoint_generation == gen1
    assert any("reused" in n for n in rep.notes_runtime)
    # final state rehydrated without re-running the row for publication
    assert any("state-only rerun" in n for n in rep.notes_runtime)


def test_corrupt_journal_quarantines_and_reruns_fresh(make_batch, tmp_path):
    ckpt = str(tmp_path / "ckpt")
    out = tmp_path / "out"

    from dataclasses import replace
    opts = replace(make_batch(2, out_folder=out).execution, checkpoint=ckpt)

    p = make_batch(2, out_folder=out)
    p.RunBatch("accessibility", parallel=True, workers=2, execution=opts)

    with open(os.path.join(ckpt, "journal.json"), "w", encoding="utf-8") as f:
        f.write("{corrupt!!")

    p2 = make_batch(2, out_folder=out)
    p2.RunBatch("accessibility", parallel=True, workers=2, execution=opts)
    # quarantine preserved (never deleted) with its reason; fresh run done
    quarantined = [n for n in os.listdir(ckpt) if n.endswith(".quarantined")]
    assert quarantined
    assert os.path.exists(os.path.join(ckpt, quarantined[0] + ".reason"))
    assert [r.status for r in p2.batch_report.rows] == ["ran", "ran"]
    assert not any("reused" in n for n in p2.batch_report.notes_runtime)


def test_resume_rebinds_skipped_rows_and_reports_them_skipped(
        make_batch, smoke_fixture, tmp_path):
    """A skipped row recorded in a previous generation keeps its ordered
    semantics across resume: its bind transition re-applies (dossier 04
    NOTE-2) and it reports skipped — never silently 'ran'."""
    ckpt = str(tmp_path / "ckpt")
    out = tmp_path / "out"

    def skip_row1(i, s):
        # SET EVERY ROW: make_batch reuses one live Settings object
        # across rows, so a conditional assignment leaks into the next
        # rows' project snapshots (the composite opt-out test's finding).
        s.network_file = "" if i == 1 else "network.geojson"

    def extra_kill(i, s):
        skip_row1(i, s)
        s.inject_specs = ([{"phase": "startup", "action": "exit",
                            "arg": 1}] if i == 2 else [])

    from dataclasses import replace
    data_dir = _copied_data(smoke_fixture, tmp_path)
    p1 = make_batch(4, out_folder=out, extra=extra_kill, data_dir=data_dir)
    with pytest.raises(BatchCancelledError):
        p1.RunBatch("accessibility", parallel=True, workers=1,
                    execution=replace(p1.execution, checkpoint=ckpt))

    p2 = make_batch(4, out_folder=out, extra=skip_row1, data_dir=data_dir)
    p2.RunBatch("accessibility", parallel=True, workers=2,
                execution=replace(p2.execution, checkpoint=ckpt))

    rep = p2.batch_report
    assert [r.status for r in rep.rows] == \
        ["ran", "skipped", "ran", "ran"]
    assert [r.phase for r in rep.rows] == \
        ["COMMITTED", "SKIPPED", "COMMITTED", "COMMITTED"]
    # the skipped row's bind transition re-applied, exactly as serial:
    # the documented precedence (row > script > data_folder/Results)
    # resolves the cleared row to the batch's script-level folder (the
    # out folder both runs configured on the instance)
    assert p2.projects[1].output_folder == str(out)
    assert any("reused" in n for n in rep.notes_runtime)


def test_resume_folds_composite_once_no_duplicate_columns(
        make_batch, smoke_fixture, tmp_path):
    ckpt = str(tmp_path / "ckpt")
    out = tmp_path / "out"

    def composite(i, s):
        s.batch_composite_output = True

    def extra_kill(i, s):
        composite(i, s)
        s.inject_specs = ([{"phase": "startup", "action": "exit",
                            "arg": 1}] if i == 1 else [])

    from dataclasses import replace
    p1 = make_batch(3, out_folder=out, extra=extra_kill,
                    data_dir=_copied_data(smoke_fixture, tmp_path))
    with pytest.raises(BatchCancelledError):
        p1.RunBatch("accessibility", parallel=True, workers=1,
                    execution=replace(p1.execution, checkpoint=ckpt))

    # serial reference for the same composite batch
    ser = make_batch(3, out_folder=tmp_path / "ser", extra=composite)
    ser.RunBatch("accessibility")
    ser_cols = [c for c in ser.composite_result.columns
                if c.startswith("knn_access_")]

    p2 = make_batch(3, out_folder=out, extra=composite,
                    data_dir=_copied_data(smoke_fixture, tmp_path))
    p2.RunBatch("accessibility", parallel=True, workers=2,
                execution=replace(p2.execution, checkpoint=ckpt))

    comp = p2.composite_result
    cols = [c for c in comp.columns if c.startswith("knn_access_")]
    # row 0's capture came from the journal, rows 1..2 from their runs:
    # exactly one column per row, in caller order — never duplicated
    assert cols == ser_cols == ["knn_access_row0", "knn_access_row1",
                                "knn_access_row2"]
    import numpy as np
    for c in cols:
        assert np.array_equal(np.asarray(comp[c]),
                              np.asarray(ser.composite_result[c]))

    # dossier 05: re-running the now-COMPLETED batch must not duplicate
    # composites — the journal's finalized phase suppresses refolding
    comp_after_p2 = [f for f in os.listdir(out) if f.startswith("composite")]
    mtimes_p2 = {f: os.stat(os.path.join(out, f)).st_mtime_ns
                 for f in comp_after_p2}
    assert comp_after_p2 == [f for f in os.listdir(tmp_path / "ser")
                             if f.startswith("composite")]

    p3 = make_batch(3, out_folder=out, extra=composite,
                    data_dir=_copied_data(smoke_fixture, tmp_path))
    p3.RunBatch("accessibility", parallel=True, workers=2,
                execution=replace(p3.execution, checkpoint=ckpt))
    cols3 = [c for c in p3.composite_result.columns
             if c.startswith("knn_access_")] \
        if p3.composite_result is not None else []
    assert cols3 == [] or cols3 == cols
    # idempotence down to the bytes on disk: the finalized generation is
    # not refolded, so no composite file is REwritten either
    comp_after_p3 = [f for f in os.listdir(out) if f.startswith("composite")]
    assert comp_after_p3 == comp_after_p2
    assert {f: os.stat(os.path.join(out, f)).st_mtime_ns
            for f in comp_after_p3} == mtimes_p2
