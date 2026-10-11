"""Deterministic commit ordering and ordered failure semantics (dossier 04).

The coordinator may receive worker results in ANY completion order; rows
must still commit strictly in caller order at a single cursor (the
observable-prefix invariant).  These tests use injected sleeps to force a
completion order opposite to caller order, a commit hook to record the
actual commit sequence, and serial/parallel comparisons for the ordered
failure paths.
"""
from __future__ import annotations

import os

import pytest

from conftest import tree_bytes

pytestmark = pytest.mark.batch_runtime


def _run_direct(una, analysis, workers, on_commit=None, execution=None):
    from urban_network_analysis.Execution import admit_execution
    from urban_network_analysis.batch import runtime as rt

    requested = execution or una.execution
    rt.run_batch_parallel(
        una, analysis, workers=workers,
        requested=requested, effective=admit_execution(requested),
        script_output_folder=None, notes=(), _on_commit=on_commit)


def test_commit_order_is_caller_order_under_completion_noise(
        make_batch, tmp_path, monkeypatch):
    from urban_network_analysis.batch import runtime as rt

    commits = []
    orig_commit = rt._commit_worker_row

    def commit(una, plan, journal, i, *args, **kwargs):
        commits.append(i)
        return orig_commit(una, plan, journal, i, *args, **kwargs)

    monkeypatch.setattr(rt, "_commit_worker_row", commit)

    # early rows sleep LONGEST: results arrive 3,2,1,0
    def extra(i, s):
        sleeps = [1.2, 0.8, 0.4, 0.0]
        # set EVERY row: the live Settings object is shared across rows,
        # so a conditional assignment leaks into later rows' snapshots
        s.inject_specs = ([{"phase": "pre_run", "action": "sleep",
                            "arg": sleeps[i]}] if sleeps[i] else [])

    p = make_batch(4, out_folder=tmp_path / "out", extra=extra)
    p.RunBatch("accessibility", parallel=True, workers=4)

    assert commits == [0, 1, 2, 3]
    assert p.batch_report.commit_order == (0, 1, 2, 3)
    assert p.batch_report.fold_order == (0, 1, 2, 3)


def test_observable_prefix_speculation_is_invisible(
        make_batch, tmp_path):
    """At the commit of row i, rows <= i are published and rows > i exist
    nowhere in the public tree — even though later rows FINISHED first
    (dossier 04's formal invariant, checked at every commit)."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    violations = []

    def on_commit(i):
        done = {name.split(".")[0] for name in os.listdir(out)
                if name.startswith("row")}
        missing = [f"row{j}" for j in range(i + 1) if f"row{j}" not in done]
        leaked = [stem for stem in done
                  if stem[3:].isdigit() and int(stem[3:]) > i]
        if missing or leaked:
            violations.append((i, missing, leaked))

    # early rows sleep LONGEST: rows finish 3,2,1,0
    def extra(i, s):
        sleeps = [1.2, 0.8, 0.4, 0.0]
        s.inject_specs = ([{"phase": "pre_run", "action": "sleep",
                            "arg": sleeps[i]}] if sleeps[i] else [])

    p = make_batch(4, out_folder=out, extra=extra)
    _run_direct(p, "accessibility", workers=4, on_commit=on_commit)

    assert not violations, f"prefix invariant violated: {violations}"
    stems = {n.split(".")[0] for n in os.listdir(out)
             if n.split(".")[0].startswith("row")}
    assert sorted(stems) == ["row0", "row1", "row2", "row3"]
    assert not [n for n in os.listdir(out) if n.startswith(".una-batch")]


def test_plan_error_row_raises_serial_exception_after_prefix(
        make_batch, tmp_path):
    """A row the planner already knows is erroneous (truthy non-string
    required field) runs on the caller's instance at the cursor and raises
    the exact serial exception; the prefix stays committed and exposed on
    the exception; no report is set (serial parity)."""
    serial = make_batch(3, out_folder=tmp_path / "ser")
    serial.projects[1].network_file = 123       # truthy non-string
    with pytest.raises(AttributeError) as es:
        serial.RunBatch("accessibility")

    par = make_batch(3, out_folder=tmp_path / "par")
    par.projects[1].network_file = 123
    with pytest.raises(AttributeError) as ep:
        par.RunBatch("accessibility", parallel=True, workers=2)

    assert type(es.value) is type(ep.value)
    assert par.batch_report is None
    rows = ep.value.batch_rows
    assert [r.index for r in rows] == [0]
    assert rows[0].status == "ran" and rows[0].phase == "COMMITTED"
    # prefix artifacts published, error row and beyond never ran
    assert any("row0" in f for f in rows[0].output_files)
    for f in rows[0].output_files:
        assert os.path.exists(f)
    stems = {n.split(".")[0] for n in os.listdir(tmp_path / "par")}
    assert not {s for s in stems if s.startswith("row")} - {"row0"}


def test_worker_row_failure_raises_original_exception_after_prefix(
        make_batch, tmp_path):
    """A worker row that fails at runtime re-raises the original exception
    (same type/message as the serial loop would produce) once the prefix
    has committed; speculation from rows after the failed one never
    publishes."""
    def extra(i, s):
        s.inject_specs = ([{"phase": "pre_run", "action": "raise_memory"}]
                          if i == 1 else [])

    p = make_batch(3, out_folder=tmp_path / "out", extra=extra)
    with pytest.raises(MemoryError, match="injected OOM") as ei:
        p.RunBatch("accessibility", parallel=True, workers=2)

    assert p.batch_report is None
    rows = ei.value.batch_rows
    assert [r.index for r in rows] == [0]
    assert rows[0].status == "ran"
    # staged speculation discarded: only the committed prefix is public
    stems = {n.split(".")[0] for n in os.listdir(tmp_path / "out")}
    assert {s for s in stems if s.startswith("row")} == {"row0"}
    assert not [n for n in os.listdir(tmp_path / "out")
                if n.startswith(".una-batch-staging")]


def test_skipped_row_bind_semantics_match_serial(make_batch, tmp_path):
    """A falsy required field skips the row, but its bind transition
    (output-folder defaulting) still mutates the project object exactly as
    the serial loop does before the skip gate (BATCH_PLAN NOTE-2).  The
    row's output_folder is cleared so the defaulting decision is actually
    observable: the documented precedence (row > script > data_folder/
    Results) resolves the cleared row to the SCRIPT-level folder, which is
    what make_batch configured on the instance."""
    serial = make_batch(3, out_folder=tmp_path / "ser")
    serial.projects[1].network_file = ""
    serial.projects[1].output_folder = ""
    serial.RunBatch("accessibility")

    par = make_batch(3, out_folder=tmp_path / "par")
    par.projects[1].network_file = ""
    par.projects[1].output_folder = ""
    par.RunBatch("accessibility", parallel=True, workers=2)

    rep = par.batch_report
    assert [r.status for r in rep.rows] == ["ran", "skipped", "ran"]
    assert rep.rows[1].phase == "SKIPPED"
    assert "missing required fields" in rep.rows[1].detail

    # the bind transition's observable effect, identical to serial: each
    # route resolves the cleared row to its own script-level folder
    assert par.projects[1].output_folder == str(tmp_path / "par")
    assert serial.projects[1].output_folder == str(tmp_path / "ser")
    # rows still ran around the skipped one; commit_order carries the
    # rows that will EXECUTE (reviewed BATCH_PLAN semantics: a skipped
    # row transitions at the cursor but is never a commit)
    assert rep.commit_order == (0, 2)


def test_staged_digest_verification_blocks_publishing(
        make_batch, tmp_path, monkeypatch):
    """The coordinator re-verifies staged bytes against the worker's
    digests BEFORE publishing — a lying/corrupted result must abort the
    batch without publishing unverified bytes, as a ROW-ATTRIBUTED
    integrity error, never BatchCancelledError (BATCH_REVIEW
    rev-20261003T0105Z MINOR-2: corruption and cancellation are
    different failure classes with different replay semantics; a
    cancel-retry loop would spin forever on permanent corruption)."""
    from dataclasses import replace as dreplace

    from urban_network_analysis.batch import runtime as rt
    from urban_network_analysis.batch.plan import InvariantViolation

    orig_recv = rt._Pool.recv

    def recv(self, timeout):
        result = orig_recv(self, timeout)
        if result.status == "ran" and result.files:
            first = result.files[0]
            doctored = ((first[0], first[1], "0" * 64),) + result.files[1:]
            result = dreplace(result, files=doctored)
        return result

    monkeypatch.setattr(rt._Pool, "recv", recv)

    p = make_batch(2, out_folder=tmp_path / "out")
    with pytest.raises(InvariantViolation,
                       match="digest verification") as ei:
        p.RunBatch("accessibility", parallel=True, workers=1)
    assert ei.value.row == 0
    # nothing was published from the unverified result
    assert not os.path.exists(tmp_path / "out") or \
        not [n for n in os.listdir(tmp_path / "out")
             if n.startswith(("row", ".una-batch"))]


def test_staged_vanish_is_row_attributed_integrity_error(tmp_path):
    """The vanished-staged-artifact branch carries the same typed,
    row-attributed integrity contract (MINOR-2): the error names the row
    instead of masquerading as a cancellation."""
    from urban_network_analysis.batch import runtime as rt
    from urban_network_analysis.batch.plan import InvariantViolation

    with pytest.raises(InvariantViolation, match="vanished") as ei:
        rt._verify_staged(str(tmp_path),
                          (("gone.feather", 1, "0" * 64),), row=3)
    assert ei.value.row == 3


def test_publish_staged_fsyncs_files_and_directories(tmp_path, monkeypatch):
    """BATCH_REVIEW rev-20261003T0105Z MINOR-4: publication fsyncs every
    published file and its parent directory at commit — the journal's
    durability discipline mirrored at the os.replace site (without it, a
    crash/power-loss window can leave torn or mixed-generation published
    outputs behind the rename)."""
    from urban_network_analysis.batch import runtime as rt

    staged = tmp_path / "staging"
    staged.mkdir()
    public = tmp_path / "public"
    public.mkdir()
    (staged / "a.feather").write_bytes(b"AAAA")
    (staged / "sub").mkdir()
    (staged / "sub" / "b.csv").write_bytes(b"BBBB")

    real_open = os.open
    fsynced = []

    def spy_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        # every os.open in _publish_staged is an fsync target (files via
        # _fsync_published, directories via _fsync_dir): the digests
        # arrive precomputed and nothing else here reads by descriptor
        fsynced.append(os.path.abspath(path))
        return fd

    monkeypatch.setattr(rt.os, "open", spy_open)
    published = rt._publish_staged(
        str(staged), str(public),
        (("a.feather", 4, "0" * 64), ("sub" + os.sep + "b.csv", 4,
                                      "0" * 64)))

    assert sorted(os.path.relpath(p, public) for p in published) == \
        ["a.feather", os.path.join("sub", "b.csv")]
    assert all(os.path.exists(p) for p in published)
    # both published files AND every touched parent directory fsynced
    assert fsynced[:2] == [os.path.join(public, "a.feather"),
                           os.path.join(public, "sub", "b.csv")]
    assert set(fsynced[2:]) == {str(public), str(public / "sub")}


def test_shared_output_name_serializes_and_matches_serial(
        make_batch, tmp_path):
    """Two rows writing the same output basenames are a planner-proven
    write-write hazard: both must run on the serial route (recorded
    reasons), in caller order, and the final tree must match the serial
    route's overwrite semantics exactly."""
    def extra(i, s):
        s.name = "dup"                # same stem -> same output basenames

    serial = make_batch(2, out_folder=tmp_path / "ser", extra=extra)
    serial.RunBatch("accessibility")

    par = make_batch(2, out_folder=tmp_path / "par", extra=extra)
    par.RunBatch("accessibility", parallel=True, workers=2)

    rep = par.batch_report
    assert rep.worker_admissible == ()
    assert [idx for idx, _ in rep.serialized] == [0, 1]
    assert all(r.admission == "serial" and r.admission_reasons
               for r in rep.rows)
    assert all(r.worker_pid == 0 for r in rep.rows)   # never dispatched
    assert tree_bytes(tmp_path / "ser") == tree_bytes(tmp_path / "par")
