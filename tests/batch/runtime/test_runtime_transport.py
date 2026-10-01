"""Cross-process transport guarantees (review MINOR-2 / MINOR-3 / MINOR-4).

The worker pool moves results through multiprocessing queues whose feeder
threads die SILENTLY on an unpicklable object — stranding every later
message.  Every result is therefore degraded to a transportable form
(preserving the original failure type/message/args), every job is
probe-pickled in the coordinator before dispatch, and a rehydration that
cannot deliver serial public state fails the batch loudly instead of
silently deviating.
"""
from __future__ import annotations

import pickle
import threading

import pytest

from urban_network_analysis.batch.report import (BatchCancelledError,
                                                 BatchCheckpointError)
from urban_network_analysis.batch.worker import (RowJob, RowResult,
                                                 _transport_safe)

pytestmark = pytest.mark.batch_runtime


class _HoldsLock(Exception):
    """An exception whose payload resists pickling (carries a live lock)."""

    def __init__(self, msg):
        super().__init__(msg)
        self.lock = threading.Lock()      # unpicklable attribute


def _base_result(**kw):
    defaults = dict(job_id="j", row_index=0, status="ran")
    defaults.update(kw)
    return RowResult(**defaults)


# -- worker-side degradation -------------------------------------------------

def test_transportable_result_passes_through():
    r = _base_result(settings_post={"a": 1}, files=(("x", 1, "d"),))
    assert _transport_safe(r) is r


def test_unpicklable_exception_degrades_keeping_type_msg_args():
    exc = _HoldsLock("boom")
    failed = RowResult(job_id="j", row_index=3, status="failed",
                       error_exc=exc, error_type="ValueError",
                       error_msg="boom", error_args=("boom",))
    out = _transport_safe(failed)
    assert out is not failed
    assert out.status == "failed"
    assert out.error_exc is None                    # live object dropped
    assert out.error_type == "ValueError"           # original type kept
    assert out.error_msg == "boom"                  # original message kept
    assert out.error_args == ("boom",)              # exact construction
    pickle.dumps(out)                               # now transportable


def test_unpicklable_payload_reduces_to_text_failure():
    r = _base_result(settings_post={"f": lambda: None})   # lambda: unpicklable
    out = _transport_safe(r)
    assert out.status == "failed"
    assert out.error_type == "RuntimeError"
    assert "could not be transported" in out.error_msg
    assert out.settings_post == {}
    pickle.dumps(out)


# -- coordinator-side reconstruction -----------------------------------------

def test_worker_failure_rebuilds_exact_serial_construction():
    from urban_network_analysis.batch.runtime import _worker_failure

    # error_exc wins when it survived transport
    exc = ValueError("direct")
    assert _worker_failure(_base_result(
        status="failed", error_exc=exc)) is exc

    # picklable args rebuild the EXACT serial exception, including
    # __str__ transformations (KeyError repr-quotes its argument)
    e = _worker_failure(_base_result(
        status="failed", error_exc=None, error_type="KeyError",
        error_msg="'missing'", error_args=("missing",)))
    assert type(e) is KeyError and str(e) == "'missing'"

    # builtin type, no transportable args: rebuilt from the recorded text
    e = _worker_failure(_base_result(
        status="failed", error_exc=None, error_type="MemoryError",
        error_msg="injected OOM", error_args=None))
    assert type(e) is RuntimeError
    assert "MemoryError" in str(e) and "injected OOM" in str(e)


def test_untransportable_job_fails_its_row_loudly(make_batch, tmp_path,
                                                  monkeypatch):
    """A row job whose state resists pickling must fail THAT row through
    the normal ordering path (earliest failure, prefix preserved) — never
    strand the inbox feeder and spin to a misleading deadline (review
    MINOR-3)."""
    from urban_network_analysis.batch import runtime as rt

    def unpicklable_job(una, plan, i, staging_dir, analysis,
                        script_output_folder, need_state):
        return RowJob(
            job_id=f"j{i}", row_index=i, analysis=analysis,
            settings_state={"callback": lambda: None},  # unpicklable
            script_output_folder=script_output_folder,
            staging_dir=staging_dir, capture=False, need_state=need_state)

    monkeypatch.setattr(rt, "_job_for", unpicklable_job)

    p = make_batch(2, out_folder=tmp_path / "out")
    with pytest.raises(RuntimeError, match="could not be transported") as ei:
        p.RunBatch("accessibility", parallel=True, workers=1)
    assert p.batch_report is None
    assert ei.value.batch_rows == ()      # nothing had committed
    import os
    assert not os.path.exists(tmp_path / "out") or \
        not [n for n in os.listdir(tmp_path / "out")
             if n.startswith(("row", ".una-batch"))]


def test_rehydration_failure_is_loud_not_silent(make_batch, tmp_path,
                                                monkeypatch):
    """When the state-only rerun cannot rehydrate the final public state,
    the batch must fail with the typed error and the committed prefix
    attached — serial leaves live final state on the instance, so a silent
    success would deviate from serial public state (review MINOR-4)."""
    from dataclasses import replace

    from urban_network_analysis.batch import runtime as rt

    ckpt = str(tmp_path / "ckpt")
    out = tmp_path / "out"

    p = make_batch(2, out_folder=out)
    p.RunBatch("accessibility", parallel=True, workers=2,
               execution=replace(p.execution, checkpoint=ckpt))

    monkeypatch.setattr(rt, "_state_only_rerun", lambda *a, **k: None)

    p2 = make_batch(2, out_folder=out)
    with pytest.raises(BatchCancelledError,
                       match="could not be rehydrated") as ei:
        p2.RunBatch("accessibility", parallel=True, workers=2,
                    execution=replace(p2.execution, checkpoint=ckpt))
    assert p2.batch_report is None
    # the committed prefix (both reused rows) is attached for recovery
    assert [r.index for r in ei.value.batch_rows] == [0, 1]
    assert all(r.phase == "COMMITTED" for r in ei.value.batch_rows)


def test_checkpoint_identity_binds_the_transport_shim(make_batch, tmp_path,
                                                      monkeypatch):
    """The spawn shim lives outside the package dir the code identity
    walks; it must still be part of the checkpoint identity (review
    MAJOR-1 secondary gap)."""
    from dataclasses import replace

    import urban_network_analysis.batch.runtime as rt

    ckpt = str(tmp_path / "ckpt")
    p = make_batch(2, out_folder=tmp_path / "out")
    p.RunBatch("accessibility", parallel=True, workers=2,
               execution=replace(p.execution, checkpoint=ckpt))

    # only run 2 sees a changed shim (patching both runs would shift both
    # identities together and reject nothing)
    orig = rt._transport_identity

    def shimmed():
        d = dict(orig())
        d["spawn_shim"] = "0" * 64          # a different shim
        return d

    monkeypatch.setattr(rt, "_transport_identity", shimmed)

    p2 = make_batch(2, out_folder=tmp_path / "out")
    with pytest.raises(BatchCheckpointError, match="identity mismatch"):
        p2.RunBatch("accessibility", parallel=True, workers=2,
                    execution=replace(p2.execution, checkpoint=ckpt))
