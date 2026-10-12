"""C31 closure repair pins (campaign una_10x).

- C31-MINOR-1: the residency sizing law is pinned AS ENFORCED —
  per-worker cap = ``max(64 MiB, allowance // workers)``; whenever the
  floor binds, the per-worker caps sum ABOVE the allowance (a
  documented exception with worst case ``64 MiB * workers``, not the
  earlier draft's false "sum stays inside the allowance").
- C31-NOTE-3: a directly-attached session refuses a turns-enabled
  ``o_access`` (counted) and the wrapper falls back to the inner
  engine — the UNA.py dispatch gate is not the only guard, because the
  trace identity asserts turn semantics "not-present-in-prepared-domain".
"""
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.tenx_cohort

MIB = 1 << 20


def test_sizing_floor_law_enforced(monkeypatch):
    """C31-MINOR-1: the floor is deliberate and its sum exception is
    pinned, not hidden.  4 GiB available -> allowance = 25% of
    0.60 * 4 GiB = 644245094 B (the review's sizing probe host)."""
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory",
        lambda: SimpleNamespace(available=4 * (1 << 30)))
    from urban_network_analysis.batch.cohort import (
        per_worker_residency_cap)

    allowance = int(0.25 * 0.60 * 4 * (1 << 30))
    # floor not binding: the caps stay inside the allowance
    assert per_worker_residency_cap(1) == allowance
    assert per_worker_residency_cap(4) == allowance // 4
    # floor binding: every worker gets exactly the 64 MiB floor ...
    assert per_worker_residency_cap(16) == 64 * MIB
    assert per_worker_residency_cap(64) == 64 * MIB
    # ... and the DOCUMENTED exception: the caps sum above the allowance
    assert 16 * (64 * MIB) == (1 << 30)
    assert 16 * (64 * MIB) > allowance          # 1.67x (review probe)
    assert 64 * (64 * MIB) == 4 * (1 << 30)     # 6.67x (review probe)


def test_directly_attached_session_refuses_turns(tmp_path):
    """C31-NOTE-3: a turns-enabled o_access refuses (counted) no matter
    how the session was attached, and the wrapper falls back to the
    inner engine's own answer."""
    from urban_network_analysis.batch.cohort import (
        CohortGraphEngine, CohortSession)

    session = CohortSession(str(tmp_path / ".una-batch-cohort"),
                            residency_cap_bytes=64 * MIB)
    settings = SimpleNamespace(turns=True)
    sentinel = object()

    class _Inner:
        def o_access(self, o_idx, settings, d_weights):
            return sentinel

        def o_scope(self, o_idx, search_radius):
            raise AssertionError("o_scope must not be reached")

    engine = CohortGraphEngine(_Inner(), None, session)
    out = engine.o_access(0, settings, None)
    assert out is sentinel                    # inner engine's own result
    assert session.counters["refusals"] == {
        "turns_present_in_prepared_domain": 1}
    assert session.counters["rows_attempted"] == 1
    assert session.last_trace_id is None
    assert session.last_result_key is None
