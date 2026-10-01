"""The formal observable-prefix invariant, as an executable specification.

A deterministic simulator applies per-row effects two ways — the serial
reference and a plan-driven coordinator that may compute rows speculatively
but must COMMIT them in the plan's commit order — and compares observable
state after EVERY commit.  The dossier's two mutants live here: publishing
out of order, and folding composites by completion order, must both be
selected (fail) against this specification.
"""
from __future__ import annotations

import pytest

from urban_network_analysis.batch import (InvariantViolation, plan_batch,
                                          verify_prefix)


# ---------------------------------------------------------------------------
# Deterministic simulation harness.  Row effects are pure functions of
# (row name, the observable state at application time) so that a wrong
# schedule produces observably different state — the property the invariant
# exists to catch.
# ---------------------------------------------------------------------------

def _effect(name, state):
    """One row's observable effect: appends its own marker and, when a
    dependency's marker is present, folds it in (order-sensitively)."""
    log = list(state["log"])
    acc = state["acc"]
    log.append(name)
    dep = state.get(f"dep:{name}")
    if dep is not None:
        acc = acc + f"<{dep}"
    return {"log": tuple(log), "acc": acc + name}


def _serial_reference(names, deps):
    state = {"log": (), "acc": ""}
    snapshots = []
    for n in names:
        state = dict(state)
        for _dname, dval in (deps.get(n) or {}).items():
            state[f"dep:{n}"] = dval         # serial: it ran earlier
        state = _effect(n, state)
        snapshots.append(_freeze(state))
    return snapshots


def _freeze(state):
    return (state["log"], state["acc"])


def _run_coordinator(plan, names, deps, commit_order=None):
    """Coordinator simulation: workers produce each row's result privately;
    the coordinator applies the row's observable effect to the public state
    at COMMIT time (dossier 04 steps 6-7), folding in a dependency only if
    that dependency has already committed.  Returns the observable snapshot
    after every commit."""
    commit_order = commit_order or plan.commit_order
    state = {"log": (), "acc": ""}
    committed = set()
    snapshots = []
    for idx in commit_order:                 # commit_order holds row indices
        n = names[idx]
        state = dict(state)
        for _dname, dval in (deps.get(n) or {}).items():
            if dval in committed:            # dependency already committed
                state[f"dep:{n}"] = dval
        state = _effect(n, state)
        committed.add(n)
        snapshots.append(_freeze(state))
    return snapshots


def _dependent_batch(make_settings, tmp_path):
    """Two rows where row 1 consumes row 0's published artifact — the plan
    must serialize them, and the simulator folds the dependency."""
    data0 = tmp_path / "d0"
    rows = [
        make_settings("producer", data_folder=data0,
                      inputs=("net.geojson", "o.geojson", "d.geojson"),
                      stem="flow_out", folder=data0 / "Results"),
        make_settings("consumer", data_folder=data0 / "Results",
                      inputs=("flow_out.csv", "o.geojson", "d.geojson"),
                      stem="downstream", folder=tmp_path / "out_c"),
    ]
    deps = {"consumer": {"producer": "producer"}}
    return rows, deps


@pytest.mark.batch_plan
def test_invariant_holds_after_every_commit(make_settings, tmp_path):
    rows, deps = _dependent_batch(make_settings, tmp_path)
    plan = plan_batch(rows)
    assert plan.commit_order == (0, 1)      # serialized, caller order

    serial = _serial_reference(["producer", "consumer"], deps)
    committed = _run_coordinator(plan, ["producer", "consumer"], deps)
    # after EVERY committed row — not just at batch completion
    for k in range(1, len(committed) + 1):
        verify_prefix(committed[:k], serial[:k])       # raises on mismatch


@pytest.mark.batch_plan
def test_mutant_publishing_out_of_order_is_selected(make_settings, tmp_path):
    """Dossier 04 mutant: 'intentionally publish out of order'.  A
    coordinator that commits by COMPLETION order (the consumer finished
    first) violates the prefix invariant and must fail this spec."""
    rows, deps = _dependent_batch(make_settings, tmp_path)
    plan = plan_batch(rows)
    serial = _serial_reference(["producer", "consumer"], deps)
    out_of_order = _run_coordinator(
        plan, ["producer", "consumer"], deps, commit_order=(1, 0))
    with pytest.raises(InvariantViolation) as ei:
        verify_prefix(out_of_order, serial)
    assert ei.value.row == 0               # earliest differing prefix


@pytest.mark.batch_plan
def test_mutant_independent_rows_published_out_of_order_still_violates(
        make_settings, tmp_path):
    """Even with NO dependencies, commit order is observable (the row log):
    committing in completion order changes public state — the invariant is
    what forbids it, for independent rows too."""
    rows = [make_settings("a", folder=tmp_path / "oa"),
            make_settings("b", folder=tmp_path / "ob",
                          inputs=("nb.geojson", "ob.geojson", "db.geojson"))]
    plan = plan_batch(rows)
    assert plan.commit_order == (0, 1)
    serial = _serial_reference(["a", "b"], {})
    reordered = _run_coordinator(plan, ["a", "b"], {}, commit_order=(1, 0))
    with pytest.raises(InvariantViolation):
        verify_prefix(reordered, serial)


# ---------------------------------------------------------------------------
# Composite fold order: capture/fold in caller order is numerically
# load-bearing (float addition is not associative) — the second dossier
# mutant folds by completion order and must produce different bits.
# ---------------------------------------------------------------------------

def _fold(values, order):
    acc = 0.0
    for i in order:
        acc = acc + values[i]
    return acc


@pytest.mark.batch_plan
def test_composite_folds_in_caller_order(make_settings, tmp_path):
    plan = plan_batch([make_settings(f"r{i}", folder=tmp_path / f"o{i}")
                       for i in range(3)])
    assert plan.fold_order == (0, 1, 2) == plan.commit_order


@pytest.mark.batch_plan
def test_mutant_folding_by_completion_order_is_selected():
    """Dossier 04 mutant: 'sum composites by completion order'.  With
    values a=1e16, b=-1e16, c=1.0: caller-order folding gives
    ((1e16 + -1e16) + 1.0) == 1.0, while the completion order
    (c first) gives ((1.0 + 1e16) + -1e16) == 0.0 — different bits, so the
    plan's fold_order discipline is what keeps composites bitwise-stable."""
    values = {0: 1e16, 1: -1e16, 2: 1.0}
    caller_order = (0, 1, 2)
    completion_order = (2, 0, 1)          # row 2 finished first
    assert _fold(values, caller_order) == 1.0
    assert _fold(values, completion_order) == 0.0
    assert _fold(values, caller_order) != _fold(values, completion_order)


@pytest.mark.batch_plan
def test_verify_prefix_rejects_length_mismatch():
    serial = [("log", "acc")]
    with pytest.raises(InvariantViolation, match="length mismatch"):
        verify_prefix([], serial)
    with pytest.raises(InvariantViolation, match="length mismatch"):
        verify_prefix(serial, [])
