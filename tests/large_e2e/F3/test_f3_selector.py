"""F3 selector contract battery (proof sections 3 / 10).

Covers: contract bounds (1 <= c <= remaining; floor semantics; NO_FIT
signal, never 0, never a negative floor result); the NEGATIVE-BUDGET
case (h04 N2, driven two ways); determinism; purity (module source
audit + forbidden-callable interposition — no environment query); and
the max-transient theorem as a seeded property test (proof section 3:
retained + c*per_source + margin <= cap by integer-floor construction).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import fixtures_f3
from harness_f3 import cns, record_note


def expected_c(v_prime, remaining, dist_it, pred_it, mask_it,
               fixed_live, retained, row_temp, src_cost, margin, cap):
    """Independent statement of the section 3 contract."""
    per_source = (v_prime * (dist_it + pred_it + mask_it)
                  + row_temp + src_cost)
    budget = cap - fixed_live - retained - margin
    if remaining < 1 or budget < per_source:
        return -1  # NO_FIT
    return min(remaining, budget // per_source)


def _args(inp, lfws, *, remaining, retained=0, cap=None, margin=None,
          fixed_live=None):
    return (inp["v_prime"], remaining, inp["dist_itemsize"],
            inp["pred_itemsize"], inp["mask_itemsize"],
            inp["fixed_live_base"] if fixed_live is None else fixed_live,
            retained, inp["row_temporaries"], inp["source_index_cost"],
            lfws.MARGIN_BYTES if margin is None else margin,
            lfws.DEFAULT_CAP_BYTES if cap is None else cap)


def test_contract_bounds_and_floor():
    """c in [1, remaining]; floor semantics; never 0; never negative."""
    lfws = cns().lfws
    inp = fixtures_f3.selector_inputs(fixtures_f3.case_stub(
        fixtures_f3.base_case()))
    ps = fixtures_f3.per_source(inp)
    # budget exactly k*per_source -> c = min(remaining, k)
    for k in (1, 2, 3, 7, 1000):
        cap = inp["fixed_live_base"] + lfws.MARGIN_BYTES + k * ps
        got = lfws.select_chunk(*_args(inp, lfws, remaining=5, cap=cap))
        assert got == min(5, k), f"floor contract at k={k}: got {got}"
    # budget = k*per_source - 1 -> floor gives k-1 (k >= 2)
    cap = inp["fixed_live_base"] + lfws.MARGIN_BYTES + 3 * ps - 1
    got = lfws.select_chunk(*_args(inp, lfws, remaining=100, cap=cap))
    assert got == 2, f"floor semantics at budget=3ps-1: got {got}"
    # never 0: budget just below per_source is NO_FIT, not 0
    cap = inp["fixed_live_base"] + lfws.MARGIN_BYTES + ps - 1
    got = lfws.select_chunk(*_args(inp, lfws, remaining=5, cap=cap))
    assert got == lfws.NO_FIT and got != 0, f"near-fit must be NO_FIT: {got}"
    # c <= remaining even with a huge cap
    got = lfws.select_chunk(*_args(inp, lfws, remaining=4,
                                   cap=2**62))
    assert got == 4
    # remaining < 1 -> NO_FIT (contract total on the input domain)
    got = lfws.select_chunk(*_args(inp, lfws, remaining=0, cap=2**62))
    assert got == lfws.NO_FIT


def test_negative_budget_no_fit_two_ways():
    """h04 N2: budget_k is signed arithmetic and can go negative; the
    NO_FIT branch dominates and no c_k — not even a negative Python
    floor result — escapes.  Driven two ways."""
    lfws = cns().lfws
    inp = fixtures_f3.selector_inputs(fixtures_f3.case_stub(
        fixtures_f3.base_case()))
    ps = fixtures_f3.per_source(inp)
    # Way 1: cap < fixed_live + margin at m_k = 0 (negative budget).
    cap = inp["fixed_live_base"] + lfws.MARGIN_BYTES - 1
    assert cap - inp["fixed_live_base"] - 0 - lfws.MARGIN_BYTES < 0
    got = lfws.select_chunk(*_args(inp, lfws, remaining=5, cap=cap))
    assert got == lfws.NO_FIT, f"negative budget (way 1) must be NO_FIT: {got}"
    # Way 2: retained parts accumulate past cap - fixed - margin.
    cap = inp["fixed_live_base"] + lfws.MARGIN_BYTES + 2 * ps
    retained = 3 * ps          # pushes budget to -ps (negative)
    got = lfws.select_chunk(*_args(inp, lfws, remaining=5,
                                   retained=retained, cap=cap))
    assert got == lfws.NO_FIT, f"negative budget (way 2) must be NO_FIT: {got}"
    # Deeply negative budget (order of the cap itself) — same verdict.
    got = lfws.select_chunk(*_args(inp, lfws, remaining=5, retained=10**12,
                                   cap=cap))
    assert got == lfws.NO_FIT
    record_note("negative_budget", {
        "way1_cap": int(cap), "way2_retained": int(retained),
        "per_source": int(ps), "verdict": "NO_FIT both ways",
    })


def test_determinism():
    """Same integers -> same c, 1000 repetitions across a spread of
    inputs (dossier line 13: record the actual schedule; identical
    inputs must yield the identical schedule)."""
    lfws = cns().lfws
    rng = np.random.default_rng(20260927)
    for _ in range(200):
        v = int(rng.integers(1, 10**6))
        rem = int(rng.integers(1, 5000))
        its = (int(rng.integers(1, 16)), int(rng.integers(1, 16)),
               int(rng.integers(1, 16)))
        live = int(rng.integers(0, 2**30))
        ret = int(rng.integers(0, 2**30))
        rowt = int(rng.integers(0, 2**24))
        sic = int(rng.integers(0, 64))
        marg = int(rng.integers(0, 2**28))
        cap = int(rng.integers(0, 2**31))
        first = lfws.select_chunk(v, rem, *its, live, ret, rowt, sic,
                                  marg, cap)
        for _ in range(4):
            assert lfws.select_chunk(v, rem, *its, live, ret, rowt,
                                     sic, marg, cap) == first
        assert first == expected_c(v, rem, *its, live, ret, rowt, sic,
                                   marg, cap) or first == lfws.NO_FIT


def test_max_transient_theorem_property():
    """Whenever a width c is returned (not NO_FIT):
    retained + c*per_source + margin <= cap — by integer-floor
    construction, at EVERY selection (proof section 3)."""
    lfws = cns().lfws
    rng = np.random.default_rng(31415926)
    checked = 0
    for _ in range(500):
        v = int(rng.integers(1, 10**6))
        rem = int(rng.integers(1, 3000))
        its = (int(rng.integers(1, 16)), int(rng.integers(1, 16)),
               int(rng.integers(1, 16)))
        live = int(rng.integers(0, 2**28))
        ret = int(rng.integers(0, 2**28))
        rowt = int(rng.integers(0, 2**22))
        sic = int(rng.integers(0, 64))
        marg = int(rng.integers(1, 2**28))
        cap = int(rng.integers(0, 2**31))
        ps = v * sum(its) + rowt + sic
        c = lfws.select_chunk(v, rem, *its, live, ret, rowt, sic, marg, cap)
        if c != lfws.NO_FIT:
            assert c >= 1
            assert ret + c * ps + marg <= cap, (
                f"theorem violated: v={v} its={its} live={live} "
                f"ret={ret} marg={marg} cap={cap} c={c}")
            checked += 1
    assert checked > 0
    record_note("max_transient_property", {"returns_checked": checked})


def test_purity_source_audit():
    """Module CODE audit (AST-level, immune to docstring prose that
    names the prohibited things): no import statements, and no
    environment/I/O identifier anywhere in the module's code (spec
    C2)."""
    import ast
    lfws = cns().lfws
    tree = ast.parse(inspect.getsource(lfws))
    imports = [n for n in ast.walk(tree)
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not imports, f"import statements in selector module: {imports}"
    forbidden = {"psutil", "getloadavg", "virtual_memory", "environ",
                 "getrusage", "socket", "urandom", "getrandom", "seed",
                 "open", "system", "popen", "statvfs", "sysconf"}
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
    hits = sorted(names & forbidden)
    assert not hits, f"forbidden surface in selector code: {hits}"


def test_purity_interposition(monkeypatch):
    """Forbidden-callable interposition: plausible environment probes
    are patched to raise; the selector's results are unchanged, so its
    execution path cannot touch them."""
    import os

    lfws = cns().lfws
    inp = fixtures_f3.selector_inputs(fixtures_f3.case_stub(
        fixtures_f3.base_case()))
    ps = fixtures_f3.per_source(inp)
    cap = inp["fixed_live_base"] + lfws.MARGIN_BYTES + 2 * ps

    def _boom(*a, **k):
        raise AssertionError("environment probe called from selector path")

    monkeypatch.setattr(os, "getloadavg", _boom, raising=False)
    monkeypatch.setattr(os, "getrusage", _boom, raising=False)
    monkeypatch.delenv("MALLOC", raising=False)
    before = lfws.select_chunk(*_args(inp, lfws, remaining=9, cap=cap))
    after = lfws.select_chunk(*_args(inp, lfws, remaining=9, cap=cap))
    assert before == after == 2
