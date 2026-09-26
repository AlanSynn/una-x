"""F1 turns-exclusion empirical pin (proof 10; H04-N1): T8 — the
turns=True pipeline is untouched by the change: candidate-vs-B0
end-to-end byte equality at K=2, turns-kernel targetoptions census
equal (no nogil in either arm), and the candidate tree's total src diff
is exactly the single decorator line."""
from __future__ import annotations

import difflib
import sys

from harness_f1 import assert_engine_bytes, b0ns, cns, record_census
from b0_import import B0_ROOT

import fixtures_f1


B0_MOD = "urban_network_analysis.Engines.AggregateFlow"
CAND_MOD = "una_f1_cand.Engines.AggregateFlow"


def test_t8_turns_engine_byte_equal():
    """turns=True engine case at K=2 (executor path of the turns
    driver): end-to-end outputs byte-identical across arms."""
    eng_b, settings_b = fixtures_f1.build_engine(
        b0ns(), "base", 2, extra_overrides={"turns": True})
    eng_c, settings_c = fixtures_f1.build_engine(
        cns(), "base", 2, extra_overrides={"turns": True})
    eng_b.Centrality(settings_b)
    eng_c.Centrality(settings_c)
    assert_engine_bytes(fixtures_f1.engine_output_arrays(eng_c),
                        fixtures_f1.engine_output_arrays(eng_b),
                        "t8/turns_true/k2")


def test_t8_turns_kernel_targetoptions():
    """_accumulate_od_flow_turns targetoptions are IDENTICAL across
    arms (no nogil in either — the turns kernel is not part of the
    change surface)."""
    b0_disp = dict(vars(sys.modules[B0_MOD]))
    c_disp = dict(vars(sys.modules[CAND_MOD]))
    name = "_accumulate_od_flow_turns"
    assert name in b0_disp and name in c_disp, \
        "turns kernel missing from an arm's module"
    opts_b = dict(b0_disp[name].targetoptions)
    opts_c = dict(c_disp[name].targetoptions)
    assert opts_b == opts_c, \
        f"turns kernel targetoptions diverged: {opts_b} vs {opts_c}"
    assert not opts_c.get("nogil", False), \
        "turns kernel unexpectedly carries nogil"
    record_census("turns_kernel_targetoptions",
                  {name: {"b0": opts_b, "cand": opts_c}})


def test_t8_source_diff_single_decorator_line():
    """The candidate tree's AggregateFlow.py differs from B0's file in
    EXACTLY one line: the kernel decorator gaining nogil=True."""
    cand_path = cns().module_files["Engines.AggregateFlow"]
    b0_path = (f"{B0_ROOT}/urban_network_analysis/Engines/AggregateFlow")
    import os
    b0_path = os.path.join(B0_ROOT, "urban_network_analysis",
                           "Engines", "AggregateFlow.py")
    with open(cand_path, "r", encoding="utf-8") as fh:
        cand_lines = fh.read().splitlines()
    with open(b0_path, "r", encoding="utf-8") as fh:
        b0_lines = fh.read().splitlines()
    diff = [l[1:] for l in difflib.unified_diff(
        b0_lines, cand_lines, lineterm="", n=0)
        if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    assert len(diff) == 2, \
        f"expected exactly one removed + one added line, got {diff!r}"
    removed, added = diff
    assert removed == "@nb.njit(cache=True, fastmath=True)", \
        f"unexpected removed line: {removed!r}"
    assert added == "@nb.njit(cache=True, fastmath=True, nogil=True)", \
        f"unexpected added line: {added!r}"
    record_census("source_diff", {
        "b0_path": b0_path, "cand_path": cand_path,
        "removed": removed, "added": added})
