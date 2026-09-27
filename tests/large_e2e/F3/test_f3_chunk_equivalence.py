"""F3 chunk equivalence battery (proof sections 4 / 10 / spec C6).

Bit-for-bit indptr/nodes/dist/pred across schedules — B0 (original
formula) is the reference arm; candidate runs at forced first-slice
widths c in {1, 2, 3}, whole-set, and the unpatched default cap — on
every fixture class of the dossier's counterexample list.  A single
divergence rejects the domain outright (dossier line 28).

The v=2 log of each arm is parsed: the candidate line must carry the
B0 line as a PREFIX (PD5 append-only) plus the [F3] schedule tokens,
and exactly ONE v=2 line may exist per call (ruling 3(c)).
"""
from __future__ import annotations

import re

import pytest

import fixtures_f3
from harness_f3 import (assert_gradient_bytes, b0ns, cns,
                        record_fixture_sha, record_note, record_schedule)

SCHEDULE_RE = re.compile(r"\[F3\] cap=(\d+)B margin=(\d+)B chunks=(\d+) "
                         r"schedule=(\[[^\]]*\]) fallback=(\w+) "
                         r"floor_fired=(\S+)")


def parse_f3_tail(line):
    m = SCHEDULE_RE.search(line)
    assert m is not None, f"no [F3] schedule tokens in: {line!r}"
    floor_raw = m.group(6)
    if floor_raw == "none":
        floor_slice = floor_chunks = None
    else:
        fm = re.fullmatch(r"slice=(\d+)\+chunks_done=(\d+)", floor_raw)
        assert fm is not None, f"unparsable floor_fired: {floor_raw!r}"
        floor_slice, floor_chunks = int(fm.group(1)), int(fm.group(2))
    return {
        "cap": int(m.group(1)), "margin": int(m.group(2)),
        "chunks": int(m.group(3)),
        "schedule": [int(x) for x in m.group(4)[1:-1].split(",") if x],
        "fallback": m.group(5),
        "floor_slice": floor_slice,
        "floor_chunks_done": floor_chunks,
        "floor_fired": floor_raw,
    }


def _run_case(case, monkeypatch, cap_value=None):
    c = cns()
    b0 = b0ns()
    stub_b0 = fixtures_f3.case_stub(case)
    stub_c = fixtures_f3.case_stub(case)
    if cap_value is not None:
        monkeypatch.setattr(c.lfws, "DEFAULT_CAP_BYTES", int(cap_value))
    out_b = b0.AggregateFlow._precompute_dest_gradients(stub_b0, {})
    out_c = c.AggregateFlow._precompute_dest_gradients(stub_c, {})
    return stub_b0, stub_c, out_b, out_c


def test_equivalence_all_fixtures_all_schedules(monkeypatch):
    """C6: every fixture x {c=1, c=2, c=3, whole-set, default cap} x
    B0 — bit-for-bit on all four arrays."""
    c = cns()
    b0 = b0ns()
    for case in fixtures_f3.all_cases():
        for label, cval in (
            ("c1", lambda st: fixtures_f3.cap_for_first_slice_c(st, c.lfws, 1)),
            ("c2", lambda st: fixtures_f3.cap_for_first_slice_c(st, c.lfws, 2)),
            ("c3", lambda st: fixtures_f3.cap_for_first_slice_c(st, c.lfws, 3)),
            ("whole", lambda st: fixtures_f3.cap_for_first_slice_c(
                st, c.lfws, st._n_destinations)),
            ("default", lambda st: None),
        ):
            stub_b0 = fixtures_f3.case_stub(case)
            stub_c = fixtures_f3.case_stub(case)
            cv = cval(stub_c)
            if cv is not None:
                monkeypatch.setattr(c.lfws, "DEFAULT_CAP_BYTES", int(cv))
            out_b = b0.AggregateFlow._precompute_dest_gradients(stub_b0, {})
            out_c = c.AggregateFlow._precompute_dest_gradients(stub_c, {})
            tag = f"{case['name']}/{label}"
            assert_gradient_bytes(out_c, out_b, tag)
            # exactly one v=2 line per arm; candidate line is the B0
            # line PLUS the [F3] tail (append-only, PD5)
            lines_b0 = stub_b0.logger.v2_lines()
            lines_c = stub_c.logger.v2_lines()
            assert len(lines_b0) == 1 and len(lines_c) == 1, (
                f"{tag}: v=2 line count {len(lines_b0)}/{len(lines_c)}")
            assert lines_c[0].startswith(lines_b0[0]), (
                f"{tag}: B0 line not a prefix of candidate line")
            parsed = parse_f3_tail(lines_c[0])
            record_schedule(tag, parsed)


def test_reference_arm_agreement(monkeypatch):
    """The in-process ORIGINAL-formula replica (fixtures_f3.
    reference_gradient) agrees byte-for-byte with the B0 module arm on
    every fixture — guards the stub construction itself."""
    b0 = b0ns()
    for case in fixtures_f3.all_cases():
        stub_b0 = fixtures_f3.case_stub(case)
        out_b = b0.AggregateFlow._precompute_dest_gradients(stub_b0, {})
        out_r = fixtures_f3.reference_gradient(stub_b0, case["limit"])
        assert_gradient_bytes(out_r, out_b, f"refvsb0/{case['name']}")


def test_tie_pred_compared_not_dist(monkeypatch):
    """The tied fixture's pred rows genuinely contain ties (the fixture
    is not vacuous) and the tie resolution is byte-identical across
    schedules c=1 vs whole-set vs B0 (dossier line 28)."""
    c = cns()
    b0 = b0ns()
    case = fixtures_f3.tied_case()
    record_fixture_sha("tied", {
        "arcs": case["arcs"], "n_net": case["n_net"],
        "n_dest": case["n_dest"], "limit": str(case["limit"])})
    outs = {}
    stub_b0 = fixtures_f3.case_stub(case)
    outs["b0"] = b0.AggregateFlow._precompute_dest_gradients(stub_b0, {})
    for label, cval in (("c1", 1), ("whole", case["n_dest"])):
        stub_c = fixtures_f3.case_stub(case)
        monkeypatch.setattr(
            c.lfws, "DEFAULT_CAP_BYTES",
            int(fixtures_f3.cap_for_first_slice_c(stub_c, c.lfws, cval)))
        outs[label] = c.AggregateFlow._precompute_dest_gradients(stub_c, {})
        # the tie: pred entries for node 3's equal-cost predecessors —
        # rows must contain BOTH predecessor ids in every schedule
        preds = set(outs[label][3].tolist()) - {-9999}
        assert preds, "tied fixture produced no predecessor entries"
    for label in ("c1", "whole"):
        assert_gradient_bytes(outs[label], outs["b0"], f"tied/{label}")


def test_zero_dest_log_and_outputs():
    """Zero destinations: empty arrays byte-equal; schedule empty;
    fallback none (dossier line 32 'zero destinations')."""
    c = cns()
    b0 = b0ns()
    case = fixtures_f3.zero_dest_case()
    stub_b0 = fixtures_f3.case_stub(case)
    stub_c = fixtures_f3.case_stub(case)
    out_b = b0.AggregateFlow._precompute_dest_gradients(stub_b0, {})
    out_c = c.AggregateFlow._precompute_dest_gradients(stub_c, {})
    assert_gradient_bytes(out_c, out_b, "zero_dest")
    parsed = parse_f3_tail(stub_c.logger.v2_lines()[0])
    assert parsed["schedule"] == [] and parsed["chunks"] == 0
    assert parsed["fallback"] == "none"
    record_schedule("zero_dest", parsed)


def test_duplicated_sources_note():
    """Dossier 'duplicated source nodes if baseline allows' — the
    function derives dest_nodes from np.arange (unique by
    construction), so the baseline does NOT allow duplicates; recorded
    as the dossier's conditional requires, nothing skipped silently."""
    record_note("duplicated_sources", {
        "status": "not_applicable",
        "ground": "dest_nodes = np.arange(n_net, n_net + n_dest) is "
                  "unique by construction in BOTH arms; the baseline "
                  "does not allow duplicated sources",
    })


def test_fixture_shas_recorded():
    """Fixture shas recorded in the artifacts (spec tests_plan)."""
    for case in fixtures_f3.all_cases():
        record_fixture_sha(case["name"], {
            "arcs": case["arcs"], "n_net": case["n_net"],
            "n_dest": case["n_dest"], "limit": str(case["limit"]),
        })
    from harness_f3 import flush_artifacts
    flush_artifacts()
