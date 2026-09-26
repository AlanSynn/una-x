"""F1 compiler census (proof 10, obligations 9.1/9.3): T7 — full-module
njit targetoptions census across arms (the ONLY delta permitted is
nogil=True on _accumulate_od_flow) and specialization-set equality
after an identical call battery, with warm-up wall time recorded."""
from __future__ import annotations

import sys
import time

import fixtures_f1
from harness_f1 import (b0ns, cns, flush_artifacts, record_census,
                        record_warmup)


B0_MOD = "urban_network_analysis.Engines.AggregateFlow"
CAND_MOD = "una_f1_cand.Engines.AggregateFlow"


def _dispatchers(mod):
    """name -> targetoptions dict for every njit dispatcher in a
    module (detection via numba's Dispatcher attributes)."""
    out = {}
    for name, obj in sorted(vars(mod).items()):
        if hasattr(obj, "targetoptions") and hasattr(obj, "signatures") \
                and hasattr(obj, "py_func"):
            out[name] = dict(obj.targetoptions)
    return out


def test_t7_targetoptions_census():
    """Candidate module census vs B0 module census: identical name set;
    identical targetoptions everywhere EXCEPT _accumulate_od_flow,
    which must add exactly nogil=True."""
    b0ns()  # ensure both arms are imported before the census
    cns()
    b0_disp = _dispatchers(sys.modules[B0_MOD])
    c_disp = _dispatchers(sys.modules[CAND_MOD])
    assert set(b0_disp) == set(c_disp), \
        f"dispatcher name sets differ: {set(b0_disp) ^ set(c_disp)}"
    deltas = {}
    for name in sorted(b0_disp):
        opts_b = b0_disp[name]
        opts_c = c_disp[name]
        keys_b = {k: opts_b.get(k, "<absent>") for k in
                  set(opts_b) | set(opts_c)}
        keys_c = {k: opts_c.get(k, "<absent>") for k in
                  set(opts_b) | set(opts_c)}
        diff = {k for k in keys_b if keys_b[k] != keys_c[k]}
        if name == "_accumulate_od_flow":
            assert diff == {"nogil"}, \
                f"unexpected option deltas on the pinned kernel: {diff}"
            assert keys_c["nogil"] is True
            assert keys_b["nogil"] in ("<absent>", False), \
                f"B0 nogil unexpectedly set: {keys_b['nogil']!r}"
            deltas[name] = {"b0": keys_b, "cand": keys_c}
        else:
            assert not diff, \
                f"unexpected targetoptions delta on {name}: {diff}"
            deltas[name] = {"b0": keys_b, "cand": keys_c}
    record_census("targetoptions", deltas)
    flush_artifacts()


def test_t7_specialization_sets_equal():
    """After driving an IDENTICAL micro battery through both arms (same
    case order, same argument types), the compiled specialization lists
    must be equal; warm-up wall time of the first specialization per
    arm is recorded (no performance claim)."""
    b0 = b0ns()
    c = cns()
    cases = fixtures_f1.micro_cases()

    t0 = time.perf_counter()
    cases[0].call(b0._accumulate_od_flow)
    record_warmup("specialization_warmup/b0/_accumulate_od_flow",
                  time.perf_counter() - t0)
    t0 = time.perf_counter()
    cases[0].call(c._accumulate_od_flow)
    record_warmup("specialization_warmup/cand/_accumulate_od_flow",
                  time.perf_counter() - t0)
    for case in cases[1:]:
        case.call(b0._accumulate_od_flow)
        case.call(c._accumulate_od_flow)

    sigs_b = [str(s) for s in b0._accumulate_od_flow.signatures]
    sigs_c = [str(s) for s in c._accumulate_od_flow.signatures]
    assert sigs_b == sigs_c, \
        f"specialization lists diverged:\n b0={sigs_b}\n cand={sigs_c}"
    # the battery is uniform-dtype: exactly ONE specialization so far
    assert len(set(sigs_b)) == 1, \
        f"unexpected specialization count: {sigs_b}"
    record_census("specializations", {
        "_accumulate_od_flow": {"signatures": sigs_b}})
    flush_artifacts()
