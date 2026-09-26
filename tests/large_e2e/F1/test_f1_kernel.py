"""F1 kernel-level battery (proof 10): T1 micro-case two-arm byte
equality, T13 kernel aliasing/lifetime pins, and the M3 first-divergence
negative (pass-5/pass-6 swapped mutant)."""
from __future__ import annotations

import time

import numpy as np

from comparator import assert_array_bytes_equal

import fixtures_f1
import mutant_f1
from harness_f1 import assert_kernel_bytes, cns, record_note, record_warmup
from support import b0


CASES = fixtures_f1.micro_cases()


def _b0_kernel():
    return b0()._accumulate_od_flow


def _cand_kernel():
    return cns()._accumulate_od_flow


def test_t1_kernel_battery_two_arm():
    """Compiled candidate kernel vs compiled B0 kernel on every micro
    case: (out_AB, out_BA, out_node_flow, delivered) byte equality."""
    b0_kernel = _b0_kernel()
    c_kernel = _cand_kernel()
    for i, case in enumerate(CASES):
        if i == 0:  # first compiled call of the session per arm
            t0 = time.perf_counter()
            out_b = case.call(b0_kernel)
            record_warmup(f"kernel_warmup/b0/{case.name}",
                          time.perf_counter() - t0)
            t0 = time.perf_counter()
            out_c = case.call(c_kernel)
            record_warmup(f"kernel_warmup/cand/{case.name}",
                          time.perf_counter() - t0)
        else:
            out_b = case.call(b0_kernel)
            out_c = case.call(c_kernel)
        assert_kernel_bytes(out_c, out_b, f"t1/{case.name}")


def test_t13_outputs_fresh_and_disjoint_from_inputs():
    """Kernel outputs are freshly owned buffers: no aliasing with any
    input array; the node-flow-off case carries a shape-0 buffer that
    is never written (empty bytes, byte-equal across arms)."""
    by_name = {c.name: c for c in CASES}
    case = by_name["m_base"]
    inputs = [case.indptr, case.indices, case.weights, case.edge_id,
              case.dir_of, case.d_o, case.dd_buf, case.pred_o,
              case.pd_buf]
    out_b = case.call(_b0_kernel())
    out_c = case.call(_cand_kernel())
    for tag, outs in (("b0", out_b), ("cand", out_c)):
        for oi in range(3):
            arr = np.asarray(outs[oi])
            assert arr.flags.owndata or arr.shape[0] == 0, \
                f"{tag}/out{oi} does not own its data"
            for ii, inp in enumerate(inputs):
                assert not np.shares_memory(arr, inp), \
                    f"{tag}/out{oi} aliases input {ii}"

    # Node-flow-off specialization: shape-0 out buffer, never written.
    case_nn = by_name["m_base_nonode"]
    nn_b = case_nn.call(_b0_kernel())
    nn_c = case_nn.call(_cand_kernel())
    assert nn_b[2].shape == (0,) and nn_c[2].shape == (0,)
    assert nn_c[2].tobytes() == nn_b[2].tobytes() == b""
    assert_array_bytes_equal(nn_c[0], nn_b[0], "t13/nonode/out_AB")
    assert_array_bytes_equal(nn_c[1], nn_b[1], "t13/nonode/out_BA")


def test_m3_pass_swap_mutant_diverges():
    """M3 negative: the pass-5/pass-6 swapped mutant must be CAUGHT by
    the byte comparator on at least one battery case (a negative test
    that fails to fire is itself a failure); per-case divergences are
    recorded with their first differing byte."""
    mutant = mutant_f1.build_mutant()
    prod = _cand_kernel()
    fired = []
    for case in CASES:
        out_p = case.call(prod)
        out_m = case.call(mutant)
        for name, pi, mi in (("out_AB", out_p[0], out_m[0]),
                             ("out_BA", out_p[1], out_m[1]),
                             ("out_node", out_p[2], out_m[2])):
            pb = np.asarray(pi).tobytes()
            mb = np.asarray(mi).tobytes()
            if pb != mb:
                idx = next(i for i, (x, y) in enumerate(zip(pb, mb))
                           if x != y)
                fired.append({"case": case.name, "array": name,
                              "first_diff_byte": idx})
                break
        if float(out_p[3]).hex() != float(out_m[3]).hex() and \
                not any(f["case"] == case.name for f in fired):
            fired.append({"case": case.name, "array": "delivered"})
    assert fired, ("M3 pass-swap mutant was NOT caught on any battery "
                   "case — comparator sensitivity failure")
    record_note("m3_first_divergences", {"mutations_caught": fired})
