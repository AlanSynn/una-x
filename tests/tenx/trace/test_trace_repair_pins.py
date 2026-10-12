"""C11 repair-round pins (campaign una_10x).

One test per C11 review finding repaired in round, against the
independent reviewer's own law:

- C11-MINOR-1: the producer kernel is numba-COMPILED (S11's exact
  decorator, module constants live) — pinned post-compile.
- C11-NOTE-1: no dead per-tile fold-output buffers — counters charge
  only the capture payload.
- C11-NOTE-2: tile_origins is clamped to origin_count; a huge
  tile_origins is the one-tile regime, bitwise and counters-identical
  to the equivalent width.
- C11-NOTE-3: the replay cutoff gate is typed- AND bit-exact
  (-0.0 is not +0.0).
"""
import numpy as np
import pytest

from conftest import (build, packed_equal, replay, result_equal,
                      ring_case)


@pytest.fixture
def built(tr):
    case = ring_case()
    return case, build(tr, case)


def test_producer_kernel_is_compiled(tr, sp, built):
    """C11-MINOR-1: the kernel compiled with S11's flags (one
    decorator, live NUMBA_* constants, dispatcher signatures exist
    after a real build)."""
    case, rec = built
    assert rec.complete
    kernel = tr._tr_epoch_scope_search_tile
    assert len(kernel.signatures) >= 1  # compiled, not pure Python
    # the module constants are the S11 values and are LIVE: they are
    # exactly the names the attached decorator binds
    assert tr.NUMBA_PARALLEL == sp.NUMBA_PARALLEL
    assert tr.NUMBA_CACHE == sp.NUMBA_CACHE
    assert tr.NUMBA_NOGIL == sp.NUMBA_NOGIL
    assert tr.NUMBA_FASTMATH == sp.NUMBA_FASTMATH
    # the generated kernel carries EXACTLY ONE decorator (source-level
    # check on the module's real bytes)
    import ast
    tree = ast.parse(open(tr.__file__, "rb").read().decode("utf-8"))
    kernels = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == "_tr_epoch_scope_search_tile"]
    assert len(kernels) == 1
    assert len(kernels[0].decorator_list) == 1


def test_no_dead_tile_buffers_in_source_and_counters(tr, built):
    """C11-NOTE-1: the ONLY tile_* locals in the whole module are the
    capture payload and its bookkeeping — no fold-output buffer under
    any name exists (positive inventory, not a name blacklist)."""
    import ast
    tree = ast.parse(open(tr.__file__, "rb").read().decode("utf-8"))
    targets = sorted({n.targets[0].id for n in ast.walk(tree)
                      if isinstance(n, ast.Assign)
                      and len(n.targets) == 1
                      and isinstance(n.targets[0], ast.Name)
                      and n.targets[0].id.startswith("tile_")})
    assert targets == ["tile_counts", "tile_dist", "tile_ids",
                       "tile_len", "tile_origins", "tile_temp",
                       "tile_temp_peak"], targets
    case, rec = built
    assert rec.counters.tile_temp_bytes > 0
    capture = (rec.counters.tile_origins * case["d"].shape[0]
               * 16)  # int64 ids + float64 distances
    assert rec.counters.tile_temp_bytes >= capture


def test_tile_origins_clamped_to_origin_count(tr, built):
    """C11-NOTE-2: tile_origins beyond origin_count IS the one-tile
    regime — bitwise-identical trace, identical counters, and the
    recorded width is the clamped value."""
    case = ring_case()
    rec_w = build(tr, case)                     # default 64 -> clamp 40
    args = [case[k] for k in ("o", "ow", "ptr", "vec", "wts", "flag",
                              "d", "dw", "dwt", "cutoff")]
    rec_huge = tr.build_trace(*args, tile_origins=10_000)
    rec_seven = tr.build_trace(*args, tile_origins=7)
    assert rec_huge.complete and rec_w.complete
    assert rec_huge.counters.tile_origins == 40
    assert packed_equal(rec_huge, rec_w)
    assert rec_huge.trace_id == rec_w.trace_id
    assert rec_huge.counters.tiles == rec_w.counters.tiles == 1
    assert rec_huge.counters.tile_temp_bytes == \
        rec_w.counters.tile_temp_bytes
    # partition independence still holds against a 7-tile split
    assert rec_seven.trace_id == rec_w.trace_id
    assert packed_equal(rec_seven, rec_w)


def test_replay_cutoff_gate_is_bit_exact(tr, built):
    """C11-NOTE-3: -0.0 and +0.0 cutoffs are different trace
    identities at replay; each record replays ONLY its own bits."""
    zero = ring_case(cutoff=0.0)
    neg = ring_case(cutoff=-0.0)
    rec_zero = build(tr, zero)
    rec_neg = build(tr, neg)
    assert rec_zero.complete and rec_neg.complete
    ok = replay(tr, rec_zero, zero)
    assert ok.route == "trace_replay" and ok.folds_done == \
        rec_zero.origin_count
    ref = replay(tr, rec_zero, zero)
    assert result_equal(ok, ref)
    refused = replay(tr, rec_zero, zero, cutoff=-0.0)
    assert refused.route == "trace_replay_refused"
    assert refused.refusal == "cutoff_mismatch_with_trace_identity"
    refused2 = replay(tr, rec_neg, neg, cutoff=0.0)
    assert refused2.route == "trace_replay_refused"
    assert refused2.refusal == "cutoff_mismatch_with_trace_identity"
    ok_neg = replay(tr, rec_neg, neg)
    ref_neg = replay(tr, rec_neg, neg)
    assert result_equal(ok_neg, ref_neg)
    # the two records are distinct identities bit-for-bit on the
    # cutoff carrier (the +0.0/-0.0 pair differs in the id's cutoff
    # field too)
    assert np.float64(rec_zero.cutoff_value).tobytes() \
        != np.float64(rec_neg.cutoff_value).tobytes()
