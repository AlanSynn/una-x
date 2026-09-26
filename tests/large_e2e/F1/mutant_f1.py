"""M3 first-divergence mutant: the production _accumulate_od_flow with
its pass-5 (origin legs) and pass-6 (destination legs) blocks swapped.

The mutant source is extracted from the CANDIDATE tree's real
AggregateFlow.py at test time (never a golden source), textually
patched in exactly two ways — the decorator becomes test-local
(cache=False, no nogil) and the two leg-pass blocks exchange places —
then exec'd and compiled here. If the source layout drifts from the
expected markers the extraction raises loudly instead of producing a
wrong mutant.

The mutant is a pure FP-association mutation: both leg passes are
self-contained (each reads/writes only its own accumulator), so the
swap preserves the SET of per-edge writes but regroups the order of
+= applications on edges fed by both legs — exactly the class of
change the byte comparator must catch.
"""
from __future__ import annotations

import numpy as np
import numba as nb

import cand_load
from harness_f1 import cns

_CONSTS = "_DECAY_EQUAL       = 0"
_HELPER_DEC = "@nb.njit(cache=True, inline='always')\ndef _find_arc"
_KERNEL_DEC = ("@nb.njit(cache=True, fastmath=True, nogil=True)\n"
               "def _accumulate_od_flow")
_END = "    return acc_d[dest_virtual_node]\n"
_M5 = "    # ── 5. Origin legs"
_M6 = "    # ── 6. Destination legs"
_M7 = "    # ── 7. Node flow"

_MUTANT = None


def _extract_region(path):
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    start = src.index(_CONSTS)
    end = src.index(_END, start) + len(_END)
    region = src[start:end]
    assert region.count(_KERNEL_DEC) == 1, \
        "kernel decorator marker not unique in extracted region"
    assert region.count(_M5) == 1 and region.count(_M6) == 1 \
        and region.count(_M7) == 1, "pass-block markers not unique"
    return region


def _swap_leg_passes(region):
    head = region[:region.index(_M5)]
    block5 = region[region.index(_M5):region.index(_M6)]
    block6 = region[region.index(_M6):region.index(_M7)]
    tail = region[region.index(_M7):]
    swapped = head + block6 + block5 + tail
    # The swap must be a pure block exchange: same total length, same
    # line multiset.
    assert len(swapped) == len(region)
    assert sorted(swapped.splitlines()) == sorted(region.splitlines())
    return swapped


def build_mutant():
    """Compile (once) the pass-swapped mutant kernel from the
    candidate source; returns the njit dispatcher."""
    global _MUTANT
    if _MUTANT is not None:
        return _MUTANT
    src_path = cns().module_files["Engines.AggregateFlow"]
    region = _extract_region(src_path)
    helper_dec = "@nb.njit(cache=True, inline='always')"
    assert region.count(helper_dec) == 2, \
        "expected exactly two inline helper decorators in the region"
    mutant_src = region.replace(
        helper_dec, "@nb.njit(cache=False, inline='always')")
    mutant_src = mutant_src.replace(
        _KERNEL_DEC, "@nb.njit(cache=False, fastmath=True)\n"
        "def _accumulate_od_flow")
    mutant_src = _swap_leg_passes(mutant_src)
    g = {"np": np, "nb": nb}
    exec(compile(mutant_src, "<f1_m3_mutant>", "exec"), g)
    _MUTANT = g["_accumulate_od_flow"]
    return _MUTANT
