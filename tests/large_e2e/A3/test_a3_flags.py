"""A3 static pins (proof.md section 7, T9): decorator flags pinned to
the module consts, guard signature per review pin P2, and cache-backed
compilation of the tail-free kernel in the dedicated A3 cache root.
"""
from __future__ import annotations

import ast
import glob
import os

import fixtures_a3
from harness_a3 import cns, run_tailless

from _legacy_paths import LEGACY_WS  # portable-path fix (HARNESS 2026-09-30)
SCRATCH = str(LEGACY_WS) + "/wt-large-e2e/src/urban_network_analysis/Engines/_large_access_scratch.py"
CACHE_ROOT = os.environ.get(
    "UNA_A3_NUMBA_CACHE",
    str(LEGACY_WS) + "/campaign_data/nbc_a3")

KERNEL_DECORATOR = ("nb.njit(parallel=NUMBA_PARALLEL, cache=NUMBA_CACHE, "
                    "nogil=NUMBA_NOGIL, fastmath=NUMBA_FASTMATH)")
GUARD_PARAMS = ["d_terminal_idxs", "d_count", "node_count"]


def _module():
    return ast.parse(open(SCRATCH).read())


def test_kernel_decorator_identical_to_a1_and_guard_is_bare_njit():
    fns = {n.name: n for n in _module().body if isinstance(n, ast.FunctionDef)}
    a1 = [ast.unparse(d) for d in fns["_a1_scope_search"].decorator_list]
    a3 = [ast.unparse(d) for d in fns["_a3_scope_search_tailless"].decorator_list]
    assert a1 == a3 == [KERNEL_DECORATOR]
    guard = [ast.unparse(d) for d in fns["_a3_tail_admits"].decorator_list]
    assert guard == ["nb.njit"]


def test_guard_signature_matches_pin_p2():
    fns = {n.name: n for n in _module().body if isinstance(n, ast.FunctionDef)}
    params = [a.arg for a in fns["_a3_tail_admits"].args.args]
    assert params == GUARD_PARAMS
    a3_params = [a.arg for a in fns["_a3_scope_search_tailless"].args.args]
    assert "d_count" not in a3_params  # review pin P1


def test_module_consts_untouched():
    src = open(SCRATCH).read()
    for line in (
        "NUMBA_PARALLEL = False",
        "NUMBA_CACHE = True",
        "NUMBA_NOGIL = True",
        "NUMBA_FASTMATH = True",
    ):
        assert line in src, line


def test_tailless_kernel_is_cached_in_a3_root():
    """Trigger one call, then require a cache entry for the tail-free
    kernel under the dedicated A3 root (T9: cached kernels load
    without recompile in a fresh process — every other test module
    already ran through this cache)."""
    arrays = fixtures_a3.valid()
    run_tailless(cns(), fixtures_a3.kernel_case(arrays, 0))
    hits = glob.glob(os.path.join(CACHE_ROOT, "**", "*.nbc"), recursive=True)
    assert any("_a3_scope_search_tailless" in os.path.basename(p)
               for p in hits), hits[:20]
