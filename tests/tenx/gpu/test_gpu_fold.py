"""G10 CPU-only laws for the GPU fold (campaign una_10x).

These tests run on any host and NEVER mark hardware validation: the
device battery lives in ``_device_battery.py`` and requires a real
device (G11).  What this island proves here: the engagement law, the
typed blocker (no simulated pass without hardware), the refusal
precedence, and the kernel-source region laws (no atomics, no
fastmath, one-thread-per-origin, total-order knn keys, strict-order
folds, exact-size thread-owned scratch).
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from urban_network_analysis.Engines import _gpu_fold as g

pytestmark = pytest.mark.tenx_gpu

_REPO = Path(__file__).resolve().parents[3]
_ENGINES = str(_REPO / "src" / "urban_network_analysis" / "Engines")


def _gpu_module_path():
    return g.__file__


# ------------------------------------------------------------- engagement

def test_production_route_never_references_gpu_module():
    """Static law: no delivery module references _gpu_fold."""
    hits = []
    for fn in sorted(os.listdir(_ENGINES)):
        if not fn.endswith(".py") or fn.startswith("_gpu"):
            continue
        src = open(os.path.join(_ENGINES, fn)).read()
        if "_gpu_fold" in src or "gpu_replay_fold" in src:
            hits.append(fn)
    assert hits == []


def test_fresh_interpreter_importing_delivery_engines_loads_no_gpu_module():
    """Dynamic law: importing the delivery engines must not load the
    GPU bridge (fresh process, so test-session imports cannot leak)."""
    probe = (
        "import sys; import urban_network_analysis.batch as b; "
        "import urban_network_analysis.Engines as e; "
        "mods = [m for m in sys.modules if m.endswith('_gpu_fold')]; "
        "print('LOADED' if mods else 'CLEAN')")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO / "src")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, check=True, env=env)
    assert r.stdout.strip().endswith("CLEAN"), r.stdout


# --------------------------------------------------------- typed blocker

def test_no_device_is_typed_blocker_not_simulated_pass():
    if g.cuda_available():  # device hosts exercise the battery instead
        pytest.skip("device present; blocker semantics need absence")
    with pytest.raises(g.GpuUnavailable) as ei:
        g.require_device()
    msg = str(ei.value)
    assert "blocker" in msg and "NOT passed" in msg


def test_gpu_replay_fold_raises_blocker_without_device():
    if g.cuda_available():
        pytest.skip("device present; battery owns device calls")
    ip, ids, dist, w = _lawful()
    with pytest.raises(g.GpuUnavailable):
        g.gpu_replay_fold(ip, ids, dist, w, 12.0, 0.35, 0.5, 0.75,
                          1.2, np.array([1.0, 0.8]), "exponential")


# ------------------------------------------------------------- refusals

def _lawful():
    """A lawful packed CSR: 3 origins, middle slice EMPTY, strictly
    ascending ids 0..6; node_count 8."""
    chunks = [3, 0, 4]
    ids = list(range(7))
    ip = np.zeros(len(chunks) + 1, dtype=np.int64)
    ip[1:] = np.cumsum(chunks, dtype=np.int64)
    rng = np.random.default_rng(5)
    return (ip, np.array(ids, dtype=np.int64),
            np.round(rng.uniform(0.1, 9.0, len(ids)), 4),
            np.round(rng.uniform(0.25, 2.0, 8), 3))


def test_refusals_precede_the_device_gate():
    if g.cuda_available():
        pytest.skip("device present; battery owns device calls")
    ip, ids, dist, w = _lawful()
    with pytest.raises(g.GpuRefused):
        g.gpu_replay_fold(ip.astype(np.int32), ids, dist, w, 12.0,
                          0.35, 0.5, 0.75, 1.2, np.array([1.0, 0.8]),
                          "exponential")
    with pytest.raises(g.GpuRefused):
        g.gpu_replay_fold(ip[:-1], ids, dist, w, 12.0, 0.35, 0.5,
                          0.75, 1.2, np.array([1.0, 0.8]), "exponential")
    with pytest.raises(g.GpuRefused):
        g.gpu_replay_fold(ip, ids, dist[:-1], w, 12.0, 0.35, 0.5,
                          0.75, 1.2, np.array([1.0, 0.8]), "exponential")
    with pytest.raises(g.GpuRefused):
        g.gpu_replay_fold(ip, ids, dist, w, float("nan"), 0.35, 0.5,
                          0.75, 1.2, np.array([1.0, 0.8]), "exponential")


def test_phase_b_slice_order_refusal_is_typed():
    ip, ids, dist, w = _lawful()
    bad = ids.copy()
    bad[0], bad[1] = bad[1], bad[0]  # breaks strict ascent in slice 0
    with pytest.raises(g.GpuRefused, match="slice-order"):
        g.gpu_replay_fold(ip, bad, dist, w, 12.0, 0.35, 0.5, 0.75,
                          1.2, np.array([1.0, 0.8]), "exponential")


def test_phase_b_range_refusal_is_typed():
    ip, ids, dist, w = _lawful()
    bad = ids.copy()
    # last element of the last (non-empty) slice: ascent still holds
    # inside every slice, but the id leaves [0, node_count)
    bad[ids.shape[0] - 1] = 10 ** 6
    with pytest.raises(g.GpuRefused, match="structural"):
        g.gpu_replay_fold(ip, bad, dist, w, 12.0, 0.35, 0.5, 0.75,
                          1.2, np.array([1.0, 0.8]), "exponential")


# ------------------------------------------------- kernel source laws

def _module_tree():
    return ast.parse(open(_gpu_module_path()).read())


def test_kernel_has_no_atomics_and_no_fastmath():
    tree = _module_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert "atomic" not in node.attr.lower(), ast.dump(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "jit":
            assert all(kw.arg != "fastmath" for kw in node.keywords), \
                "fastmath is forbidden by the region spec"


def test_kernel_is_one_thread_per_origin_with_bounds_guard():
    src = open(_gpu_module_path()).read()
    assert "_cuda.grid(1)" in src, "one-thread-per-origin mapping missing"
    assert "if o >= n_origins:" in src, "bounds guard missing"
    assert "s_reach += w" in src, "order-bearing reach fold missing"


def test_knn_keys_are_total_order_filtered_index():
    src = open(_gpu_module_path()).read()
    # the recorded FILTERED position (never the slice index) - the
    # exact N00-IF-2 law - and an explicit (d, i) tie comparison
    assert "pairs_i[lo + kept] = kept" in src
    assert "db == da and ib > ia" in src


def test_strict_branch_terms_present():
    src = open(_gpu_module_path()).read()
    assert "w / math.exp(beta * clipped)" in src
    assert "math.exp(-gr * (pos - mid))" in src
    assert "math.exp(-beta * clipped)" in src
    assert "math.exp(-gr * pos2)" in src


def test_scratch_is_exact_size_and_thread_owned():
    src = open(_gpu_module_path()).read()
    assert "device_array(nnz, dtype=np.float64)" in src
    assert "device_array(nnz, dtype=np.int64)" in src
    assert "device_array(origin_count * nnz" not in src


def test_sync_precedes_publication():
    src = open(_gpu_module_path()).read()
    sync_at = src.index("_cuda.synchronize()")
    copy_at = src.index("copy_to_host()")
    assert sync_at < copy_at, "results published before synchronize()"


# --------------------------------------------- hardware-validation law

def test_device_battery_requires_device():
    """The battery (G11) must hard-require a device: on CPU-only hosts
    it can print no result at all, let alone a passing one."""
    battery = os.path.normpath(os.path.join(
        os.path.dirname(_gpu_module_path()), "..", "..", "..",
        "tests", "tenx", "gpu", "_device_battery.py"))
    assert os.path.isfile(battery), battery
    src = open(battery).read()
    assert "require_device()" in src
    assert "GpuUnavailable" in src
