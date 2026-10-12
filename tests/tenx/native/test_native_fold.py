"""N00 native-fold bit battery (campaign una_10x).

What the battery proves, on the campaign host:

1. The strict-IEEE C++17 fold (``Engines/_native/metric_fold.cpp``)
   builds and loads, and the loaded .so is byte-identical to the one
   the recorded map was measured against.
2. The 72-point (n, knn_decay, duplicates, seed) equality map between
   the delivered numba fastmath fold and the native fold reproduces
   live, bit-for-bit in both directions: every recorded E is
   bit-identical, every recorded D still diverges at ULP level.
3. The recorded structure of that map closes the admission question:
   divergence is data-dependent from n=3 upward with NO class-wise
   rule (kind, duplicates, trip count) that separates E from D, so
   under the campaign's bitwise-parity law no substitution domain can
   be admitted.  The production replay route therefore stays on the
   numba fold (engagement test below), and the native fold serves the
   battery/qualification scope only.
4. The bridge's typed laws: dtype/byteorder refusal, indptr endpoints,
   finite-scalar law, Phase-B slice-order refusal (rc==2), input
   immutability, fresh owned outputs, kept/knn semantics, NaN
   exclusion identical to the numpy filter, structure independence
   (a packed origin's outputs depend only on its own slice bytes),
   and call-to-call determinism.

The map rows are MEASURED evidence regenerated mechanically by
``_n00_map_emit.py`` (scratch, uncommitted per the P00 raw-evidence
law); ``_measured_map.py`` records the environment fingerprint so a
stale map cannot silently re-verify.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

import _measured_map as MM

from urban_network_analysis.Engines.Accessibility import (
    reach_gravity_knn_access)
from urban_network_analysis.Engines._native_fold import (
    NativeRefused, native_replay_fold, so_sha256)

REPO = Path(__file__).resolve().parents[3]

# The fold scalars of every measured row (fixed by the generator).
KNN_W = (1.0, 0.8, 0.6)


def make_data(n, dup, seed):
    """Mirror of the map generator's data law (identical op order)."""
    rng = np.random.default_rng(seed)
    d = np.round(rng.uniform(0.1, 9.0, n), 3)
    if dup:
        d[rng.integers(0, n, max(1, n // 3))] = d[0]
    w = np.round(rng.uniform(0.25, 2.0, n), 3)
    order = np.argsort(d, kind="stable")
    return d[order], w[order]


def native_case(d, w, kind):
    ids = np.arange(d.shape[0], dtype=np.int64)
    indptr = np.array([0, d.shape[0]], dtype=np.int64)
    knn_w = np.array(KNN_W)
    return native_replay_fold(
        indptr, ids, d, w, 10.0, 0.35, 0.5, 0.75, 1.2, knn_w, kind)


def numba_case(d, w, kind):
    knn_w = np.array(KNN_W)
    return reach_gravity_knn_access(
        d, w, 10.0, 0.35, 0.5, 0.75, 0.0, knn_w, 0.2, kind, 0.5, 1.2, 0.9)


def run_case(n, kind, dup, seed):
    d, w = make_data(n, dup, seed)
    return numba_case(d, w, kind), native_case(d, w, kind)


def flags(nb, nt):
    return "".join(
        "E" if np.float64(nb[i]).tobytes()
        == np.float64(nt[i][0]).tobytes() else "D"
        for i in range(4))


def test_so_is_the_measured_bits():
    assert so_sha256() == MM.FINGERPRINT["so_sha256"]
    import numba
    import platform
    assert numba.__version__ == MM.FINGERPRINT["numba"]
    assert (platform.processor() or platform.machine()) \
        == MM.FINGERPRINT["cpu"]


@pytest.mark.parametrize(
    "n, kind, dup, seed, expected",
    [pytest.param(*row, id="n%d-%s-dup%d-s%d" % row[:4]) for row in MM.MAP])
def test_measured_map_live_reproduction(n, kind, dup, seed, expected):
    nb, nt = run_case(n, kind, dup, seed)
    live = flags(nb, nt)
    assert live == expected, (
        f"map drift at n={n} kind={kind} dup={dup} seed={seed}: "
        f"recorded {expected}, live {live}")


def test_no_classwise_admission_domain():
    """The LAW the map records: D appears in every (kind, dup) class at
    n>=8 and even at n=3, so no trip-count floor or data-class rule
    admits the native fold under the bitwise law."""
    all_e = [r for r in MM.MAP if "D" not in r[4]]
    assert all(r[0] <= 8 for r in all_e), \
        f"unexpected all-equal rows at n>8: {all_e}"
    for kind in ("exponential", "logistic", "flat"):
        for dup in (0, 1):
            deep = [r for r in MM.MAP
                    if r[0] >= 8 and r[1] == kind and r[2] == dup]
            assert deep, (kind, dup)
            assert any("D" in r[4] for r in deep), \
                f"class (kind={kind}, dup={dup}) fully bit-equal at n>=8"
    tiny = [r for r in MM.MAP if r[0] == 3]
    assert any("D" in r[4] for r in tiny), "n=3 fully bit-equal"


def test_probe_case_bit_equality_is_accidental_not_lawful():
    """The n=3 probe case that motivated the port is bit-equal on all
    four outputs AT THE PROBE'S OWN SCALARS (beta=0.1, plateau=0,
    growth=1.0) -- and the map shows its neighbours diverge (at the
    battery's default scalars ge already splits by 1 ULP at n=3).
    Pinned so the equality class stays observable, never promoted to
    a rule."""
    d = np.array([1.0, 2.0, 3.0])
    w = np.array([0.5, 0.5, 0.5])
    ids = np.arange(3, dtype=np.int64)
    indptr = np.array([0, 3], dtype=np.int64)
    knn_w = np.ones(2)
    nb = reach_gravity_knn_access(
        d, w, 10.0, 0.1, 0.0, 0.0, 0.0, knn_w, 0.1,
        "exponential", 0.0, 1.0, 1.0)
    nt = native_replay_fold(indptr, ids, d, w, 10.0, 0.1, 0.0, 0.0,
                            1.0, knn_w, "exponential")
    assert nb[0] == 1.5
    for i in range(4):
        assert np.float64(nb[i]).tobytes() == np.float64(nt[i][0]).tobytes()


# ----------------------------------------------------------------------
# Typed bridge laws (independent of the map grid).
# ----------------------------------------------------------------------

def test_refuses_non_float64_distances():
    d, w = make_data(8, 0, 7)
    ids = np.arange(8, dtype=np.int64)
    indptr = np.array([0, 8], dtype=np.int64)
    with pytest.raises(NativeRefused, match="dist_all dtype"):
        native_replay_fold(
            indptr, ids, d.astype(np.float32), w, 10.0, 0.35, 0.5,
            0.75, 1.2, np.array(KNN_W), "flat")


def test_refuses_indptr_endpoints_mismatch():
    d, w = make_data(8, 0, 7)
    with pytest.raises(NativeRefused, match="endpoints"):
        native_replay_fold(
            np.array([0, 7], dtype=np.int64),
            np.arange(8, dtype=np.int64), d, w, 10.0, 0.35, 0.5,
            0.75, 1.2, np.array(KNN_W), "flat")


def test_refuses_nonfinite_scalar():
    d, w = make_data(8, 0, 7)
    with pytest.raises(NativeRefused, match="finite"):
        native_replay_fold(
            np.array([0, 8], dtype=np.int64),
            np.arange(8, dtype=np.int64), d, w, float("nan"),
            0.35, 0.5, 0.75, 1.2, np.array(KNN_W), "flat")


def test_phase_b_refuses_nonascending_slice():
    """Descending ids inside a slice hit the Phase-B structural scan
    (rc==2 mapped to the typed refusal), not a silent wrong answer."""
    d = np.array([3.0, 2.0, 1.0])
    w = np.array([0.5, 0.5, 0.5])
    ids = np.array([2, 1, 0], dtype=np.int64)
    with pytest.raises(NativeRefused, match="slice-order corruption"):
        native_replay_fold(
            np.array([0, 3], dtype=np.int64), ids, d, w, 10.0,
            0.35, 0.5, 0.75, 1.2, np.array(KNN_W), "flat")


def test_inputs_immutable_and_outputs_owned():
    d, w = make_data(16, 0, 7)
    ids = np.arange(16, dtype=np.int64)
    indptr = np.array([0, 16], dtype=np.int64)
    knn_w = np.array(KNN_W)
    scalars = (10.0, 0.35, 0.5, 0.75, 1.2)
    arrays = (indptr, ids, d, w, knn_w)
    before = tuple(a.tobytes() for a in arrays)
    r1 = native_replay_fold(indptr, ids, d, w, *scalars, knn_w,
                            "exponential")
    after = tuple(a.tobytes() for a in arrays)
    assert before == after, "an input array was written through"
    for arr in r1:
        assert arr.base is None, "output aliases a cached buffer"
    # In-place damage to returned arrays cannot corrupt a rerun.
    r1[0][:] = -1.0
    r2 = native_replay_fold(indptr, ids, d, w, *scalars, knn_w,
                            "exponential")
    r3 = native_replay_fold(indptr, ids, d, w, *scalars, knn_w,
                            "exponential")
    for a, b in zip(r2, r3):
        assert a.tobytes() == b.tobytes()
    assert r2[0].tobytes() != r1[0].tobytes()


def test_kept_and_knn_semantics_multi_origin():
    d = np.array([1.0, 5.0, 20.0, 2.0, 3.0, 0.5, 9.0, 30.0, 4.0, 6.0])
    w = np.round(np.linspace(0.25, 2.0, 10), 3)
    ids = np.arange(10, dtype=np.int64)
    indptr = np.array([0, 5, 5, 10], dtype=np.int64)  # origin 1 empty
    knn_w = np.array(KNN_W)
    reach, ge, gl, kn, kept = native_replay_fold(
        indptr, ids, d, w, 10.0, 0.35, 0.5, 0.75, 1.2, knn_w, "flat")
    for i, (lo, hi) in enumerate(((0, 5), (5, 5), (5, 10))):
        mask = d[lo:hi] <= 10.0
        assert kept[i] == int(np.count_nonzero(mask)), (i, kept[i])
        # Reach folds the filtered slice in slice order (the C++
        # order); a strict Python sum reproduces it bit-for-bit,
        # numpy's pairwise reduction is NOT a lawful expectation.
        strict = 0.0
        for v in w[lo:hi][mask]:
            strict += float(v)
        assert np.float64(reach[i]).tobytes() \
            == np.float64(strict).tobytes()
    assert kn[1] == 0.0 and kept[1] == 0  # empty origin
    # Origin 0: sorted d = [1,2,3,5,20]; knn crops k=min(5,3)=3 ->
    # (d,w) pairs (1,w0),(2,w3),(3,w4) with flat coefficients 1/0.8/0.6.
    strict = 0.0
    for c, v in ((1.0, w[0]), (0.8, w[3]), (0.6, w[4])):
        strict += c * float(v)
    assert np.float64(kn[0]).tobytes() == np.float64(strict).tobytes()


def test_empty_structure_zeros():
    indptr = np.array([0, 0], dtype=np.int64)
    out = native_replay_fold(
        indptr, np.zeros(0, dtype=np.int64), np.zeros(0),
        np.zeros(0), 10.0, 0.35, 0.5, 0.75, 1.2, np.array(KNN_W),
        "exponential")
    for arr in out:
        assert arr.shape == (1,) and not arr.any()
    # numba fold on an empty distance array agrees bit-for-bit (+0.0)
    nb = numba_case(np.zeros(0), np.zeros(0), "exponential")
    assert all(np.float64(v).tobytes() == np.float64(0.0).tobytes()
               for v in nb)


def test_nan_distance_excluded_like_numpy():
    """NaN compares False against the cutoff, so the numpy filter drops
    it; each side must then fold EXACTLY its filtered subset (full ==
    subset bitwise within each side)."""
    d = np.array([1.0, np.nan, 3.0, 2.0])
    sub = np.array([1.0, 3.0, 2.0])
    w = np.array([0.5, 0.7, 0.9, 1.1])
    w_sub = np.array([0.5, 0.9, 1.1])
    ids = np.arange(4, dtype=np.int64)
    indptr = np.array([0, 4], dtype=np.int64)
    knn_w = np.array(KNN_W)
    nb_full = reach_gravity_knn_access(
        d, w, 10.0, 0.35, 0.5, 0.75, 0.0, knn_w, 0.2,
        "exponential", 0.5, 1.2, 0.9)
    nb_sub = reach_gravity_knn_access(
        sub, w_sub, 10.0, 0.35, 0.5, 0.75, 0.0, knn_w, 0.2,
        "exponential", 0.5, 1.2, 0.9)
    for a, b in zip(nb_full, nb_sub):
        assert np.float64(a).tobytes() == np.float64(b).tobytes()
    nt_full = native_replay_fold(
        indptr, ids, d, w, 10.0, 0.35, 0.5, 0.75, 1.2, knn_w,
        "exponential")
    nt_sub = native_replay_fold(
        np.array([0, 3], dtype=np.int64),
        np.arange(3, dtype=np.int64), sub, w_sub,
        10.0, 0.35, 0.5, 0.75, 1.2, knn_w, "exponential")
    for a, b in zip(nt_full[:4], nt_sub[:4]):
        assert np.float64(a[0]).tobytes() == np.float64(b[0]).tobytes()
    assert nt_full[4][0] == 3  # NaN is not kept


def test_structure_independence_of_packed_origins():
    """A packed origin's outputs depend only on its own slice bytes:
    fold origin 2 alone and inside a 3-origin structure -- identical."""
    d, w = make_data(31, 1, 7)
    ids_all = np.arange(31, dtype=np.int64)
    indptr = np.array([0, 4, 12, 31], dtype=np.int64)
    knn_w = np.array(KNN_W)
    packed = native_replay_fold(indptr, ids_all, d, w, 10.0, 0.35,
                                0.5, 0.75, 1.2, knn_w, "logistic")
    lo, hi = 12, 31
    d2 = np.ascontiguousarray(d[lo:hi])
    w2 = np.ascontiguousarray(w[lo:hi])
    ids2 = np.arange(hi - lo, dtype=np.int64)
    alone = native_replay_fold(
        np.array([0, hi - lo], dtype=np.int64), ids2, d2, w2,
        10.0, 0.35, 0.5, 0.75, 1.2, knn_w, "logistic")
    for a, b in zip(packed[:4], alone[:4]):
        assert np.float64(a[2]).tobytes() == np.float64(b[0]).tobytes()
    assert packed[4][2] == alone[4][0]


def test_call_to_call_determinism():
    d, w = make_data(31, 0, 7)
    nt1 = native_case(d, w, "exponential")
    nt2 = native_case(d, w, "exponential")
    for a, b in zip(nt1[:4], nt2[:4]):
        assert a.tobytes() == b.tobytes()
    nb1 = numba_case(d, w, "exponential")
    nb2 = numba_case(d, w, "exponential")
    assert tuple(np.float64(v).tobytes() for v in nb1) \
        == tuple(np.float64(v).tobytes() for v in nb2)


def test_production_route_never_imports_native():
    """ENGAGEMENT LAW: no production module references the native fold
    bridge; importing the delivery engines does not load it."""
    root = REPO / "src" / "urban_network_analysis"
    offenders = []
    for p in sorted(root.rglob("*.py")):
        if p.name == "_native_fold.py":
            continue
        if "_native_fold" in p.read_text(encoding="utf-8",
                                         errors="replace"):
            offenders.append(str(p.relative_to(root)))
    assert not offenders, f"production references to native fold: {offenders}"
    # Fresh interpreter: importing the delivery engines must not load
    # the bridge (this battery session itself has it imported, so the
    # engagement check is only meaningful in a clean process).
    import subprocess
    probe = (
        "import sys; sys.path.insert(0, 'src'); "
        "import urban_network_analysis.Engines.Accessibility; "
        "import urban_network_analysis.Engines._trace_access_scratch; "
        "assert 'urban_network_analysis.Engines._native_fold' "
        "not in sys.modules; print('CLEAN')")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True)
    assert r.returncode == 0 and "CLEAN" in r.stdout, (
        f"engagement probe failed: rc={r.returncode} {r.stderr[-400:]}")
