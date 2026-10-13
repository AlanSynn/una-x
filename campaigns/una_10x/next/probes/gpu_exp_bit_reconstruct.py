"""Direction 1 road (a): device exp bitwise reconstruction vs the delivered host authority.

The delivered authority's exp is glibc 2.34 scalar exp (established in-probe:
numba njit fastmath=True == fastmath=False == math.exp == ctypes libm on the
compute node; np.exp is a DIFFERENT host implementation and is not the
authority).  This probe (1) characterizes the host bit pattern against exact
pure-Python simulations of the glibc algorithm (Nagy table+poly, N=128,
EXP_POLY_ORDER=5, TOINT_INTRINSICS=0, WANT_ROUNDING=1) under compilation
variants - plain (no fma) and fma-fused (x86-64 multiarch __exp_fma with
GCC -ffp-contract=fast, three fusion-order hypotheses) - to identify the
exact dispatched op order; (2) ports the algorithm to numba-CUDA kernels
(schemes: libdevice exp baseline; plain port under default flags plus an
explicit contraction detector; fastmath implicit-contract port; explicit
libdevice-fma ports in the same fusion orders) and bit-compares every
scheme against the authority over every argument group; (3) applies the
road-(a) gate: >=10^6 adversarial+random cases in the declared cohort
domain (|x| <= 32 plus the tiny path incl. 0.0); ANY bit divergence in the
gate domain = NO-GO for that scheme.  No CPU fallback path is touched
(CPU_ROUTE_RETAINED law); no tolerance is introduced anywhere - a bitwise
miss closes the exp-cell domain for that scheme.

Constants and the 256-entry table are taken verbatim from the glibc 2.34
sources (sysdeps/ieee754/dbl-64/e_exp.c and e_exp_data.c,
raw.githubusercontent.com/bminor/glibc release/2.34/master; the N==128
branch), parsed mechanically into the literal below, never hand-transcribed.

    PYTHONPATH=src python probes/gpu_exp_bit_reconstruct.py --out <out.json>
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import struct
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

SCHEMA = "una10x-gpu-exp-bitreconstruct-1"
SEED = 20261012
MASK64 = (1 << 64) - 1

# --- glibc 2.34 exp constants (e_exp.c / e_exp_data.c, N = 128) -------------
INV_LN2_N = float.fromhex("0x1.71547652b82fep0") * 128   # .invln2N = ... * N
NEG_LN2_HI_N = float.fromhex("-0x1.62e42fefa0000p-8")
NEG_LN2_LO_N = float.fromhex("-0x1.cf79abc9e3b3ap-47")
SHIFT = float.fromhex("0x1.8p52")
C2 = float.fromhex("0x1.ffffffffffdbdp-2")
C3 = float.fromhex("0x1.555555555543cp-3")
C4 = float.fromhex("0x1.55555cf172b91p-5")
C5 = float.fromhex("0x1.1111167a4d017p-7")
TWO_P1009 = float.fromhex("0x1p1009")
TWO_P1022M = float.fromhex("0x1p-1022")
NEG_INF_BITS = 0xFFF0000000000000
GLIBC_SRC = ("glibc release/2.34/master: sysdeps/ieee754/dbl-64/"
             "e_exp.c + e_exp_data.c (N=128, EXP_TABLE_BITS=7, "
             "EXP_POLY_ORDER=5, TOINT_INTRINSICS=0, WANT_ROUNDING=1); "
             "tab parsed mechanically from the raw source")

TAB = (
0x0,0x3ff0000000000000,0x3c9b3b4f1a88bf6e,0x3feff63da9fb3335,0xbc7160139cd8dc5d,0x3fefec9a3e778061,0xbc905e7a108766d1,0x3fefe315e86e7f85,
0x3c8cd2523567f613,0x3fefd9b0d3158574,0xbc8bce8023f98efa,0x3fefd06b29ddf6de,0x3c60f74e61e6c861,0x3fefc74518759bc8,0x3c90a3e45b33d399,0x3fefbe3ecac6f383,
0x3c979aa65d837b6d,0x3fefb5586cf9890f,0x3c8eb51a92fdeffc,0x3fefac922b7247f7,0x3c3ebe3d702f9cd1,0x3fefa3ec32d3d1a2,0xbc6a033489906e0b,0x3fef9b66affed31b,
0xbc9556522a2fbd0e,0x3fef9301d0125b51,0xbc5080ef8c4eea55,0x3fef8abdc06c31cc,0xbc91c923b9d5f416,0x3fef829aaea92de0,0x3c80d3e3e95c55af,0x3fef7a98c8a58e51,
0xbc801b15eaa59348,0x3fef72b83c7d517b,0xbc8f1ff055de323d,0x3fef6af9388c8dea,0x3c8b898c3f1353bf,0x3fef635beb6fcb75,0xbc96d99c7611eb26,0x3fef5be084045cd4,
0x3c9aecf73e3a2f60,0x3fef54873168b9aa,0xbc8fe782cb86389d,0x3fef4d5022fcd91d,0x3c8a6f4144a6c38d,0x3fef463b88628cd6,0x3c807a05b0e4047d,0x3fef3f49917ddc96,
0x3c968efde3a8a894,0x3fef387a6e756238,0x3c875e18f274487d,0x3fef31ce4fb2a63f,0x3c80472b981fe7f2,0x3fef2b4565e27cdd,0xbc96b87b3f71085e,0x3fef24dfe1f56381,
0x3c82f7e16d09ab31,0x3fef1e9df51fdee1,0xbc3d219b1a6fbffa,0x3fef187fd0dad990,0x3c8b3782720c0ab4,0x3fef1285a6e4030b,0x3c6e149289cecb8f,0x3fef0cafa93e2f56,
0x3c834d754db0abb6,0x3fef06fe0a31b715,0x3c864201e2ac744c,0x3fef0170fc4cd831,0x3c8fdd395dd3f84a,0x3feefc08b26416ff,0xbc86a3803b8e5b04,0x3feef6c55f929ff1,
0xbc924aedcc4b5068,0x3feef1a7373aa9cb,0xbc9907f81b512d8e,0x3feeecae6d05d866,0xbc71d1e83e9436d2,0x3feee7db34e59ff7,0xbc991919b3ce1b15,0x3feee32dc313a8e5,
0x3c859f48a72a4c6d,0x3feedea64c123422,0xbc9312607a28698a,0x3feeda4504ac801c,0xbc58a78f4817895b,0x3feed60a21f72e2a,0xbc7c2c9b67499a1b,0x3feed1f5d950a897,
0x3c4363ed60c2ac11,0x3feece086061892d,0x3c9666093b0664ef,0x3feeca41ed1d0057,0x3c6ecce1daa10379,0x3feec6a2b5c13cd0,0x3c93ff8e3f0f1230,0x3feec32af0d7d3de,
0x3c7690cebb7aafb0,0x3feebfdad5362a27,0x3c931dbdeb54e077,0x3feebcb299fddd0d,0xbc8f94340071a38e,0x3feeb9b2769d2ca7,0xbc87deccdc93a349,0x3feeb6daa2cf6642,
0xbc78dec6bd0f385f,0x3feeb42b569d4f82,0xbc861246ec7b5cf6,0x3feeb1a4ca5d920f,0x3c93350518fdd78e,0x3feeaf4736b527da,0x3c7b98b72f8a9b05,0x3feead12d497c7fd,
0x3c9063e1e21c5409,0x3feeab07dd485429,0x3c34c7855019c6ea,0x3feea9268a5946b7,0x3c9432e62b64c035,0x3feea76f15ad2148,0xbc8ce44a6199769f,0x3feea5e1b976dc09,
0xbc8c33c53bef4da8,0x3feea47eb03a5585,0xbc845378892be9ae,0x3feea34634ccc320,0xbc93cedd78565858,0x3feea23882552225,0x3c5710aa807e1964,0x3feea155d44ca973,
0xbc93b3efbf5e2228,0x3feea09e667f3bcd,0xbc6a12ad8734b982,0x3feea012750bdabf,0xbc6367efb86da9ee,0x3fee9fb23c651a2f,0xbc80dc3d54e08851,0x3fee9f7df9519484,
0xbc781f647e5a3ecf,0x3fee9f75e8ec5f74,0xbc86ee4ac08b7db0,0x3fee9f9a48a58174,0xbc8619321e55e68a,0x3fee9feb564267c9,0x3c909ccb5e09d4d3,0x3feea0694fde5d3f,
0xbc7b32dcb94da51d,0x3feea11473eb0187,0x3c94ecfd5467c06b,0x3feea1ed0130c132,0x3c65ebe1abd66c55,0x3feea2f336cf4e62,0xbc88a1c52fb3cf42,0x3feea427543e1a12,
0xbc9369b6f13b3734,0x3feea589994cce13,0xbc805e843a19ff1e,0x3feea71a4623c7ad,0xbc94d450d872576e,0x3feea8d99b4492ed,0x3c90ad675b0e8a00,0x3feeaac7d98a6699,
0x3c8db72fc1f0eab4,0x3feeace5422aa0db,0xbc65b6609cc5e7ff,0x3feeaf3216b5448c,0x3c7bf68359f35f44,0x3feeb1ae99157736,0xbc93091fa71e3d83,0x3feeb45b0b91ffc6,
0xbc5da9b88b6c1e29,0x3feeb737b0cdc5e5,0xbc6c23f97c90b959,0x3feeba44cbc8520f,0xbc92434322f4f9aa,0x3feebd829fde4e50,0xbc85ca6cd7668e4b,0x3feec0f170ca07ba,
0x3c71affc2b91ce27,0x3feec49182a3f090,0x3c6dd235e10a73bb,0x3feec86319e32323,0xbc87c50422622263,0x3feecc667b5de565,0x3c8b1c86e3e231d5,0x3feed09bec4a2d33,
0xbc91bbd1d3bcbb15,0x3feed503b23e255d,0x3c90cc319cee31d2,0x3feed99e1330b358,0x3c8469846e735ab3,0x3feede6b5579fdbf,0xbc82dfcd978e9db4,0x3feee36bbfd3f37a,
0x3c8c1a7792cb3387,0x3feee89f995ad3ad,0xbc907b8f4ad1d9fa,0x3feeee07298db666,0xbc55c3d956dcaeba,0x3feef3a2b84f15fb,0xbc90a40e3da6f640,0x3feef9728de5593a,
0xbc68d6f438ad9334,0x3feeff76f2fb5e47,0xbc91eee26b588a35,0x3fef05b030a1064a,0x3c74ffd70a5fddcd,0x3fef0c1e904bc1d2,0xbc91bdfbfa9298ac,0x3fef12c25bd71e09,
0x3c736eae30af0cb3,0x3fef199bdd85529c,0x3c8ee3325c9ffd94,0x3fef20ab5fffd07a,0x3c84e08fd10959ac,0x3fef27f12e57d14b,0x3c63cdaf384e1a67,0x3fef2f6d9406e7b5,
0x3c676b2c6c921968,0x3fef3720dcef9069,0xbc808a1883ccb5d2,0x3fef3f0b555dc3fa,0xbc8fad5d3ffffa6f,0x3fef472d4a07897c,0xbc900dae3875a949,0x3fef4f87080d89f2,
0x3c74a385a63d07a7,0x3fef5818dcfba487,0xbc82919e2040220f,0x3fef60e316c98398,0x3c8e5a50d5c192ac,0x3fef69e603db3285,0x3c843a59ac016b4b,0x3fef7321f301b460,
0xbc82d52107b43e1f,0x3fef7c97337b9b5f,0xbc892ab93b470dc9,0x3fef864614f5a129,0x3c74b604603a88d3,0x3fef902ee78b3ff6,0x3c83c5ec519d7271,0x3fef9a51fbc74c83,
0xbc8ff7128fd391f0,0x3fefa4afa2a490da,0xbc8dae98e223747d,0x3fefaf482d8e67f1,0x3c8ec3bc41aa2008,0x3fefba1bee615a27,0x3c842b94c3a9eb32,0x3fefc52b376bba97,
0x3c8a64a931d185ee,0x3fefd0765b6e4540,0xbc8e37bae43be3ed,0x3fefdbfdad9cbe14,0x3c77893b4d91cd9d,0x3fefe7c1819e90d8,0x3c5305c14160cc89,0x3feff3c22b8f71f1,
)

TAB = tuple(int(v, 16) if isinstance(v, str) else int(v) for v in TAB)


def f2i(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def i2f(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u & MASK64))[0]


def _fma(a: float, b: float, c: float) -> float:
    # exact fma: one rounding at the end (Fraction -> float via int/int
    # true division is correctly rounded nearest-even in CPython)
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _specialcase(tmp: float, sbits: int, ki: int, use_fma: bool) -> float:
    if (ki & 0x80000000) == 0:  # k > 0: overflow range
        scale = i2f((sbits - (1009 << 52)) & MASK64)
        inner = _fma(scale, tmp, scale) if use_fma else scale + scale * tmp
        return TWO_P1009 * inner
    scale = i2f((sbits + (1022 << 52)) & MASK64)
    y = _fma(scale, tmp, scale) if use_fma else scale + scale * tmp
    if y < 1.0:
        lo = _fma(scale, tmp, scale - y) if use_fma else \
            (scale - y) + scale * tmp
        hi = 1.0 + y
        lo = (1.0 - hi + y) + lo
        y = (hi + lo) - 1.0
        if y == 0.0:
            y = 0.0
    return TWO_P1022M * y


def _glibc_skeleton(x: float, fuse: str) -> float:
    """Exact pure-Python simulation of glibc 2.34 __exp.

    fuse selects the compilation variant's fma pattern:
      'plain' - no fma anywhere (non-fma multiarch build)
      'full'  - kd fused; r nested-fused; poly inners fused; tail+r plain;
                r2*p1, r4*p2, final scale fused
      'alt1'  - like full but kd NOT fused (z materialized first)
      'alt2'  - like full but only the poly block fused (kd and r plain)
    """
    xi = f2i(x)
    abstop = (xi >> 52) & 0x7FF
    d32 = (abstop - 0x3C9) & 0xFFFFFFFF  # C uint32 wrap semantics
    if d32 >= 0x408 - 0x3C9:
        if d32 >= 0x80000000:
            return 1.0 + x  # WANT_ROUNDING
        if abstop >= 0x409:
            if xi == NEG_INF_BITS:
                return 0.0
            if abstop >= 0x7FF:
                return 1.0 + x  # nan -> nan, +inf -> inf
            return 0.0 if (xi >> 63) else math.inf  # uflow / oflow
        abstop = 0
    if fuse in ("plain", "alt1", "alt2"):
        z = INV_LN2_N * x
        kd = z + SHIFT
    else:
        kd = _fma(INV_LN2_N, x, SHIFT)
    ki = f2i(kd) & MASK64
    kd = kd - SHIFT
    if fuse in ("plain", "alt1", "alt2"):
        r = (x + kd * NEG_LN2_HI_N) + kd * NEG_LN2_LO_N
    else:
        r = _fma(kd, NEG_LN2_LO_N, _fma(kd, NEG_LN2_HI_N, x))
    idx = 2 * (ki % 128)
    top = (ki << 45) & MASK64
    tail = i2f(TAB[idx])
    sbits = (TAB[idx + 1] + top) & MASK64
    r2 = r * r
    if fuse == "plain":
        # left association of the C expression, all plain ops
        tmp = ((tail + r) + r2 * (C2 + r * C3)) + \
            (r2 * r2) * (C4 + r * C5)
    else:
        t1 = tail + r
        t2 = _fma(r2, _fma(r, C3, C2), t1)
        tmp = _fma(r2 * r2, _fma(r, C5, C4), t2)
    if abstop == 0:
        return _specialcase(tmp, sbits, ki, use_fma=(fuse != "plain"))
    scale = i2f(sbits)
    if fuse == "plain":
        return scale + scale * tmp
    return _fma(scale, tmp, scale)


def sim_plain(x: float) -> float:
    return _glibc_skeleton(x, "plain")


def sim_fma_full(x: float) -> float:
    return _glibc_skeleton(x, "full")


def sim_fma_alt1(x: float) -> float:
    return _glibc_skeleton(x, "alt1")


def sim_fma_alt2(x: float) -> float:
    return _glibc_skeleton(x, "alt2")


# --- argument construction (deterministic, seed recorded) -------------------
def _rep4000(seed: int = SEED):
    # identical builder to the committed gpu probes (same seed)
    rng = np.random.default_rng(seed)
    origins, target_nnz = 4000, 238387
    base = target_nnz // origins
    extra = target_nnz - base * origins
    counts = np.full(origins, base, dtype=np.int64)
    counts[:extra] += 1
    indptr = np.zeros(origins + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    ids = np.empty(int(indptr[-1]), dtype=np.int64)
    dist = np.empty(int(indptr[-1]), dtype=np.float64)
    for o in range(origins):
        lo, hi = int(indptr[o]), int(indptr[o + 1])
        ids[lo:hi] = np.sort(rng.choice(20000, hi - lo, replace=False))
        dist[lo:hi] = np.sort(rng.uniform(1.0, 2000.0, hi - lo))
    return indptr, ids, dist


def _ulp_shift(x: float, n: int) -> float:
    return i2f(f2i(x) + n)


def build_args():
    rng = np.random.default_rng(SEED)
    groups, how = {}, {}

    groups["dense_grid_1m"] = np.linspace(-32.0, 32.0, 1 << 20)
    how["dense_grid_1m"] = "np.linspace(-32, 32, 2^20), uniform grid"

    groups["random_uniform_2m"] = rng.uniform(-32.0, 32.0, 2_000_000)
    how["random_uniform_2m"] = "rng.uniform(-32, 32), 2e6"

    e = rng.integers(1018, 1029, 1_000_000).astype(np.uint64)
    m = rng.integers(0, 1 << 52, 1_000_000).astype(np.uint64)
    s = (rng.integers(0, 2, 1_000_000) << 63).astype(np.uint64)
    groups["random_bitpat_1m"] = ((e << 52) | m | s).view(np.float64)
    how["random_bitpat_1m"] = ("random exponent band 2^-5..2^4 + random "
                               "52-bit mantissa + random sign, 1e6")

    mids = np.arange(-4600, 4601)
    xb = np.array([(mm + 0.5) / INV_LN2_N for mm in mids])
    groups["k_boundary"] = np.array(
        [_ulp_shift(v, n) for v in xb for n in (-3, -2, -1, 0, 1, 2, 3)],
        dtype=np.float64)
    how["k_boundary"] = ("x = (m+1/2)/InvLn2N +-1..3 ulp for m in "
                         "[-4600, 4600] (rounding flip surfaces)")

    xt = np.array([i / INV_LN2_N for i in range(0, 129)])
    # true 1-ulp neighbors: the bit-walk _ulp_shift(0.0, -1) wraps to NaN
    # (0x0 - 1 = 0xffff...), which is not a domain arg
    up = np.nextafter(xt, np.inf)
    dn = np.nextafter(xt, -np.inf)
    groups["idx_boundary"] = np.array(
        [v for t in zip(dn, xt, up) for v in t], dtype=np.float64)
    how["idx_boundary"] = ("x = i/InvLn2N with true 1-ulp neighbors "
                           "(nextafter both directions), i in [0, 128] "
                           "(idx flips)")

    groups["tiny_path"] = np.concatenate([
        np.array([0.0, -0.0, 5e-324, -5e-324]),
        np.array([_ulp_shift(2.0 ** -54, n) for n in (-2, -1, 0, 1, 2)]),
        rng.uniform(-2.0 ** -52, 2.0 ** -52, 200_000)])
    how["tiny_path"] = ("0, +-min-subnormal, ulp neighborhood of 2^-54, "
                        "2e5 uniform in +-2^-52 (the 1.0+x fast path)")

    # cohort-actual: rep4000 seeded structure, residual-probe logistic scalars
    _ip, _ids, dist = _rep4000()
    d = np.sort(dist[dist <= 1500.0])
    beta, plateau, mid, gr = 0.001, 0.0, 500.0, 0.01
    ge = beta * np.maximum(0.0, d - plateau)
    gl = -gr * ((d - plateau) - mid)
    # kn (knn=1) crop args: the minimum filtered distance per origin
    ip, ids2, dist2 = _rep4000()
    kn = np.empty(4000, dtype=np.float64)
    for o in range(4000):
        lo, hi = int(ip[o]), int(ip[o + 1])
        seg = dist2[lo:hi]
        kn[o] = seg[seg <= 1500.0].min()
    kn_gl = -gr * ((kn - plateau) - mid)
    kn_ge = beta * np.maximum(0.0, kn - plateau)
    groups["actual_rep4000"] = np.concatenate([ge, gl, kn_gl, kn_ge])
    how["actual_rep4000"] = ("rep4000 seeded structure, dist<=1500: "
                             "ge=beta*max(0,d-plateau), "
                             "gl=-gr*((d-plateau)-mid), kn(knn=1) crop "
                             "args per origin (both decay forms); "
                             "beta=0.001 gr=0.01 plateau=0 mid=500 "
                             "(the residual-probe logistic scalars)")

    betas = np.linspace(0.001, 0.004, 32)
    dgrid = np.linspace(0.0, 1500.0, 4096)
    groups["ge_box"] = (betas[:, None] *
                        np.maximum(0.0, dgrid[None, :])).reshape(-1)
    how["ge_box"] = ("beta in 32 manifest betas (0.001..0.004) x d in "
                     "4096-grid [0, 1500]: x = beta*max(0, d)")
    grs = np.array([0.002, 0.004, 0.006, 0.008, 0.01])
    pgrid = np.linspace(0.0, 1500.0, 4096)
    groups["gl_box"] = (-grs[:, None] *
                        (pgrid[None, :] - 500.0)).reshape(-1)
    how["gl_box"] = ("gr in {0.002,0.004,0.006,0.008,0.01} x p in "
                     "4096-grid [0, 1500]: x = -gr*(p-500)")

    groups["ood_large"] = np.concatenate([
        rng.uniform(33.0, 709.0, 20_000),
        -rng.uniform(33.0, 745.0, 20_000),
        np.array([709.782712893384, 709.782712893385, -744.440071921381,
                  -745.1332191019411, math.inf, -math.inf, math.nan,
                  2.0 ** 512, -(2.0 ** 512)])])
    how["ood_large"] = ("|x| in (32, 745] incl. overflow/underflow edges, "
                        "+-inf, nan (informational, OUTSIDE the gate "
                        "domain; exercises specialcase)")

    return groups, how


GATE_GROUPS = ["dense_grid_1m", "random_uniform_2m", "random_bitpat_1m",
               "k_boundary", "idx_boundary", "tiny_path", "actual_rep4000",
               "ge_box", "gl_box"]

# --- device kernels ----------------------------------------------------------
# One explicit kernel per scheme (numba.cuda jit cannot switch schemes at
# runtime); all share the glibc 2.34 skeleton.  Bitcasts go through
# 1-element local arrays (exact; not rounding ops).  For scheme='plain'
# no fast flags are set anywhere; scheme='fastmath' sets fastmath=True
# (contract+reassoc class); scheme='fma_full'/'fma_alt1'/'fma_alt2' call
# the explicitly rounded libdevice fma in the corresponding fusion order
# under default flags.


def _make_kernels():
    from numba import cuda
    from numba.cuda.libdevice import exp as ld_exp
    from numba.cuda.libdevice import fma as ld_fma
    f64 = np.float64
    u64 = np.uint64

    @cuda.jit(device=True)
    def _f2u(v):
        a = cuda.local.array(1, dtype=f64)
        a[0] = v
        return a.view(u64)[0]

    @cuda.jit(device=True)
    def _u2f(u):
        a = cuda.local.array(1, dtype=u64)
        a[0] = u
        return a.view(f64)[0]

    @cuda.jit(cache=False, fastmath=False)
    def kern_libdevice(xs, tab_u, out):
        i = cuda.grid(1)
        if i < xs.shape[0]:
            out[i] = _f2u(ld_exp(xs[i]))

    @cuda.jit(cache=False, fastmath=False)
    def kern_plain(xs, tab_u, out):
        i = cuda.grid(1)
        if i >= xs.shape[0]:
            return
        x = xs[i]
        xi = _f2u(x)
        abstop = (xi >> u64(52)) & u64(0x7FF)
        d32 = (abstop - u64(0x3C9)) & u64(0xFFFFFFFF)
        if d32 >= u64(0x3F):
            if d32 >= u64(0x80000000):
                out[i] = _f2u(1.0 + x)
                return
            if abstop >= u64(0x409):
                if xi == u64(0xFFF0000000000000):
                    out[i] = u64(0)
                    return
                if abstop >= u64(0x7FF):
                    out[i] = _f2u(1.0 + x)
                    return
                out[i] = u64(0) if xi >> u64(63) \
                    else u64(0x7FF0000000000000)
                return
            abstop = u64(0)
        z = INV_LN2_N * x
        kd = z + SHIFT
        ki = _f2u(kd)
        kd = kd - SHIFT
        r = (x + kd * NEG_LN2_HI_N) + kd * NEG_LN2_LO_N
        idx = 2 * (ki % u64(128))
        top = ki << u64(45)
        tail = _u2f(tab_u[idx])
        sbits = tab_u[idx + 1] + top
        r2 = r * r
        tmp = ((tail + r) + r2 * (C2 + r * C3)) + \
            (r2 * r2) * (C4 + r * C5)
        if abstop == u64(0):
            if (ki & u64(0x80000000)) == u64(0):
                sc = _u2f(sbits - u64(1009 << 52))
                out[i] = _f2u(TWO_P1009 * (sc + sc * tmp))
            else:
                sc = _u2f(sbits + u64(1022 << 52))
                y = sc + sc * tmp
                if y < 1.0:
                    lo = (sc - y) + sc * tmp
                    hi = 1.0 + y
                    lo = (1.0 - hi + y) + lo
                    y = (hi + lo) - 1.0
                    if y == 0.0:
                        y = 0.0
                out[i] = _f2u(TWO_P1022M * y)
            return
        sc = _u2f(sbits)
        out[i] = _f2u(sc + sc * tmp)

    @cuda.jit(cache=False, fastmath=True)
    def kern_fastmath(xs, tab_u, out):
        i = cuda.grid(1)
        if i >= xs.shape[0]:
            return
        x = xs[i]
        xi = _f2u(x)
        abstop = (xi >> u64(52)) & u64(0x7FF)
        d32 = (abstop - u64(0x3C9)) & u64(0xFFFFFFFF)
        if d32 >= u64(0x3F):
            if d32 >= u64(0x80000000):
                out[i] = _f2u(1.0 + x)
                return
            if abstop >= u64(0x409):
                if xi == u64(0xFFF0000000000000):
                    out[i] = u64(0)
                    return
                if abstop >= u64(0x7FF):
                    out[i] = _f2u(1.0 + x)
                    return
                out[i] = u64(0) if xi >> u64(63) \
                    else u64(0x7FF0000000000000)
                return
            abstop = u64(0)
        z = INV_LN2_N * x
        kd = z + SHIFT
        ki = _f2u(kd)
        kd = kd - SHIFT
        r = (x + kd * NEG_LN2_HI_N) + kd * NEG_LN2_LO_N
        idx = 2 * (ki % u64(128))
        top = ki << u64(45)
        tail = _u2f(tab_u[idx])
        sbits = tab_u[idx + 1] + top
        r2 = r * r
        tmp = ((tail + r) + r2 * (C2 + r * C3)) + \
            (r2 * r2) * (C4 + r * C5)
        if abstop == u64(0):
            if (ki & u64(0x80000000)) == u64(0):
                sc = _u2f(sbits - u64(1009 << 52))
                out[i] = _f2u(TWO_P1009 * (sc + sc * tmp))
            else:
                sc = _u2f(sbits + u64(1022 << 52))
                y = sc + sc * tmp
                if y < 1.0:
                    lo = (sc - y) + sc * tmp
                    hi = 1.0 + y
                    lo = (1.0 - hi + y) + lo
                    y = (hi + lo) - 1.0
                    if y == 0.0:
                        y = 0.0
                out[i] = _f2u(TWO_P1022M * y)
            return
        sc = _u2f(sbits)
        out[i] = _f2u(sc + sc * tmp)

    @cuda.jit(cache=False, fastmath=False)
    def kern_fma_full(xs, tab_u, out):
        i = cuda.grid(1)
        if i >= xs.shape[0]:
            return
        x = xs[i]
        xi = _f2u(x)
        abstop = (xi >> u64(52)) & u64(0x7FF)
        d32 = (abstop - u64(0x3C9)) & u64(0xFFFFFFFF)
        if d32 >= u64(0x3F):
            if d32 >= u64(0x80000000):
                out[i] = _f2u(1.0 + x)
                return
            if abstop >= u64(0x409):
                if xi == u64(0xFFF0000000000000):
                    out[i] = u64(0)
                    return
                if abstop >= u64(0x7FF):
                    out[i] = _f2u(1.0 + x)
                    return
                out[i] = u64(0) if xi >> u64(63) \
                    else u64(0x7FF0000000000000)
                return
            abstop = u64(0)
        kd = ld_fma(INV_LN2_N, x, SHIFT)
        ki = _f2u(kd)
        kd = kd - SHIFT
        r = ld_fma(kd, NEG_LN2_LO_N, ld_fma(kd, NEG_LN2_HI_N, x))
        idx = 2 * (ki % u64(128))
        top = ki << u64(45)
        tail = _u2f(tab_u[idx])
        sbits = tab_u[idx + 1] + top
        r2 = r * r
        t1 = tail + r
        t2 = ld_fma(r2, ld_fma(r, C3, C2), t1)
        tmp = ld_fma(r2 * r2, ld_fma(r, C5, C4), t2)
        if abstop == u64(0):
            if (ki & u64(0x80000000)) == u64(0):
                sc = _u2f(sbits - u64(1009 << 52))
                out[i] = _f2u(TWO_P1009 * ld_fma(sc, tmp, sc))
            else:
                sc = _u2f(sbits + u64(1022 << 52))
                y = ld_fma(sc, tmp, sc)
                if y < 1.0:
                    lo = ld_fma(sc, tmp, sc - y)
                    hi = 1.0 + y
                    lo = (1.0 - hi + y) + lo
                    y = (hi + lo) - 1.0
                    if y == 0.0:
                        y = 0.0
                out[i] = _f2u(TWO_P1022M * y)
            return
        sc = _u2f(sbits)
        out[i] = _f2u(ld_fma(sc, tmp, sc))

    @cuda.jit(cache=False, fastmath=False)
    def kern_fma_alt1(xs, tab_u, out):
        i = cuda.grid(1)
        if i >= xs.shape[0]:
            return
        x = xs[i]
        xi = _f2u(x)
        abstop = (xi >> u64(52)) & u64(0x7FF)
        d32 = (abstop - u64(0x3C9)) & u64(0xFFFFFFFF)
        if d32 >= u64(0x3F):
            if d32 >= u64(0x80000000):
                out[i] = _f2u(1.0 + x)
                return
            if abstop >= u64(0x409):
                if xi == u64(0xFFF0000000000000):
                    out[i] = u64(0)
                    return
                if abstop >= u64(0x7FF):
                    out[i] = _f2u(1.0 + x)
                    return
                out[i] = u64(0) if xi >> u64(63) \
                    else u64(0x7FF0000000000000)
                return
            abstop = u64(0)
        z = INV_LN2_N * x
        kd = z + SHIFT
        ki = _f2u(kd)
        kd = kd - SHIFT
        r = ld_fma(kd, NEG_LN2_LO_N, ld_fma(kd, NEG_LN2_HI_N, x))
        idx = 2 * (ki % u64(128))
        top = ki << u64(45)
        tail = _u2f(tab_u[idx])
        sbits = tab_u[idx + 1] + top
        r2 = r * r
        t1 = tail + r
        t2 = ld_fma(r2, ld_fma(r, C3, C2), t1)
        tmp = ld_fma(r2 * r2, ld_fma(r, C5, C4), t2)
        if abstop == u64(0):
            if (ki & u64(0x80000000)) == u64(0):
                sc = _u2f(sbits - u64(1009 << 52))
                out[i] = _f2u(TWO_P1009 * ld_fma(sc, tmp, sc))
            else:
                sc = _u2f(sbits + u64(1022 << 52))
                y = ld_fma(sc, tmp, sc)
                if y < 1.0:
                    lo = ld_fma(sc, tmp, sc - y)
                    hi = 1.0 + y
                    lo = (1.0 - hi + y) + lo
                    y = (hi + lo) - 1.0
                    if y == 0.0:
                        y = 0.0
                out[i] = _f2u(TWO_P1022M * y)
            return
        sc = _u2f(sbits)
        out[i] = _f2u(ld_fma(sc, tmp, sc))

    @cuda.jit(cache=False, fastmath=False)
    def kern_fma_alt2(xs, tab_u, out):
        i = cuda.grid(1)
        if i >= xs.shape[0]:
            return
        x = xs[i]
        xi = _f2u(x)
        abstop = (xi >> u64(52)) & u64(0x7FF)
        d32 = (abstop - u64(0x3C9)) & u64(0xFFFFFFFF)
        if d32 >= u64(0x3F):
            if d32 >= u64(0x80000000):
                out[i] = _f2u(1.0 + x)
                return
            if abstop >= u64(0x409):
                if xi == u64(0xFFF0000000000000):
                    out[i] = u64(0)
                    return
                if abstop >= u64(0x7FF):
                    out[i] = _f2u(1.0 + x)
                    return
                out[i] = u64(0) if xi >> u64(63) \
                    else u64(0x7FF0000000000000)
                return
            abstop = u64(0)
        z = INV_LN2_N * x
        kd = z + SHIFT
        ki = _f2u(kd)
        kd = kd - SHIFT
        r = (x + kd * NEG_LN2_HI_N) + kd * NEG_LN2_LO_N
        idx = 2 * (ki % u64(128))
        top = ki << u64(45)
        tail = _u2f(tab_u[idx])
        sbits = tab_u[idx + 1] + top
        r2 = r * r
        t1 = tail + r
        t2 = ld_fma(r2, ld_fma(r, C3, C2), t1)
        tmp = ld_fma(r2 * r2, ld_fma(r, C5, C4), t2)
        if abstop == u64(0):
            if (ki & u64(0x80000000)) == u64(0):
                sc = _u2f(sbits - u64(1009 << 52))
                out[i] = _f2u(TWO_P1009 * ld_fma(sc, tmp, sc))
            else:
                sc = _u2f(sbits + u64(1022 << 52))
                y = ld_fma(sc, tmp, sc)
                if y < 1.0:
                    lo = ld_fma(sc, tmp, sc - y)
                    hi = 1.0 + y
                    lo = (1.0 - hi + y) + lo
                    y = (hi + lo) - 1.0
                    if y == 0.0:
                        y = 0.0
                out[i] = _f2u(TWO_P1022M * y)
            return
        sc = _u2f(sbits)
        out[i] = _f2u(ld_fma(sc, tmp, sc))

    @cuda.jit(cache=False, fastmath=False)
    def kern_contract_probe(a, b, c, out):
        # a*b + c under DEFAULT flags: detects whether NVVM contracts.
        # The chosen triples satisfy fma(a,b,c) != round(round(a*b)+c).
        i = cuda.grid(1)
        if i < a.shape[0]:
            out[i] = _f2u(a[i] * b[i] + c[i])

    return {"libdevice": kern_libdevice, "plain": kern_plain,
            "fastmath": kern_fastmath, "fma_full": kern_fma_full,
            "fma_alt1": kern_fma_alt1, "fma_alt2": kern_fma_alt2,
            "contract_probe": kern_contract_probe}


SCHEMES = ["libdevice", "plain", "fastmath", "fma_full", "fma_alt1",
           "fma_alt2"]


def _ulp_dist(a_bits: np.ndarray, b_bits: np.ndarray) -> np.ndarray:
    """Informational ulp distance (same-sign finite only; else -1)."""
    ia = a_bits.astype(np.int64)
    ib = b_bits.astype(np.int64)
    sa = (a_bits >> np.uint64(63)).astype(np.int64)
    sb = (b_bits >> np.uint64(63)).astype(np.int64)
    fa = np.where(sa == 1, -ia, ia)
    fb = np.where(sb == 1, -ib, ib)
    d = np.abs(fb - fa)
    return np.where((sa == sb), d, -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--id-subsampling", type=int, default=20000)
    args = ap.parse_args()

    result = {"schema": SCHEMA,
              "purpose": "Direction 1 road (a): can a numba-CUDA exp "
                         "reconstruction be BITWISE equal to the delivered "
                         "host (glibc 2.34) exp over the cohort domain; "
                         "gate = any divergence in domain => no-go",
              "python": sys.executable,
              "numpy": np.__version__,
              "numba": __import__("numba").__version__,
              "seed": SEED,
              "glibc_source": GLIBC_SRC,
              "domain": ("gate domain |x| <= 32 plus the tiny path (incl "
                         "0.0), covering ge/gl/kn cohort feeds with margin; "
                         "ood_large is informational only")}
    t0 = time.perf_counter()
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

    groups, how = build_args()
    result["groups"] = how
    all_x = {k: np.ascontiguousarray(v, dtype=np.float64)
             for k, v in groups.items()}
    result["case_counts"] = {k: int(v.size) for k, v in all_x.items()}
    gate_total = sum(all_x[k].size for k in GATE_GROUPS)
    result["gate_domain_cases"] = int(gate_total)
    print(json.dumps({"stage": "args", "gate_cases": gate_total,
                      "total": int(sum(v.size for v in all_x.values()))}),
          flush=True)

    # host legs + authority (glibc scalar exp)
    libm = ctypes.CDLL("libm.so.6")
    libm.exp.restype = ctypes.c_double
    libm.exp.argtypes = [ctypes.c_double]
    probe_x = all_x["random_uniform_2m"][:5000]
    from numba import njit

    @njit(cache=False)
    def _auth_loop(xs):
        out = np.empty(xs.shape[0], dtype=np.float64)
        for i in range(xs.shape[0]):
            out[i] = math.exp(xs[i])
        return out

    leg = {"numba_njit_math_exp": _auth_loop(probe_x).view(np.uint64)}
    leg["ctypes_libm"] = np.array([libm.exp(float(v)) for v in probe_x],
                                  dtype=np.float64).view(np.uint64)
    leg["math_exp"] = np.array([math.exp(float(v)) for v in probe_x],
                               dtype=np.float64).view(np.uint64)
    leg["np_exp"] = np.exp(probe_x).view(np.uint64)
    auth0 = leg["numba_njit_math_exp"]
    result["host_leg_mismatches_vs_numba_5000"] = {
        k: int(np.count_nonzero(v != auth0)) for k, v in leg.items()
        if k != "numba_njit_math_exp"}
    legs_ok = (result["host_leg_mismatches_vs_numba_5000"]["ctypes_libm"]
               == 0 and
               result["host_leg_mismatches_vs_numba_5000"]["math_exp"] == 0)
    result["authority"] = (
        "glibc 2.34 scalar exp: numba njit math.exp == ctypes libm == "
        "math.exp; np.exp is a different host implementation (recorded "
        "mismatches), not the authority" if legs_ok else
        "UNEXPECTED host-leg disagreement - results not interpretable")
    print(json.dumps({"stage": "host_legs",
                      "mism": result["host_leg_mismatches_vs_numba_5000"]}),
          flush=True)

    # variant identification vs exact sims (normal-range subsample)
    id_x = np.concatenate([
        all_x["random_uniform_2m"][:args.id_subsampling // 2],
        all_x["random_bitpat_1m"][:args.id_subsampling // 2]])
    sims = {"sim_plain": sim_plain, "sim_fma_full": sim_fma_full,
            "sim_fma_alt1": sim_fma_alt1, "sim_fma_alt2": sim_fma_alt2}
    ident = {}
    for nm, fn in sims.items():
        ident[nm] = sum(1 for v in id_x
                        if f2i(fn(float(v))) != f2i(math.exp(float(v))))
    identified = min(ident, key=ident.get)
    result["variant_identification"] = {
        "subsample": int(id_x.size),
        "mismatches_vs_glibc": ident,
        "identified": identified,
        "note": "bit-level comparison (f2i); 0 mismatches = exact "
                "variant; the matching device fma-scheme port targets it"}
    print(json.dumps({"stage": "variant_id", "mism": ident}), flush=True)

    # fusion-order discrimination: the kd/r fusion placements only matter
    # near ki-tick boundaries, so re-identify on the boundary groups
    id2_x = np.concatenate([all_x["k_boundary"][::3],
                            all_x["idx_boundary"][::3]])
    ident2 = {}
    for nm, fn in sims.items():
        ident2[nm] = sum(1 for v in id2_x
                         if f2i(fn(float(v))) != f2i(math.exp(float(v))))
    result["variant_identification_boundaries"] = {
        "subsample": int(id2_x.size), "stride": 3,
        "mismatches_vs_glibc": ident2,
        "note": "bit-level comparison; discriminates fusion orders "
                "(kd/r fma placement) that random args cannot"}
    print(json.dumps({"stage": "variant_id_boundary", "mism": ident2}),
          flush=True)

    # authority bits per group
    auth_bits = {k: _auth_loop(v).view(np.uint64) for k, v in all_x.items()}

    # --- device section (typed gate, never simulated) --------------------
    device = {"status": "not_run"}
    try:
        from numba import cuda
        if not cuda.is_available():
            raise RuntimeError("no CUDA device / driver init failed")
        dev = cuda.get_current_device()
        name = dev.name.decode() if isinstance(dev.name, bytes) \
            else str(dev.name)
        try:
            uuid = str(dev.uuid)
        except Exception:
            uuid = None
        device = {"status": "ok", "name": name, "uuid": uuid,
                  "cc": ".".join(map(str, dev.compute_capability))}

        kernels = _make_kernels()
        tab_dev = cuda.to_device(np.array(TAB, dtype=np.uint64))
        device["schemes"] = {}
        for scheme in SCHEMES:
            per = {}
            for gname, xs in all_x.items():
                out = np.empty(xs.shape[0], dtype=np.uint64)
                dx = cuda.to_device(xs)
                douts = cuda.device_array(xs.shape[0], dtype=np.uint64)
                kernels[scheme][xs.shape[0] // 128 + 1, 128](
                    dx, tab_dev, douts)
                cuda.synchronize()
                douts.copy_to_host(out)
                a = auth_bits[gname]
                diff = out != a
                n_diff = int(np.count_nonzero(diff))
                entry = {"n": int(xs.size), "n_diff": n_diff}
                if n_diff:
                    ulp = _ulp_dist(a[diff], out[diff])
                    fin = ulp >= 0
                    entry["max_ulp_finite"] = (int(ulp[fin].max())
                                               if fin.any() else None)
                    entry["n_sign_flips"] = int(np.count_nonzero(~fin))
                    first = np.flatnonzero(diff)[:8]
                    entry["first_mismatch"] = [
                        {"x_hex": hex(f2i(float(xs[i]))),
                         "authority_hex": hex(int(a[i])),
                         "scheme_hex": hex(int(out[i]))} for i in first]
                per[gname] = entry
            gate_diff = sum(per[g]["n_diff"] for g in GATE_GROUPS)
            device["schemes"][scheme] = {"groups": per,
                                         "gate_domain_diff": int(gate_diff),
                                         "verdict":
                                             "GO" if gate_diff == 0
                                             else "NO-GO"}
            print(json.dumps({"stage": "scheme", "scheme": scheme,
                              "gate_domain_diff": int(gate_diff)}),
                  flush=True)

        # contraction detector: triples with fma != round(round(ab)+c)
        da = np.array([np.nextafter(1.0, 2.0), 1.0 + 2 ** -52,
                       3.141592653589793, 0.1, 1e8 + 0.5])
        db = np.array([1.0000000000000002, 3.0, 0.9999999999999999,
                       1e-8, 1.000000000000001])
        dc = np.array([2 ** -53, 1.0, 2 ** -54, 1e8, 2 ** -52])
        exact_fma = np.array([_fma(float(a), float(b), float(c))
                              for a, b, c in zip(da, db, dc)])
        sep = np.array([float(a) * float(b) + float(c)
                        for a, b, c in zip(da, db, dc)])
        differs = exact_fma != sep  # at least one triple must differ
        if not differs.any():
            raise RuntimeError("contraction detector triples are degenerate")
        outp = np.empty(da.shape[0], dtype=np.uint64)
        d1, d2, d3 = (cuda.to_device(da), cuda.to_device(db),
                      cuda.to_device(dc))
        douts = cuda.device_array(da.shape[0], dtype=np.uint64)
        kernels["contract_probe"][1, 8](d1, d2, d3, douts)
        cuda.synchronize()
        douts.copy_to_host(outp)
        got = np.array([i2f(int(v)) for v in outp])
        device["contraction_detector"] = {
            "triples_where_fma_differs": int(np.count_nonzero(differs)),
            "device_matches_exact_fma": int(np.count_nonzero(got ==
                                                             exact_fma)),
            "device_matches_separate": int(np.count_nonzero(got == sep)),
            "reading": ("NVVM contracts a*b+c under default flags"
                        if np.count_nonzero(got == exact_fma) >
                        np.count_nonzero(got == sep) else
                        "NVVM does NOT contract under default flags")}

    except Exception as e:  # typed blocker, never simulated
        device = {"status": "device_unavailable", "error": repr(e),
                  "note": "host characterization above still recorded; "
                          "no device numbers produced, nothing simulated"}
    print(json.dumps({"stage": "device", "status": device["status"]}),
          flush=True)

    result["device"] = device
    if device["status"] == "ok":
        go = [s for s in SCHEMES
              if device["schemes"][s]["verdict"] == "GO"]
        result["road_a_verdict"] = (
            ("GO at the primitive level: " + ", ".join(go) +
             " bitwise-equal to glibc on every gate-domain case; fold-level "
             "admission still additionally requires the delivered fold's "
             "compiled reduction lowering to match (gpuprobe2 census gap)")
            if go else
            "NO-GO at the primitive level: no scheme is bitwise-equal on "
            "the gate domain; the exp-cell domain stays closed (no "
            "tolerance weakening)")
    else:
        result["road_a_verdict"] = ("device_unavailable: no device facts; "
                                    "host characterization recorded only")
    result["elapsed_s"] = time.perf_counter() - t0
    result["law"] = ("bitwise only; any gate-domain divergence = no-go; "
                     "no tolerance; no CPU fallback touched; no route/"
                     "admission claim created")
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(args.out),
                      "verdict": result["road_a_verdict"][:80]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
