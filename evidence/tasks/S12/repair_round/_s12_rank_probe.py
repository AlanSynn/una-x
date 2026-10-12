"""Probe baseline vs sparse behavior for degenerate-rank inputs (S12
review F1/F2/F3 + same-law origin/idx classes) before pinning tests."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests/tenx/sparse")
sys.path.insert(0, "src")
from conftest import admitted_case, bits, driver_args  # noqa: E402

from urban_network_analysis.Engines.AccessibilityWElevation import (  # noqa: E402
    integrated_scope_access)
from urban_network_analysis.Engines import _sparse_access_scratch as sp  # noqa: E402

RESULT_FIELDS = ("reach", "gravity_exponential", "gravity_logistic",
                 "knn_access")


def classify(name, mutate):
    case = admitted_case("base")
    mutate(case)
    args = driver_args(case)
    try:
        ref = integrated_scope_access(*args)
        base = ("runs", None)
    except Exception as e:  # noqa: BLE001 - classification probe
        base = ("raises", type(e).__name__)
        ref = None
    try:
        res = sp.sparse_scope_access(*args)
        got = ("runs", res.route, res.refusal,
               None if ref is None else all(
                   bits(getattr(res, f)) == bits(b)
                   for f, b in zip(RESULT_FIELDS, ref)))
        if base[0] == "runs":
            verdict = "PARITY-OK" if (got[1] == sp.ROUTE_FALLBACK
                                      and got[2] == "shape" and got[3]) \
                else "HOLE"
        else:
            verdict = "HOLE (baseline raises, sparse ran)"
    except Exception as e:  # noqa: BLE001
        got = ("raises", type(e).__name__)
        verdict = ("PARITY-OK" if base[0] == "raises"
                   and type(e).__name__ == base[1] else "DIVERGENT")
    print("%-14s baseline=%-22s sparse=%-46s %s"
          % (name, base, str(got)[:46], verdict))
    return verdict


def m_dwt_2d(c):
    c["dwt"] = np.ascontiguousarray(c["dwt"][:, None])


def m_dwt_0d(c):
    c["dwt"] = np.array(0.5)


def m_dw_1d(c):
    c["dw"] = np.ascontiguousarray(c["dw"][:, 0])


def m_d_1d(c):
    c["d"] = np.ascontiguousarray(c["d"][:, 0])


def m_ow_1d(c):
    c["ow"] = np.ascontiguousarray(c["ow"][:, 0])


def m_o_1d(c):
    c["o"] = np.ascontiguousarray(c["o"][:, 0])


results = {}
for name, fn in [("dwt_2d", m_dwt_2d), ("dwt_0d", m_dwt_0d),
                 ("dw_1d", m_dw_1d), ("d_1d", m_d_1d),
                 ("ow_1d", m_ow_1d), ("o_1d", m_o_1d)]:
    results[name] = classify(name, fn)
print("SUMMARY:", results)
