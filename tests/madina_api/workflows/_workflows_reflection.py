"""MADINA_WORKFLOWS reflection probe: surface pin of the pairing-
workflow API (module exports, workflow signatures, Logger method
signatures and defaults) captured from EACH arm's own module — the two
arms' digests must be identical, pinning the facade surface against
the pinned upstream reference.

Usage: python _workflows_reflection.py --arm {facade,reference} --out D
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# annotation objects stringify with the arm-qualified module path
# (madina.zonal.zonal.Zonal vs the compat path) — environment layout,
# not behavior; normalize before comparing arms.  Longest path first:
# the compat path ends with the madina path, so the shorter replace
# must not run first.
_QUALIFIED_ZONAL_PATHS = (
    "urban_network_analysis.compat.madina.zonal.zonal.Zonal",
    "madina.zonal.zonal.Zonal",
)


def _norm_sig(sig):
    if not isinstance(sig, str):
        return sig
    for path in _QUALIFIED_ZONAL_PATHS:
        sig = sig.replace(path, "<Zonal>")
    return sig


def capture(arm):
    if arm == "reference":
        sys.path.insert(0, str(REPO / ".refs" / "madina_ref" / "src"))
        import madina.una.workflows as wf        # noqa: PLC0415
    elif arm == "facade":
        sys.path.insert(0, str(REPO / "src"))
        import urban_network_analysis.compat.madina.una.workflows as wf  # noqa: PLC0415
    else:
        raise ValueError(f"unknown arm {arm}")

    out = {}
    out["module_file"] = Path(wf.__file__).name
    out["public_functions"] = sorted(
        n for n in vars(wf)
        if not n.startswith("_") and inspect.isfunction(getattr(wf, n))
        and getattr(wf, n).__module__ == wf.__name__)
    out["classes"] = sorted(
        n for n in vars(wf)
        if not n.startswith("_") and inspect.isclass(getattr(wf, n))
        and getattr(wf, n).__module__ == wf.__name__)
    for fname in ("betweenness_flow_simulation", "KNN_accessibility"):
        fn = getattr(wf, fname, None)
        out[f"signature::{fname}"] = (
            _norm_sig(str(inspect.signature(fn))) if fn else "<absent>")
        out[f"docstring_head::{fname}"] = (
            (fn.__doc__ or "").splitlines()[0] if fn else "<absent>")
    logger = getattr(wf, "Logger", None)
    if logger is None:
        out["Logger"] = "<absent>"
    else:
        out["Logger::methods"] = sorted(
            n for n in vars(logger)
            if not n.startswith("__") and inspect.isfunction(vars(logger)[n]))
        for m in ("log", "pairing_end", "simulation_end",
                  "flow_map_template_1"):
            meth = vars(logger).get(m)
            out[f"signature::Logger.{m}"] = (
                _norm_sig(str(inspect.signature(meth))) if meth
                else "<absent>")
        init = vars(logger).get("__init__")
        out["signature::Logger.__init__"] = (
            _norm_sig(str(inspect.signature(init))) if init else "<absent>")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["facade", "reference"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = capture(args.arm)
    out["arm"] = args.arm
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
