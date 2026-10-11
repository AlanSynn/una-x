"""Compat top-level namespace reflection probe (API_REVIEW R1 closure,
mda-20261001T1215Z): dump the public top-level namespace of one arm's
``compat.madina`` / upstream ``madina``, the identity of its submodule
bindings, and the public ``una`` namespace, as a JSON digest.

Executed as a subprocess under the reference venv interpreter (the only
interpreter where the pinned upstream ``madina`` is importable) for the
reference arm; the facade arm runs under any interpreter with geopandas.
The test compares the two digests modulo an explicit, ledgered delta
list.  Live enumeration from ``.refs`` — the name lists are never
hardcoded in the test.

Identity facts pinned here (upstream, pinned 8b5c3bd3):
  m.zonal       inner zonal.py MODULE (star-import shadowing quirk);
                the compat namespace deliberately restores the package
                (compat/madina/__init__.py, PACKAGE handoff I6).
  m.una         the una PACKAGE in BOTH arms (the una namespace carries
                no ``una``/``zonal`` name, so ``from .una import *``
                does not shadow it — no restore needed or performed).
  m.betweenness / m.paths   the una submodules, both arms.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def pub(ns):
    return sorted(n for n in dir(ns) if not n.startswith("_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["facade", "reference"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = {"arm": args.arm}
    try:
        if args.arm == "reference":
            sys.path.insert(0, str(REPO / ".refs" / "madina_ref" / "src"))
            import madina as m                       # noqa: PLC0415
        else:
            sys.path.insert(0, str(REPO / "src"))
            import urban_network_analysis.compat.madina as m  # noqa: PLC0415
        result["top_public"] = pub(m)
        result["identity"] = {
            name: {
                "is_package": hasattr(obj, "__path__"),
                "module": getattr(obj, "__name__", None),
            }
            for name, obj in (
                ("zonal", getattr(m, "zonal", None)),
                ("una", getattr(m, "una", None)),
                ("betweenness", getattr(m, "betweenness", None)),
                ("paths", getattr(m, "paths", None)),
            )
        }
        una = sys.modules[m.__name__ + ".una"]
        result["una_public"] = pub(una)
    except Exception:
        result["probe_error"] = traceback.format_exc()[-4000:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=1))
    print(f"reflection: {out}")


if __name__ == "__main__":
    main()
