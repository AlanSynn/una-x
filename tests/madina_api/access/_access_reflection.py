"""MADINA_ACCESS reflection probe: dump the public surface of one arm's
``una`` namespace — the package, ``una.betweenness`` and ``una.tools``
modules (public names, function signatures via ``inspect.signature``) —
as a JSON digest.

Executed as a subprocess under the reference venv interpreter (the only
interpreter where the pinned upstream ``madina`` is importable).  The
test compares the two arm digests modulo an explicit, ledgered delta
list — reflection is necessary, not sufficient (dossier 01).

Post-ACCESS ledger state (see compat una/__init__ ledger): the
access-engine trio (get_origin_properties / one_access /
parallel_access) and the three tools functions (validate_zonal_ready /
accessibility / service_area) are delivered verbatim; the reference-only
names shrink to the five undelivered betweenness functions (MADINA_FLOW
extends the same facade module and retires the rest).
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

BETWEENNESS_FUNCTIONS = (
    "get_origin_properties",
    "one_access",
    "parallel_access",
)

TOOLS_FUNCTIONS = (
    "validate_zonal_ready",
    "accessibility",
    "service_area",
    "alternative_paths",
)


def surface(una_pkg):
    # importing the submodules explicitly: neither package __init__
    # star-imports tools, so the attribute binding comes from this
    # import (same for both arms)
    btd_mod = importlib.import_module(una_pkg.__name__ + ".betweenness")
    tools_mod = importlib.import_module(una_pkg.__name__ + ".tools")
    out = {}
    # module namespaces (public names only; order-insensitive sets as
    # sorted lists)
    out["una_public"] = sorted(
        n for n in dir(una_pkg) if not n.startswith("_"))
    out["betweenness_public"] = sorted(
        n for n in dir(btd_mod) if not n.startswith("_"))
    out["tools_public"] = sorted(
        n for n in dir(tools_mod) if not n.startswith("_"))
    # exact signatures of the delivered surface (verbatim-port claim)
    out["betweenness_functions"] = {
        fn: str(inspect.signature(getattr(btd_mod, fn)))
        for fn in BETWEENNESS_FUNCTIONS}
    out["tools_functions"] = {
        fn: str(inspect.signature(getattr(tools_mod, fn)))
        for fn in TOOLS_FUNCTIONS}
    # cross-module identity: una.<fn> is the module attribute (upstream
    # star-import semantics) for the betweenness trio; the tools
    # functions are NOT bound at the una level on either side (neither
    # package __init__ imports tools) — recorded as a binding map the
    # test pins equal across arms
    out["star_exports_are_betweenness_functions"] = all(
        getattr(una_pkg, fn) is getattr(btd_mod, fn)
        for fn in BETWEENNESS_FUNCTIONS)
    out["una_level_tools_bindings"] = {
        fn: hasattr(una_pkg, fn) for fn in TOOLS_FUNCTIONS}
    # submodule identity
    out["una_betweenness_is_betweenness"] = una_pkg.betweenness is btd_mod
    out["una_tools_is_tools"] = una_pkg.tools is tools_mod
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["facade", "reference"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = {"arm": args.arm}
    try:
        if args.arm == "reference":
            sys.path.insert(0, str(REPO / ".refs" / "madina_ref" / "src"))
            import madina.una as una_pkg        # noqa: PLC0415
        else:
            sys.path.insert(0, str(REPO / "src"))
            from urban_network_analysis.compat.madina import una as una_pkg  # noqa: PLC0415
        result["identity"] = {"package_has_path": hasattr(una_pkg, "__path__")}
        result["surface"] = surface(una_pkg)
    except Exception:
        result["probe_error"] = traceback.format_exc()[-4000:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=1))
    print(f"reflection: {out}")


if __name__ == "__main__":
    main()
