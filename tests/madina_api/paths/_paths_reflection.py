"""MADINA_PATHS reflection probe: dump the public surface of one arm's
``una`` namespace — the package, ``una.paths`` and ``una.tools`` modules
(public names, function signatures via ``inspect.signature``) — as a
JSON digest.

Executed as a subprocess under the reference venv interpreter (the only
interpreter where the pinned upstream ``madina`` is importable).  The
test compares the two arm digests modulo an explicit, ledgered delta
list — reflection is necessary, not sufficient (dossier 01).

Interim-surface note (see compat una/__init__ ledger): upstream
star-imports ``.betweenness`` before ``.paths`` and its tools.py also
holds accessibility/service_area/betweenness/validate_zonal_ready; those
facade modules land with the accessibility/betweenness tasks.  This
probe records the interim delta explicitly instead of hiding it.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

PATHS_FUNCTIONS = (
    "path_generator",
    "bfs_subgraph_generation",
    "bfs_paths_many_targets_iterative",
    "wandering_messenger",
    "bfs_path_edges_many_targets_iterative",
    "turn_o_scope",
    "turn_penalty_value",
    "angle_deviation_between_two_lines",
)

TOOLS_FUNCTIONS = ("alternative_paths",)


def surface(una_pkg):
    import importlib
    paths_mod = una_pkg.paths
    # importing the tools submodule explicitly: the package __init__ on
    # BOTH sides star-imports only betweenness/paths, so the attribute
    # binding for tools comes from this import (same for both arms)
    tools_mod = importlib.import_module(una_pkg.__name__ + ".tools")
    out = {}
    # module namespaces (public names only; order-insensitive sets as
    # sorted lists)
    out["una_public"] = sorted(
        n for n in dir(una_pkg) if not n.startswith("_"))
    out["paths_public"] = sorted(
        n for n in dir(paths_mod) if not n.startswith("_"))
    out["tools_public"] = sorted(
        n for n in dir(tools_mod) if not n.startswith("_"))
    # exact signatures of the paths surface (verbatim-port claim)
    out["paths_functions"] = {
        fn: str(inspect.signature(getattr(paths_mod, fn)))
        for fn in PATHS_FUNCTIONS}
    out["tools_functions"] = {
        fn: str(inspect.signature(getattr(tools_mod, fn)))
        for fn in TOOLS_FUNCTIONS}
    # cross-module identity: una.<fn> is una.paths.<fn>
    out["star_exports_are_paths_functions"] = all(
        getattr(una_pkg, fn) is getattr(paths_mod, fn)
        for fn in PATHS_FUNCTIONS)
    # submodule identity
    out["una_paths_is_paths"] = una_pkg.paths is paths_mod
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
            result["identity"] = {"package_has_path": hasattr(una_pkg, "__path__")}
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
