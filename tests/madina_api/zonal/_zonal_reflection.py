"""MADINA_ZONAL reflection probe: dump the public surface of one arm's
``zonal`` namespace (module constants, class/instance methods with
exact ``inspect.signature`` strings, properties) plus its module-level
public names, as a JSON digest.

Executed as a subprocess under the reference venv interpreter (the only
interpreter where the pinned upstream ``madina`` is importable).  The
test compares the two arm digests modulo an explicit, ledgered delta
list — reflection is necessary, not sufficient (dossier 01).

Upstream identity quirk recorded here (PACKAGE handoff I6): in pinned
upstream, ``import madina.zonal as mz`` binds the INNER ``zonal.py``
MODULE (the star-imports copy the submodule attribute over the package
binding), while ``sys.modules['madina.zonal']`` stays the package.  The
compat namespace deliberately restores the package binding
(compat/madina/__init__.py).  Both give users the same classes; the
module-level namespace delta is ledgered explicitly below.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def surface(mz):
    out = {"module_public": sorted(
        n for n in dir(mz) if not n.startswith("_"))}

    for cls_name in ("Zonal", "Layer", "Layers", "Network"):
        cls = getattr(mz, cls_name)
        members = {}
        for name, obj in inspect.getmembers(cls):
            # parity-relevant dunders are part of the surface (review
            # finding 5: Layers.__getitem__/__setitem__/__contains__/
            # __iter__/__next__/__str__, Zonal.__getitem__)
            if name.startswith("_") and name != "__init__" and name not in (
                    "__getitem__", "__setitem__", "__contains__",
                    "__iter__", "__next__", "__str__"):
                continue
            # only members defined on this class (not inherited object bits)
            if name != "__init__" and not getattr(
                    obj, "__qualname__", "").startswith(cls_name):
                continue
            kind = ("property" if isinstance(obj, property) else
                    "classmethod" if isinstance(obj, classmethod) else
                    "method")
            try:
                members[name] = f"{kind}:{inspect.signature(obj)}"
            except (TypeError, ValueError):
                members[name] = f"{kind}:?"
        out[cls_name] = members

    for fn_name in ("prepare_geometry", "color_gdf", "create_deckGL_map",
                    "node_edge_builder", "efficient_node_insertion"):
        fn = getattr(mz, fn_name, None)
        out.setdefault("module_functions", {})[fn_name] = (
            "?" if fn is None else str(inspect.signature(fn)))

    for const in ("VERSION", "RELEASE_DATE", "DEFAULT_COLORS"):
        out.setdefault("constants", {})[const] = repr(getattr(mz, const, None))
    # CRS constants are Zonal CLASS attributes (not module-level), so
    # read them off the class in both arms
    zc = mz.Zonal
    for const in ("DEFAULT_PROJECTED_CRS", "DEFAULT_GEOGRAPHIC_CRS"):
        out.setdefault("constants", {})[const] = repr(getattr(zc, const, None))
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
            import madina.zonal as mz                   # noqa: PLC0415
            result["identity"] = {
                "attribute_is_package": hasattr(mz, "__path__"),
            }
        else:
            sys.path.insert(0, str(REPO / "src"))
            from urban_network_analysis.compat.madina import zonal as mz  # noqa: PLC0415
            result["identity"] = {
                "attribute_is_package": hasattr(mz, "__path__"),
            }
        result["state"] = surface(mz)
    except Exception:
        result["probe_error"] = traceback.format_exc()[-4000:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=1))
    print(f"reflection: {out}")


if __name__ == "__main__":
    main()
