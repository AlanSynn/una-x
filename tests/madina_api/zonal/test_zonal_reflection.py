"""Reflection matrix: the facade's public zonal surface must match the
pinned upstream surface — module constants, every class method with its
exact signature, and the module-level namespace modulo an EXPLICIT
ledgered delta list.

Reflection is necessary but not sufficient (dossier 01); the behavior
scenarios live in test_zonal_surface_parity.py.  Ledgered deltas here:

class surfaces
  Zonal.create_map return annotation 'pdk.Deck' (string) instead of the
  pydeck type object — the D4 lazy-pydeck delta (pydeck absent at
  import time in a facade-only environment).

module namespace (users' ``import <pkg>.zonal as mz`` view; underscore
names are filtered from BOTH digests — the pinned upstream's
``_discard_redundant_edges``/``_split_redundant_edges`` bindings live
only in the inner-module view and never enter either list)
  reference-only names: os  (D3: no USE_PYGEOS env mutation at import),
  pdk  (D4: pydeck lazily imported; upstream users see the inner
  zonal.py MODULE through the star-import shadowing quirk — the compat
  namespace exposes the PACKAGE; the compat/madina/__init__ restore is
  the documented identity decision, PACKAGE handoff I6).
  facade-only names: the package namespace's submodule bindings and
  their imports (layer, network, network_utils, utils, zonal, np, nx,
  geo, split, snap, GeoDataFrame, GeoPandaExtractor) plus the two
  network_utils helpers the inner-module view does not re-export
  (tolerance_network_nodes_edges, vectorized_node_edge_builder).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.madina_api

# ---- ledgered deltas (see module docstring) -------------------------
REFERENCE_ONLY_MODULE_NAMES = {
    "os", "pdk",
}
FACADE_ONLY_MODULE_NAMES = {
    "layer", "network", "network_utils", "utils", "zonal",
    "np", "nx", "geo", "split", "snap",
    "GeoDataFrame", "GeoPandaExtractor",
    "tolerance_network_nodes_edges", "vectorized_node_edge_builder",
}
# class-surface deltas: signature-string -> replacement that counts as
# equal (D4 annotation only)
SIGNATURE_ALIASES = {
    "method:(self, layer_list: list = None, save_as: str = None, basemap: bool = False) -> pydeck.bindings.deck.Deck":
        "method:(self, layer_list: list = None, save_as: str = None, basemap: bool = False) -> 'pdk.Deck'",
}


def _normalized(state, is_facade):
    """Return (class_surfaces, constants, module_names) with ledgered
    deltas normalized away so the remainder must be EXACTLY equal."""
    classes = {}
    for cls_name in ("Zonal", "Layer", "Layers", "Network"):
        members = dict(state[cls_name])
        for name, sig in list(members.items()):
            if sig in SIGNATURE_ALIASES:
                members[name] = SIGNATURE_ALIASES[sig]
        classes[cls_name] = members
    names = set(state["module_public"])
    if is_facade:
        names -= FACADE_ONLY_MODULE_NAMES
    else:
        names -= REFERENCE_ONLY_MODULE_NAMES
    return classes, state["constants"], names


def test_reflection_matches_pinned_reference(reflection_runner):
    ref = reflection_runner("reference")
    facade = reflection_runner("facade")
    assert "state" in ref and "state" in facade

    # identity quirk: upstream users see the inner module (not a
    # package); the compat namespace deliberately restores the package
    # (compat/madina/__init__.py) — recorded, ledgered, PACKAGE I6.
    assert ref["identity"]["attribute_is_package"] is False
    assert facade["identity"]["attribute_is_package"] is True

    ref_classes, ref_constants, ref_names = _normalized(ref["state"], False)
    fa_classes, fa_constants, fa_names = _normalized(facade["state"], True)

    assert fa_classes == ref_classes, "class surfaces differ beyond the D4 ledger"
    assert fa_constants == ref_constants, (
        f"constants differ: {ref_constants} != {fa_constants}")
    assert fa_names == ref_names, (
        f"unledgered module-namespace delta: only-ref={sorted(ref_names - fa_names)} "
        f"only-facade={sorted(fa_names - ref_names)}")


def test_version_constants_pinned(reflection_runner):
    """The port tracks pinned madina 0.0.15 (2023-02-16)."""
    ref = reflection_runner("reference")
    assert ref["state"]["constants"]["VERSION"] == repr("0.0.15")
    assert ref["state"]["constants"]["RELEASE_DATE"] == repr("2023-02-16")
    facade = reflection_runner("facade")
    assert facade["state"]["constants"] == ref["state"]["constants"]
