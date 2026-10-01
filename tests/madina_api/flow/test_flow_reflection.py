"""Reflection matrix: the facade's public una surface must match the
pinned upstream surface — every betweenness-module and tools function
with its exact signature, star-export identity, and the module
namespaces.

Reflection is necessary but not sufficient (dossier 01); the behavior
scenarios live in test_flow_surface_parity.py.  At MADINA_FLOW delivery
(see compat una/__init__ ledger) the facade una package is COMPLETE —
the parity ledgers in the access/paths suites are EMPTY, and this probe
carries no ledger at all: any name delta at all fails.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.madina_api


def _annotation_aliases(sig):
    """Annotation module qualifiers differ between the trees (the
    verbatim port binds ``Zonal`` from the compat package, upstream from
    madina.zonal); strip the qualifier on BOTH sides so the remainder
    must be exactly equal."""
    # longest first: the compat qualifier CONTAINS the upstream one as
    # a substring
    for prefix in ("urban_network_analysis.compat.madina.zonal.network.",
                   "urban_network_analysis.compat.madina.zonal.zonal.",
                   "madina.zonal.network.", "madina.zonal.zonal."):
        sig = sig.replace(prefix, "")
    return sig


def test_reflection_matches_pinned_reference(reflection_runner):
    ref = reflection_runner("reference")
    facade = reflection_runner("facade")
    assert "surface" in ref and "surface" in facade
    assert ref["identity"]["package_has_path"] is True
    assert facade["identity"]["package_has_path"] is True

    rs, fs = ref["surface"], facade["surface"]
    # no ledger at the completed delivery: any delta at all fails
    assert set(fs["una_public"]) == set(rs["una_public"]), (
        f"unledgered una delta: "
        f"only-ref={sorted(set(rs['una_public']) - set(fs['una_public']))} "
        f"only-facade={sorted(set(fs['una_public']) - set(rs['una_public']))}")
    assert set(fs["tools_public"]) == set(rs["tools_public"]), (
        f"unledgered tools delta: "
        f"only-ref={sorted(set(rs['tools_public']) - set(fs['tools_public']))} "
        f"only-facade={sorted(set(fs['tools_public']) - set(rs['tools_public']))}")
    assert set(fs["betweenness_public"]) == set(rs["betweenness_public"]), (
        f"unledgered betweenness-module delta: only-ref="
        f"{sorted(set(rs['betweenness_public']) - set(fs['betweenness_public']))} "
        f"only-facade="
        f"{sorted(set(fs['betweenness_public']) - set(rs['betweenness_public']))}")

    # delivered surface is a verbatim port: EXACT equality modulo the
    # documented annotation aliases
    for key in ("betweenness_functions", "tools_functions"):
        ref_sigs = {k: _annotation_aliases(v) for k, v in rs[key].items()}
        fa_sigs = {k: _annotation_aliases(v) for k, v in fs[key].items()}
        assert fa_sigs == ref_sigs, f"{key} differ after alias stripping"
    # the betweenness signature set is the FULL engine surface
    assert set(rs["betweenness_functions"]) == {
        "parallel_betweenness", "one_betweenness_2",
        "clockwiseangle_and_distance", "betweenness_exposure",
        "paralell_betweenness_exposure", "get_origin_properties",
        "one_access", "parallel_access"}
    assert set(rs["tools_functions"]) == {
        "validate_zonal_ready", "accessibility", "service_area",
        "alternative_paths", "betweenness",
        "paralell_betweenness_exposure"}


def test_star_exports_and_submodule_identity(reflection_runner):
    """una.<fn> is the module attribute for every delivered function
    (upstream star-import semantics), and the submodule bindings
    resolve."""
    rs = reflection_runner("reference")["surface"]
    fs = reflection_runner("facade")["surface"]
    assert rs["star_exports_are_betweenness_functions"] is True
    assert fs["star_exports_are_betweenness_functions"] is True
    # una-level bindings for the tools names: identical on both arms,
    # and split exactly by name collision with the betweenness module —
    # the tools-only names are NOT bound at the una level (the tools
    # surface is reachable only through the submodule), while
    # `betweenness` (submodule binding) and
    # `paralell_betweenness_exposure` (star export) ARE.
    assert rs["una_level_tools_bindings"] == fs[
        "una_level_tools_bindings"]
    tools_bindings = fs["una_level_tools_bindings"]
    assert tools_bindings["validate_zonal_ready"] is False
    assert tools_bindings["accessibility"] is False
    assert tools_bindings["service_area"] is False
    assert tools_bindings["alternative_paths"] is False
    assert tools_bindings["betweenness"] is True
    assert tools_bindings["paralell_betweenness_exposure"] is True
    assert rs["una_betweenness_is_betweenness"] is True
    assert fs["una_betweenness_is_betweenness"] is True
    assert rs["una_tools_is_tools"] is True
    assert fs["una_tools_is_tools"] is True
