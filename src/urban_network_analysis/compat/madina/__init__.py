"""Madina parity namespace (urban_network_analysis.compat.madina).

Attributed port of City-Form-Lab/madina @ 8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6
(MIT License, Copyright (c) 2023 MIT City Form Lab / Atallah Albashir and
contributors).  This namespace, and the separately installed drop-in
``madina`` shim distribution built on it, preserve upstream attribution;
see the zonal and una package docstrings for the delta ledgers against
the pinned source.

Mirrors the pinned top-level package layout::

    from .zonal import *          # madina.zonal — imported eagerly
    from .una import *            # madina.una — the engine/path surface
                                  # (MADINA_ACCESS/PATHS/FLOW/WORKFLOWS):
                                  # betweenness.py + paths.py public
                                  # names, exactly as upstream's una
                                  # star-imports.  Materialized
                                  # LAZILY on first una-surface access
                                  # (PEP 562 __getattr__/__dir__, the
                                  # D3-LAZY delta below); the una.tools /
                                  # una.workflows helpers stay
                                  # una-subpath-only, exactly as
                                  # upstream, whose una/__init__ stars
                                  # only betweenness and paths: reach
                                  # them through the imported una
                                  # package after an explicit
                                  # ``import ...madina.una.tools`` —
                                  # plain ``import madina`` does not
                                  # bind ``una.tools`` upstream either.

Top-level namespace delta vs pinned upstream (API_REVIEW R1 closure,
mda-20261001T1215Z): ``pdk`` only — the D4 lazy-pydeck ledger (pydeck
absent from the facade import chain); ``os`` returns via the una star
import, whose pinned betweenness.py imports os at module top.  The
documented identity decision below (PACKAGE handoff I6) keeps ``zonal``
as the package where upstream users see the inner zonal.py module.

Import side effect (upstream parity, LAZY trigger — D3-LAZY
cd3-20261012T0030Z, the one deliberate delta vs the eager R1 layout at
de2872f): importing THIS namespace alone no longer executes the una
chain, so it no longer writes the environment.  The pinned una-chain
environment write ``os.environ['USE_PYGEOS'] = '0'`` (betweenness.py:105,
workflows.py:112 — carried verbatim; upstream ``import madina`` writes
the same value at zonal.py:3, whose removal at the zonal site is the D3
ledger) now fires at the FIRST load of the una surface — ``m.una``, any
of the 61 reexported names, ``from ...compat.madina import <una name>``,
``import ...compat.madina.una``, or ``dir(m)`` — exactly once, at its
upstream site.  After that first load the namespace is byte-identical to
the eager layout it replaces (star-copy semantics, una names winning the
zonal star per upstream __init__ order, the I6 zonal package restore
re-applied last).  No ``__all__`` is defined, mirroring upstream; a
star-import of this namespace materializes the surface first (the
machinery's ``__all__`` lookup routes through the lazy trigger) and then
lands the full public dict.  This split exists so ``import
...compat.madina.zonal`` — the D3 facade import pinned by
tests/platform_geometry — stays environment-pristine; the D3 zonal-site
removal itself is unchanged."""

from .zonal import *  # noqa: F401,F403

import importlib as _importlib
import sys as _sys

_una_loaded = False


def _load_una_star():
    """Materialize the ``from .una import *`` reexport, exactly once.

    Imports the una package (which itself executes the pinned upstream
    una chain, including the betweenness.py:105 USE_PYGEOS write), then
    copies its public namespace over this module dict with star-import
    semantics — una's names win over the zonal star's, matching the
    upstream __init__ order the eager layout followed — and re-applies
    the zonal package restore so the I6 identity decision keeps winning
    last even in the (pinned-absent, live-verified by
    tests/madina_api/zonal/test_top_level_reflection.py) case of una
    ever carrying a ``zonal`` name.
    """
    global _una_loaded
    una = _importlib.import_module('.una', __name__)
    if not _una_loaded:
        _globals = globals()
        for _name, _value in list(vars(una).items()):
            if not _name.startswith('_'):
                _globals[_name] = _value
        _globals['zonal'] = _sys.modules[__name__ + '.zonal']
        _una_loaded = True
    return una


def __getattr__(name):
    """PEP 562 lazy trigger: the FIRST attribute resolution on this
    namespace (``m.una``, an engine/path name, ``from ... import <name>``)
    imports the una surface once and serves the name from the
    materialized dict.  Unknown names raise AttributeError exactly like
    an eager namespace."""
    _load_una_star()
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    """Live full surface: dir() materializes the una reexport through the
    same lazy trigger, so dir()-based parity (the R1 top-reflection
    digest) sees the complete namespace without a hardcoded name list."""
    _load_una_star()
    return sorted(n for n in globals() if not n.startswith('_'))


# The zonal star import copies the zonal package's submodule attribute
# ``zonal`` (= the zonal.py module) into this namespace, shadowing the
# package binding; ``import urban_network_analysis.compat.madina.zonal.<sub>``
# resolves submodules by attribute traversal, so restore the package.
# The una star materialization has NO analogous shadowing: the una
# package's namespace carries no ``una``/``zonal`` name, so ``una`` stays
# the una package and ``zonal`` the package (re-asserted after the lazy
# copy in _load_una_star; upstream identity — pinned in
# tests/madina_api/zonal/test_top_level_reflection.py).
globals()['zonal'] = _sys.modules[__name__ + '.zonal']
