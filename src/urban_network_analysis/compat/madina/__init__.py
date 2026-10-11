"""Madina parity namespace (urban_network_analysis.compat.madina).

Attributed port of City-Form-Lab/madina @ 8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6
(MIT License, Copyright (c) 2023 MIT City Form Lab / Atallah Albashir and
contributors).  This namespace, and the separately installed drop-in
``madina`` shim distribution built on it, preserve upstream attribution;
see the zonal and una package docstrings for the delta ledgers against
the pinned source.

Mirrors the pinned top-level package layout::

    from .zonal import *          # madina.zonal
    from .una import *            # madina.una — the engine/path surface
                                  # (MADINA_ACCESS/PATHS/FLOW/WORKFLOWS):
                                  # betweenness.py + paths.py public
                                  # names, exactly as upstream's una
                                  # star-imports.  The una.tools /
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

Import side effect (upstream parity, not a delta): importing this
namespace now carries the pinned una-chain environment write
``os.environ['USE_PYGEOS'] = '0'`` (betweenness.py:3, workflows.py:6 —
carried verbatim; upstream ``import madina`` writes the same value at
zonal.py:3, whose removal at the zonal site is the D3 ledger)."""

from .zonal import *  # noqa: F401,F403
from .una import *  # noqa: F401,F403

# The star import copies the zonal package's submodule attribute ``zonal``
# (= the zonal.py module) into this namespace, shadowing the package
# binding; ``import urban_network_analysis.compat.madina.zonal.<sub>``
# resolves submodules by attribute traversal, so restore the package.
# The una star import has NO analogous shadowing: the una package's
# namespace carries no ``una``/``zonal`` name, so ``una`` stays the una
# package (upstream identity — pinned in
# tests/madina_api/zonal/test_top_level_reflection.py).
import sys as _sys
globals()['zonal'] = _sys.modules[__name__ + '.zonal']
