"""Madina zonal parity: attributed port of madina.zonal.

Source: City-Form-Lab/madina @ 8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6,
``src/madina/zonal/`` (MIT License, Copyright (c) 2023 MIT City Form Lab).
Upstream archived read-only 2026-08-25; this port tracks that exact commit.

DELTA LEDGER (complete list of deviations from the pinned source; the
reference tree ``.refs/madina_ref`` carries the same first two deltas as
the CONTRACT-hashed dependency bridge ``evidence/contract/madina_bridge.patch``):

D1  network_utils.node_edge_builder: ``geometry_gdf.geometry.values.data``
    -> ``np.asarray(geometry_gdf.geometry.values, dtype=object)``.
    GeoPandas >= 1.0 GeometryArray lost ``.data`` (issue #8 / PR #10;
    packet INC02).  Dependency bridge delta only — produces the same
    coordinate array; not a science change.

D2  every ``pd.Series(..., fastpath=True, ...)`` drops the ``fastpath``
    keyword (34 sites).  The private kwarg was removed in pandas 3.0
    (issue #12 / PR #13 part 1; packet INC01a).  It was a passthrough
    construction flag; series values are unchanged.

D3  zonal.py no longer sets ``os.environ['USE_PYGEOS']`` at import.  The
    pinned comment itself notes this is unneeded from geopandas 1.0; with
    geopandas >= 1.0 the variable is ignored, and a library must not
    mutate the user's environment at import time.

D4  ``pydeck`` is imported lazily inside the visualization entry points
    (``Zonal.create_map`` / ``utils.create_deckGL_map``) and raises an
    actionable ImportError when absent.  Visualization stays optional at
    installation (campaign contract); with pydeck installed the behavior
    is the pinned one.

Everything else — signatures, defaults, validation messages, mutation
targets, IDs, orderings, dtypes, RNG consumption order, known pinned
quirks (Layer.set_style AttributeError = BUG-LAYER-SET-STYLE-ATTRERROR,
dead describe() geo_center branch, np.append no-op in
vectorized_node_edge_builder, light_graph prerequisite of d_graph, ...) —
is a verbatim port.  Registered unsupported pinned paths keep their
pinned failure behavior under the madina parity profile.
"""

from .zonal import *  # noqa: F401,F403
from .layer import *  # noqa: F401,F403
from .network import *  # noqa: F401,F403
from .network_utils import *  # noqa: F401,F403
