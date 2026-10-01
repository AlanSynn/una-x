"""Madina una parity: attributed port of madina.una.

Source: City-Form-Lab/madina @ 8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6,
``src/madina/una/`` (MIT License, Copyright (c) 2023 MIT City Form Lab).
Upstream archived read-only 2026-08-25; this port tracks that exact commit.

DELTA LEDGER (complete list of deviations from the pinned source):

betweenness.py  INTERIM: carries the verbatim module import block and the
           access-engine trio (``get_origin_properties`` / ``one_access`` /
           ``parallel_access``, upstream lines 1016-1323) delivered by the
           accessibility task.  The five betweenness functions
           (``parallel_betweenness``, ``one_betweenness_2``,
           ``clockwiseangle_and_distance``, ``betweenness_exposure``,
           ``paralell_betweenness_exposure``, upstream lines 30-1015) land
           with MADINA_FLOW extending this same file; until then the
           ``una`` namespace is missing those five names vs upstream.

paths.py   VERBATIM copy — zero deltas.  The module is pure
           math/collections/heapq/networkx over the already-bridged
           zonal Network (no pandas import); it touches no GeoArray
           ``.data``, no pandas ``fastpath``, no environment, no
           optional dependency, so the D1-D4 zonal bridge has no
           surface here.

__init__.  Upstream star-imports ``.betweenness`` BEFORE ``.paths``;
           this init does the same.  The star-import copies the
           betweenness module's public names (trio + module-level
           imports) into ``una``, matching upstream semantics.

tools.py   INTERIM surface: ``validate_zonal_ready`` / ``accessibility`` /
           ``service_area`` / ``alternative_paths`` verbatim (the
           accessibility task delivered the first three), plus the
           ``turn_o_scope`` / ``path_generator`` / ``parallel_access``
           names upstream tools.py re-imports (INTERIM: the facade
           re-imports only ``parallel_access`` until MADINA_FLOW;
           ``paralell_betweenness_exposure`` is a fourth upstream
           re-import that lands then).  ``betweenness`` joins from
           ``.betweenness`` with MADINA_FLOW, reproducing the upstream
           tools namespace exactly.

Everything else — signatures, defaults, validation messages, the 1e-5
path-acceptance tolerance, heap/set iteration orders, dict insertion
orders, sort_values("distance") ordering, GeometryCollection assembly
(whole untrimmed origin/destination segments), the UNSEEDED
``sample(frac=1)`` origin shuffle, and one_access's exception swallowing
— is the pinned behavior, verified bitwise against the reference under
the same interpreter.
"""

from .betweenness import *  # noqa: F401,F403
from .paths import *  # noqa: F401,F403
