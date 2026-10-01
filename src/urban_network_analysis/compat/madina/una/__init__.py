"""Madina una parity: attributed port of madina.una.

Source: City-Form-Lab/madina @ 8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6,
``src/madina/una/`` (MIT License, Copyright (c) 2023 MIT City Form Lab).
Upstream archived read-only 2026-08-25; this port tracks that exact commit.

DELTA LEDGER (complete list of deviations from the pinned source):

paths.py   VERBATIM copy — zero deltas.  The module is pure
           math/collections/heapq/networkx over the already-bridged
           zonal Network (no pandas import); it touches no GeoArray
           ``.data``, no pandas ``fastpath``, no environment, no
           optional dependency, so the D1-D4 zonal bridge has no
           surface here.

__init__.  Upstream star-imports ``.betweenness`` BEFORE ``.paths``.
           The facade ``betweenness`` module lands with the
           accessibility/betweenness tasks (the access engine trio
           ``get_origin_properties`` / ``one_access`` / ``parallel_access``
           lives there upstream and is needed by ``tools.accessibility``);
           until then this init carries only the ``paths`` star-import and
           ``madina.una`` exposes exactly the upstream ``paths`` names.
           The betweenness line is added by the task that delivers the
           module — no try/except silence.

tools.py   Interim surface: ``alternative_paths`` (the paths-facing
           materialization upstream keeps in tools.py) plus the
           ``turn_o_scope`` / ``path_generator`` names upstream tools.py
           re-imports.  ``validate_zonal_ready`` / ``accessibility`` /
           ``service_area`` join from ``.accessibility`` and
           ``betweenness`` from ``.betweenness`` when those modules land,
           reproducing the upstream tools namespace exactly.

Everything else — signatures, defaults, validation messages, the 1e-5
path-acceptance tolerance, heap/set iteration orders, dict insertion
orders, sort_values("distance") ordering, GeometryCollection assembly
(whole untrimmed origin/destination segments) — is the pinned behavior,
verified bitwise against the reference under the same interpreter.
"""

from .paths import *  # noqa: F401,F403
