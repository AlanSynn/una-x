"""Madina una parity: attributed port of madina.una.

Source: City-Form-Lab/madina @ 8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6,
``src/madina/una/`` (MIT License, Copyright (c) 2023 MIT City Form Lab).
Upstream archived read-only 2026-08-25; this port tracks that exact commit.

DELTA LEDGER (complete list of deviations from the pinned source):

betweenness.py  VERBATIM module — the facade file is a module-docstring
           header followed by the pinned upstream file byte for byte
           (delivered across MADINA_ACCESS: the access-engine trio
           ``get_origin_properties`` / ``one_access`` /
           ``parallel_access``; and MADINA_FLOW:
           ``parallel_betweenness``, ``one_betweenness_2``,
           ``clockwiseangle_and_distance``, ``betweenness_exposure``,
           ``paralell_betweenness_exposure``).  The header docstring is
           the only delta (upstream has no module docstring); it
           records the pinned betweenness quirks (in-place
           ``decay_method`` reassignment with a first-path decay
           factor, the 0.01 weight clamp, per-destination exception
           swallowing, the closest_destination origin-stats NameError,
           the uniform min-distance exponent decay, the all-zero-gravity
           ``task_done`` skip, the hardcoded 'cocentric-chunks'
           chunking, the 'decayed_mean_hazzad' typo column, the
           UNSEEDED ``sample(frac=1)`` origin shuffle, the low-level
           engine's num_cores>1 ``array_split`` all-zero defect, the
           one_betweenness_2 attribution degeneracy over trimmed paths,
           and the exposure engine's queue-partition nondeterminism at
           num_cores>1).

paths.py   VERBATIM copy — zero deltas.  The module is pure
           math/collections/heapq/networkx over the already-bridged
           zonal Network (no pandas import); it touches no GeoArray
           ``.data``, no pandas ``fastpath``, no environment, no
           optional dependency, so the D1-D4 zonal bridge has no
           surface here.

__init__.  Upstream star-imports ``.betweenness`` BEFORE ``.paths``;
           this init does the same.  The star-import copies the
           betweenness module's public names (all eight functions +
           module-level imports) into ``una``, matching upstream
           semantics.

tools.py   VERBATIM module — the facade file is a module-docstring
           header followed by the pinned upstream file byte for byte
           (all five public functions:
           ``validate_zonal_ready`` / ``accessibility`` /
           ``service_area`` / ``alternative_paths`` /
           ``betweenness``, the last delivered by MADINA_FLOW).
           The header docstring is the only delta.

Everything else — signatures, defaults, validation messages, the 1e-5
path-acceptance tolerance, heap/set iteration orders, dict insertion
orders, sort_values("distance") ordering, GeometryCollection assembly
(whole untrimmed origin/destination segments), the UNSEEDED
``sample(frac=1)`` origin shuffles, one_access's exception swallowing,
betweenness's ``knn_weight``/``knn_plateau`` network-attribute clobber
on every call, the duplicated ``save_gravity_as`` type check, and
keep_diagnostics joining time/memory diagnostic columns — is the pinned
behavior, verified bitwise against the reference under the same
interpreter.
"""

from .betweenness import *  # noqa: F401,F403
from .paths import *  # noqa: F401,F403
