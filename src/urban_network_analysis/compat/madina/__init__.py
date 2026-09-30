"""Madina parity namespace (urban_network_analysis.compat.madina).

Attributed port of City-Form-Lab/madina @ 8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6
(MIT License, Copyright (c) 2023 MIT City Form Lab / Atallah Albashir and
contributors).  This namespace, and the separately installed drop-in
``madina`` shim distribution built on it, preserve upstream attribution;
see the zonal package docstring for the delta ledger against the pinned
source.

Mirrors the pinned top-level package layout::

    from .zonal import *          # madina.zonal
    from .una import *            # lands with the MADINA_* tasks (una.tools,
                                  # una.paths, una.betweenness, una.workflows);
                                  # not present yet, not silently stubbed.
"""

from .zonal import *  # noqa: F401,F403

# The star import copies the zonal package's submodule attribute ``zonal``
# (= the zonal.py module) into this namespace, shadowing the package
# binding; ``import urban_network_analysis.compat.madina.zonal.<sub>``
# resolves submodules by attribute traversal, so restore the package.
import sys as _sys
globals()['zonal'] = _sys.modules[__name__ + '.zonal']
