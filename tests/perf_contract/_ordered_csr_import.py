"""Load Engines/_ordered_csr.py directly by file path.

Keeps L0 tests independent of the package __init__ (which pulls in the full
engine/GIS stack) while still testing the exact file that ships in the
candidate tree. Candidate source root can be overridden via
UNA_CANDIDATE_SRC for reviewer runs against a different checkout.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def get_try_build_ordered_csr():
    default_root = Path(__file__).resolve().parents[2] / "src"
    src_root = Path(os.environ.get("UNA_CANDIDATE_SRC", default_root))
    mod_path = src_root / "urban_network_analysis" / "Engines" / "_ordered_csr.py"
    spec = importlib.util.spec_from_file_location("_ordered_csr_under_test", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._try_build_ordered_csr
