"""Fixtures for the C20 residency suite (campaign una_10x).

The residency layer is pure cache-package code: no engine imports, no
numba.  The producer digest fixture is the sha256 of the real trace
module's bytes — the same function of the same bytes the residency
binds into keys and load-time identity checks.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

TRACE_MODULE = REPO_SRC / "urban_network_analysis" / "Engines" \
    / "_trace_access_scratch.py"


def _trace_schema_version() -> str:
    """The engine's REAL record schema string, read from the module's
    own bytes (C20-MAJOR-2 repair: fixtures invent nothing — the
    meta carries the true value the build_trace record carries)."""
    import re
    m = re.search(r'TRACE_SCHEMA_VERSION\s*=\s*"([^"]+)"',
                  TRACE_MODULE.read_text(encoding="utf-8"))
    assert m, "TRACE_SCHEMA_VERSION literal not found"
    return m.group(1)


@pytest.fixture(scope="session")
def producer_sha():
    return hashlib.sha256(TRACE_MODULE.read_bytes()).hexdigest()


@pytest.fixture
def disk_store(tmp_path):
    from urban_network_analysis.Execution import CacheOptions
    from urban_network_analysis.cache.store import ContentStore
    return ContentStore(CacheOptions(
        mode="disk", directory=str(tmp_path / "store")))


@pytest.fixture
def mem_store():
    from urban_network_analysis.Execution import CacheOptions
    from urban_network_analysis.cache.store import ContentStore
    return ContentStore(CacheOptions(mode="memory",
                                     max_memory_bytes=1 << 22))


@pytest.fixture
def residency(disk_store, producer_sha):
    from urban_network_analysis.cache.trace_residency import (
        TraceResidency)
    return TraceResidency(disk_store, max_resident_bytes=1 << 20,
                          producer_sha256=producer_sha)


def packed_triple(nnz, nrows=2, seed=5):
    """A VALID packed trace triple (dossier-04 layout law): int64
    indptr[0]==0 ending at nnz, int64 destination ids STRICTLY
    ascending within every slice (the engine's own replay gate
    refuses diff <= 0 — duplicates included), float64 distances.
    The ids are drawn WITHOUT replacement and sorted; the earlier
    draft's with-replacement draw silently produced duplicate ids at
    larger nnz (birthday collisions), a layout violation the publish
    law now refuses."""
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(1 << 20, size=nnz,
                             replace=False)).astype(np.int64)
    dist = rng.uniform(0.0, 10.0, nnz)
    cuts = np.sort(rng.choice(np.arange(1, nnz), nrows - 1,
                              replace=False)) if nnz >= nrows else \
        np.array([], np.int64)
    indptr = np.concatenate(([0], cuts, [nnz])).astype(np.int64)
    return indptr, ids, dist


def meta_for(indptr, ids, trace_id="tid-1"):
    return {"trace_id": trace_id,
            "trace_schema_version": _trace_schema_version(),
            "dest_count": int(ids.max()) + 1 if ids.shape[0] else 0,
            "cutoff_bits": float(8.0).hex(), "cutoff_dtype": "float64",
            "reach_dtype": "float64", "metric_dtype": "float64"}
