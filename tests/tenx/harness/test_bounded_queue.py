"""H00 negative tests: byte-bounded pending-result descriptors."""
from __future__ import annotations

import pytest

from benchmarks.una_10x import BoundedPendingBytes, WindowInvalid


def test_submit_under_cap_and_peak_ledger():
    q = BoundedPendingBytes(cap_bytes=100)
    q.submit(40)
    q.submit(40)
    assert q.peak_bytes == 80 and q.pending_bytes == 80
    q.complete(40)
    assert q.pending_bytes == 40
    assert q.peak_bytes == 80   # peak is retained for the ledger audit
    q.submit(60)                # exactly fills the cap
    assert q.peak_bytes == 100


def test_exceeding_the_cap_fails_instead_of_growing():
    q = BoundedPendingBytes(cap_bytes=100)
    q.submit(60)
    with pytest.raises(WindowInvalid, match="exceed cap"):
        q.submit(41)
    assert q.pending_bytes == 60   # rejected submit leaves state intact
    q.submit(40)
    assert q.peak_bytes == 100


def test_negative_or_inconsistent_descriptors_are_invalid():
    q = BoundedPendingBytes(cap_bytes=10)
    with pytest.raises(WindowInvalid, match="negative"):
        q.submit(-1)
    with pytest.raises(WindowInvalid, match="inconsistent"):
        q.complete(5)
    q.submit(10)
    with pytest.raises(WindowInvalid, match="inconsistent"):
        q.complete(11)


def test_nonpositive_cap_is_refused():
    with pytest.raises(WindowInvalid, match="positive"):
        BoundedPendingBytes(cap_bytes=0)
