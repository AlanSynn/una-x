"""Buffer ownership: typed refusals, immutable leases, bounded staging,
measured ledger.  Device allocation fails closed."""

import numpy as np
import pytest

from gpu.runtime.buffers import (InputLease, ScratchLedger, StagingPool,
                                 device_allocate, require_c_contiguous)
from gpu.runtime.errors import BudgetExceeded, HardwareUnavailable


def test_require_c_contiguous_refuses_and_names():
    with pytest.raises(TypeError, match="must be a numpy array"):
        require_c_contiguous("x", [1.0, 2.0])
    with pytest.raises(ValueError, match="ndim=2"):
        require_c_contiguous("x", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="C-contiguous"):
        require_c_contiguous("x", np.ones(6)[::2])
    # zero-length buffers are trivially contiguous (N00 law)
    require_c_contiguous("x", np.zeros(0))


def test_input_lease_refuses_wrong_dtype_without_coercion():
    with pytest.raises(ValueError, match="no coercion"):
        InputLease("d_weights", np.array([1, 2], dtype=np.int32),
                   dtype=np.float64)


def test_input_lease_view_is_readonly_and_release_closes_it():
    original = np.array([1.0, 2.0, 3.0])
    lease = InputLease("dist", original, dtype=np.float64)
    view = lease.readonly_view()
    assert not view.flags["WRITEABLE"]
    with pytest.raises(ValueError):
        view[0] = 9.0
    lease.release()
    with pytest.raises(RuntimeError, match="released"):
        lease.readonly_view()


def test_staging_pool_bounds_and_high_water():
    pool = StagingPool(budget_bytes=1000)
    a = pool.allocate("a", (10,))                  # 80 B
    b = pool.allocate("b", (10,), dtype=np.int64)  # 80 B
    assert pool.ledger.outstanding_bytes == 160
    assert pool.ledger.high_water_bytes == 160
    with pytest.raises(BudgetExceeded, match="remaining staging budget"):
        pool.allocate("c", (120,))                 # 960 B; 160+960 > 1000
    pool.release(a)
    assert pool.ledger.outstanding_bytes == 80
    pool.release(b)
    assert pool.ledger.outstanding_bytes == 0
    assert pool.ledger.high_water_bytes == 160  # measured, not reset


def test_pool_refuses_nonpositive_budget():
    with pytest.raises(ValueError, match="positive"):
        StagingPool(budget_bytes=0)


def test_ledger_release_without_allocation_is_a_defect():
    ledger = ScratchLedger()
    with pytest.raises(RuntimeError, match="underflow"):
        ledger.released(10)


def test_device_allocation_fails_closed():
    with pytest.raises(HardwareUnavailable, match="fail-closed"):
        device_allocate(1024, context=None)
