"""Light spawn entry point for UNA batch row workers (dossier 05).

``multiprocessing`` "spawn" imports the target callable's module in the
fresh interpreter BEFORE any code of ours runs.  A target inside
``urban_network_analysis`` would transitively import the package chain
(Settings -> numpy, which initializes the OpenBLAS/MKL thread pools from
the environment at library load) — so the worker's thread budgets could
no longer be applied.  This module deliberately imports only the
stdlib: it applies the budget environment FIRST, then hands off to the
real worker loop, whose first heavy import (numpy -> numba -> the
analysis stack) therefore happens inside the budget.
"""

from __future__ import annotations

import os

__all__ = ["pool_main"]


def pool_main(inbox, outbox, ready, threads_per_worker: int = 1,
              verbosity: int = 0) -> None:
    n = str(max(1, int(threads_per_worker)))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = n
    os.environ["_UNA_BATCH_IN_WORKER"] = "1"
    from urban_network_analysis.batch.worker import pool_main as _real
    _real(inbox, outbox, ready, threads_per_worker, verbosity)
