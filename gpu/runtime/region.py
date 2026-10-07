"""The heterogeneous fold region: K0+K1 device / K2+K3 host.

Composition (fixed by gpu_region.json and designs/06 "keep trusted CPU
math-sensitive coefficients/folds in a heterogeneous region"):

  K0 order-preserving compaction ......... device (bit-exact by construction)
  K1 per-element coefficients ............ device (DOUBLE precision; bit
        identity of device exp() is NOT assumed — gated per cell)
  K2 ordered selection (numba tie law) ... host, exact transcription
  K3 reductions (association trees) ...... host, under the cell-certified tree

FULL service phases (dossier 08 model): prepare -> validate -> probe ->
cell gate -> per tile {cancel barrier; H2D staging; K0 (kept plans skip
K0); K1; D2H} -> K2 -> crop/gather -> knn coefficients -> K3 -> sync ->
handoff -> ledger close.  Cold compile/setup J is a declared phase for the
G11 crossover measurement, never subtracted.

FAIL-CLOSED ORDER (each step refuses before any later step executes):
  1. request validation      -> ValueError/TypeError (typed, no coercion)
  2. driver probe            -> HardwareUnavailable (no CUDA/FP64 device)
  3. cell parity gate        -> ParityGateNotRun (the DURABLE EVIDENCE-BACKED
     gate store does not open the cell: missing store/entry/evidence, or
     evidence bytes changed since the verdict was recorded — G10R F2)
  4. device context port     -> HardwareUnavailable (port not attached)

The device launch ABI itself lands with G11 against real CUDA; this build
ships the fail-closed dispatch order, the bounded tile/staging machinery,
the host tail plans (K2/K3), and the gate harness.  There is deliberately
NO host-twin production path: the trusted CPU route lives in
src/urban_network_analysis and is not invoked from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..backends.gpu import gate_state
from ..kernels.host_twins import fold_host_reference
from .availability import probe_driver
from .buffers import require_c_contiguous
from .errors import HardwareUnavailable, ParityGateNotRun

__all__ = [
    "FoldCell", "FoldRowScalars", "FoldRegionRequest", "CancelFlag",
    "FoldRegionRecord", "execute_fold_region", "plan_tiles",
    "run_host_stage_composition", "FULL_SERVICE_PHASES",
]

# Declared service phases in execution order (dossier 08 FULL service model).
FULL_SERVICE_PHASES = (
    "prepare", "validate", "probe", "cell_gate",
    "h2d", "launch_k0", "launch_k1", "d2h",
    "k2_sort", "knn_crop", "k3_reduce",
    "sync", "handoff",
)

_VALID_DECAYS = ("exponential", "logistic", "linear")


class CancelFlag:
    """Cooperative cancellation, polled ONLY at tile barriers.

    Deliberately not a queue-join (runtime._cancel wedge, task #20): a
    canceled region refuses at the next barrier and never reports a
    truncated success.  The refusal type is errors.RegionCanceled.
    """

    __slots__ = ("_flag",)

    def __init__(self) -> None:
        self._flag = False

    def cancel(self) -> None:
        self._flag = True

    def canceled(self) -> bool:
        return self._flag


@dataclass(frozen=True)
class FoldCell:
    """The workload cell a bit-validation gate certifies (dossier 10 shape)."""

    cell_id: str
    semantic_profile: str = "una_legacy"
    association: str = ""            # certified reduction tree; "" = none yet
    bit_validated: bool = False

    def require_gate(self) -> None:
        if not self.bit_validated:
            raise ParityGateNotRun(
                f"cell {self.cell_id!r}: bit-validation gate has not passed; "
                "device dispatch refused (verification-gated region; see "
                "gpu/kernels/validate_device_bits.py)")
        if not self.association:
            raise ParityGateNotRun(
                f"cell {self.cell_id!r}: no certified reduction association")


@dataclass(frozen=True)
class FoldRowScalars:
    """Per-row fold scalars, exactly the legacy plan's signature scalars."""

    cutoff: float
    gravity_beta: float
    gravity_plateau: float
    gravity_logistic_midpoint: float
    gravity_growth_rate: float
    knn_decay: str
    knn_weights: np.ndarray

    def validate(self) -> None:
        if self.knn_decay not in _VALID_DECAYS:
            raise ValueError(f"knn_decay must be one of {_VALID_DECAYS}, "
                             f"got {self.knn_decay!r}")
        require_c_contiguous("knn_weights", self.knn_weights)
        if self.knn_weights.dtype != np.float64:
            raise ValueError("knn_weights must be float64, got "
                             f"{self.knn_weights.dtype}; no coercion")
        if not np.all(np.isfinite(self.knn_weights)) or np.any(self.knn_weights < 0):
            raise ValueError("knn_weights must be finite and nonnegative")


@dataclass(frozen=True)
class FoldRegionRequest:
    """One row's fold work: per-origin distance rows plus shared scalars.

    ``distance_rows[i]`` is origin i's dense distance-to-destinations row
    (float64).  Tile staging holds ``tile_origins`` rows at a time — never
    a dense O x D tensor (gpu_region.json memory_region).
    """

    distance_rows: Tuple[np.ndarray, ...]
    d_weights: np.ndarray
    scalars: FoldRowScalars
    cell: FoldCell
    tile_origins: int = 64
    staging_budget_bytes: int = 64 * 1024 * 1024
    gate_store: Optional[str] = None   # durable gate-store path; None falls
    # back to UNA_GPU_GATE_STORE, and with neither, every cell reads closed.


class FoldRegionRecord:
    """Outcome record: measured counters only, no parity claims."""

    __slots__ = ("origins", "tiles", "reach", "gravity_exponential",
                 "gravity_logistic", "knn_access", "association",
                 "staging_high_water_bytes", "context_kind",
                 "hardware_validated")

    def __init__(self, *, origins: int, tiles: int, association: str,
                 context_kind: str, reach=None, gravity_exponential=None,
                 gravity_logistic=None, knn_access=None,
                 staging_high_water_bytes: int = 0):
        self.origins = origins
        self.tiles = tiles
        self.reach = reach
        self.gravity_exponential = gravity_exponential
        self.gravity_logistic = gravity_logistic
        self.knn_access = knn_access
        self.association = association
        self.staging_high_water_bytes = staging_high_water_bytes
        self.context_kind = context_kind
        # Constant honesty marker: producing this record NEVER means
        # hardware validation happened; only passed gate-harness evidence
        # (validate_device_bits.py on a real device) can say that.
        self.hardware_validated = False


def validate_request(request: FoldRegionRequest) -> None:
    """Typed request validation; refusal without coercion."""
    if not isinstance(request, FoldRegionRequest):
        raise TypeError("request must be a FoldRegionRequest")
    if request.tile_origins < 1:
        raise ValueError("tile_origins must be >= 1")
    if request.staging_budget_bytes <= 0:
        raise ValueError("staging_budget_bytes must be positive")
    request.scalars.validate()
    require_c_contiguous("d_weights", request.d_weights)
    if request.d_weights.dtype != np.float64:
        raise ValueError(f"d_weights must be float64, got "
                         f"{request.d_weights.dtype}; no coercion")
    if not request.distance_rows:
        raise ValueError("distance_rows must be non-empty")
    for i, row in enumerate(request.distance_rows):
        require_c_contiguous(f"distance_rows[{i}]", row)
        if row.dtype != np.float64:
            raise ValueError(f"distance_rows[{i}] must be float64, got "
                             f"{row.dtype}; no coercion")
        if row.shape[0] != request.d_weights.shape[0]:
            raise ValueError(
                f"distance_rows[{i}] length {row.shape[0]} != d_weights "
                f"length {request.d_weights.shape[0]}; no padding, no "
                "coercion")


def plan_tiles(n_origins: int, tile_origins: int) -> Tuple[Tuple[int, int], ...]:
    """Bounded (start, stop) origin tiles; never a dense O x D materialization."""
    if tile_origins < 1:
        raise ValueError("tile_origins must be >= 1")
    return tuple((s, min(s + tile_origins, n_origins))
                 for s in range(0, n_origins, tile_origins))


def execute_fold_region(request: FoldRegionRequest,
                        context: object = None,
                        cancel: Optional[CancelFlag] = None,
                        ) -> FoldRegionRecord:
    """Run the fold region under the fail-closed order; refuse otherwise.

    The four refusal steps in the module docstring run in order; any refusal
    happens before any compute, transfer, or allocation.  With every gate
    satisfied the device launch ABI is still a G11 deliverable on the
    qualification host, so this build stops there BY DESIGN (never a silent
    fallback, never a simulated execution).
    """
    # 1. request validation
    validate_request(request)
    # 2. driver probe (no CUDA-class FP64 device -> typed refusal)
    probe = probe_driver()
    if not probe.available:
        raise HardwareUnavailable(
            f"CUDA probe refused ({probe.reason}): {probe.detail}")
    # 3. per-cell parity gate — the DURABLE EVIDENCE-BACKED store is the
    # sole authority (G10R F2): the verdict and its evidence bytes are
    # re-read and re-verified from disk on every dispatch; a caller-forged
    # cell object can never open a cell, and tampered evidence closes it.
    request.cell.require_gate()
    state = gate_state(request.cell.cell_id, request.gate_store)
    if not state["open"]:
        raise ParityGateNotRun(
            f"cell {request.cell.cell_id!r}: gate store does not open this "
            f"cell ({state['reason']}; only validate_device_bits.py on a "
            "real device may record a passed cell)")
    certified = state["association"]
    if certified != request.cell.association:
        raise ParityGateNotRun(
            f"cell {request.cell.cell_id!r}: request association "
            f"{request.cell.association!r} != gate-certified "
            f"{certified!r}")
    # 4. device context port
    if context is None:
        raise HardwareUnavailable(
            "no device context port attached (the CUDA context lands with "
            "G11 qualification wiring on the qualification host)")
    raise NotImplementedError(
        "execute_fold_region: device launch wiring is G11's deliverable on "
        "the qualification host; every gate in front of it is implemented "
        "and fail-closed in this build")


def run_host_stage_composition(distance_rows, d_weights, scalars,
                               association: str) -> FoldRegionRecord:
    """Host-side K2/K3 composition over twin inputs (gate/tests ONLY).

    This is the region's host tail (K2 sort -> crop -> knn coefficients ->
    K3 reduce) driven on host twin inputs.  It is NOT a production route
    and MUST NOT be presented as device execution or as engine parity
    without a passed cell gate.
    """
    validate_request(FoldRegionRequest(
        distance_rows=distance_rows, d_weights=d_weights, scalars=scalars,
        cell=FoldCell(cell_id="host-stage-composition",
                      association=association, bit_validated=True)))
    # (G10R F5) The bit_validated=True above is INERT: validate_request
    # never reads request.cell, and this helper performs no dispatch — the
    # flag exists only because FoldRegionRequest requires a FoldCell.  The
    # authority to dispatch lives exclusively in the durable gate store,
    # consulted by execute_fold_region, which this helper does not call.
    n = len(distance_rows)
    reach = np.empty(n, dtype=np.float64)
    ge = np.empty(n, dtype=np.float64)
    gl = np.empty(n, dtype=np.float64)
    ka = np.empty(n, dtype=np.float64)
    for i, row in enumerate(distance_rows):
        r, gexp, glog, kn = fold_host_reference(
            row, d_weights, scalars.cutoff, scalars.gravity_beta,
            scalars.gravity_plateau, scalars.gravity_logistic_midpoint,
            scalars.gravity_growth_rate, scalars.knn_decay,
            scalars.knn_weights, association)
        reach[i], ge[i], gl[i], ka[i] = r, gexp, glog, kn
    return FoldRegionRecord(
        origins=n, tiles=len(plan_tiles(n, 64)), reach=reach,
        gravity_exponential=ge, gravity_logistic=gl, knn_access=ka,
        association=association, staging_high_water_bytes=0,
        context_kind="host-tail-composition (not device execution)")
