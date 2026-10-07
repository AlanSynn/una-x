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
  5. device execution        -> K0/K1 on device via the attached port
     (DeviceRuntime), K2/K3 host tail under the gate-certified tree.

There is deliberately NO host-twin production path: the trusted CPU route
lives in src/urban_network_analysis and is not invoked from here.  Records
carry measured counters only and never claim parity (hardware_validated is
False by construction — parity is the gate harness's durable verdict).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..backends.gpu import gate_state
from ..kernels.host_twins import (argsort_tie_law, fold_host_reference,
                                  reduce_with)
from .availability import probe_driver
from .buffers import require_c_contiguous
from .errors import (HardwareUnavailable, ParityGateNotRun, RegionCanceled)

__all__ = [
    "FoldCell", "FoldRowScalars", "FoldRegionRequest", "CancelFlag",
    "FoldRegionRecord", "execute_fold_region", "plan_tiles",
    "run_host_stage_composition", "run_device_stage_composition",
    "FULL_SERVICE_PHASES",
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

    The refusal steps in the module docstring run in order; any refusal
    happens before any compute, transfer, or allocation.  ``context`` is
    the device port (gpu.kernels.device_exec.DeviceRuntime on the
    qualification host, with the committed .cu compiled and loaded); on a
    gate-open cell the region executes K0/K1 on the device and K2/K3 on
    the host under the certified association.  Never a silent fallback,
    never a simulated execution.
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
            "no device context port attached (attach a DeviceRuntime from "
            "gpu.kernels.device_exec on the qualification host)")
    # 5. device execution (G11 wiring): K0/K1 on the device through the
    # attached port, K2/K3 host tail under the gate-certified association.
    return run_device_stage_composition(request, context, certified, cancel)


_BLOCK = 256   # mirrors UNA_G10_BLOCK_THREADS in fold_kernels.cu


def run_device_stage_composition(request: FoldRegionRequest, context,
                                 certified: str,
                                 cancel: Optional[CancelFlag] = None,
                                 ) -> FoldRegionRecord:
    """Device K0/K1 + host K2/K3 composition through a live device port.

    ``context`` must expose the gate-harness device surface (``identity``,
    ``h2d``, ``d2h``, ``launch``, ``sync``) — on the qualification host,
    gpu.kernels.device_exec.DeviceRuntime with the committed .cu compiled
    and loaded.  The durable gate store has already certified the cell
    (fail-closed steps 1-4); this function makes NO parity claims of its
    own: parity is the gate's verdict, so ``hardware_validated`` stays
    False on the record by construction.  Cancellation is polled only at
    tile barriers; a canceled region refuses (RegionCanceled), never
    reports a truncated success.
    """
    for attr in ("identity", "h2d", "d2h", "launch", "sync"):
        if not hasattr(context, attr):
            raise TypeError(
                f"device context port lacks {attr!r}; attach a "
                "gpu.kernels.device_exec.DeviceRuntime")
    n = len(request.distance_rows)
    reach = np.empty(n, dtype=np.float64)
    ge = np.empty(n, dtype=np.float64)
    gl = np.empty(n, dtype=np.float64)
    ka = np.empty(n, dtype=np.float64)
    tiles = plan_tiles(n, request.tile_origins)
    sc = request.scalars
    high_water = 0
    for (s, e) in tiles:
        if cancel is not None and cancel.canceled():
            raise RegionCanceled(
                f"region canceled at tile barrier [{s}, {e}); no partial "
                "success is reported")
        for i in range(s, e):
            (reach[i], ge[i], gl[i], ka[i],
             row_bytes) = _device_fold_row(
                context, request.distance_rows[i], request.d_weights, sc,
                certified)
            high_water = max(high_water, row_bytes)
    identity = context.identity
    return FoldRegionRecord(
        origins=n, tiles=len(tiles), reach=reach, gravity_exponential=ge,
        gravity_logistic=gl, knn_access=ka, association=certified,
        staging_high_water_bytes=high_water,
        context_kind=(f"device:{identity.name} (cell "
                      f"{request.cell.cell_id} open via durable gate store)"))


def _device_fold_row(ctx, row: np.ndarray, weights: np.ndarray,
                     sc: FoldRowScalars, certified: str):
    """One origin row: K0/K1 on device, K2 sort host, knn on device, K3
    host reduce under ``certified``.  Returns (reach, ge, gl, ka, bytes)."""
    d = np.ascontiguousarray(row, dtype=np.float64).reshape(1, -1)
    width = int(d.shape[1])
    idx_like = np.empty(d.shape, dtype=np.int64)
    cnt_like = np.empty(1, dtype=np.uint64)
    dev_dist = ctx.h2d(d)
    dev_idx = ctx.h2d(idx_like)      # capacity buffer; kernel fills kept
    dev_cnt = ctx.h2d(cnt_like)
    ctx.launch("una_g10_fold_compact", (1, 1, 1), (_BLOCK, 1, 1),
               (dev_dist, ctypes.c_longlong(width),
                ctypes.c_double(float(sc.cutoff)), dev_idx, dev_cnt))
    ctx.sync()
    idx2d = ctx.d2h(dev_idx, idx_like)
    cnt = int(ctx.d2h(dev_cnt, cnt_like)[0])
    idx = idx2d[0][:cnt]
    dn = np.ascontiguousarray(row[idx], dtype=np.float64)
    wn = np.ascontiguousarray(weights[idx], dtype=np.float64)
    m = int(dn.shape[0])

    def _elem(kname, params, count):
        if count == 0:
            return np.empty(0, dtype=np.float64)
        dev_out = ctx.h2d(np.empty(count, dtype=np.float64))
        grid = ((count + _BLOCK - 1) // _BLOCK, 1, 1)
        ctx.launch(kname, grid, (_BLOCK, 1, 1), tuple(params) + (dev_out,))
        ctx.sync()
        return ctx.d2h(dev_out, np.empty(count, dtype=np.float64))

    dev_dn = ctx.h2d(dn) if m else None
    dev_wn = ctx.h2d(wn) if m else None
    c_exp = _elem("una_g10_fold_coefficients_exp",
                  (dev_dn, dev_wn, ctypes.c_longlong(m),
                   ctypes.c_double(float(sc.gravity_beta)),
                   ctypes.c_double(float(sc.gravity_plateau))), m)
    c_log = _elem("una_g10_fold_coefficients_log",
                  (dev_dn, dev_wn, ctypes.c_longlong(m),
                   ctypes.c_double(float(sc.gravity_growth_rate)),
                   ctypes.c_double(float(sc.gravity_plateau)),
                   ctypes.c_double(float(sc.gravity_logistic_midpoint))), m)

    # K3 host reduces under the gate-certified tree, mirroring
    # fold_host_reference's composition order exactly (reach first, then
    # the coefficient reduces; knn special-cases count 0 to constant 0).
    reach = reduce_with(certified, wn)
    ge = reduce_with(certified, c_exp)
    gl = reduce_with(certified, c_log)

    knn = min(m, int(sc.knn_weights.shape[0]))
    if knn == 0:
        ka = np.float64(0.0)
    else:
        order = argsort_tie_law(dn)
        dn_s = np.ascontiguousarray(dn[order[:knn]])
        wn_s = np.ascontiguousarray(wn[order[:knn]])
        kn = np.ascontiguousarray(sc.knn_weights[:knn])
        dev_s, dev_ws, dev_kn = ctx.h2d(dn_s), ctx.h2d(wn_s), ctx.h2d(kn)
        if sc.knn_decay == "exponential":
            c_knn = _elem("una_g10_fold_knn_coeffs_exp",
                          (dev_s, dev_ws, dev_kn, ctypes.c_longlong(knn),
                           ctypes.c_double(float(sc.gravity_beta)),
                           ctypes.c_double(float(sc.gravity_plateau))), knn)
        elif sc.knn_decay == "logistic":
            c_knn = _elem("una_g10_fold_knn_coeffs_log",
                          (dev_s, dev_ws, dev_kn, ctypes.c_longlong(knn),
                           ctypes.c_double(float(sc.gravity_growth_rate)),
                           ctypes.c_double(float(sc.gravity_plateau)),
                           ctypes.c_double(float(sc.gravity_logistic_midpoint))),
                          knn)
        else:
            c_knn = _elem("una_g10_fold_knn_coeffs_linear",
                          (dev_ws, dev_kn, ctypes.c_longlong(knn)), knn)
        ka = reduce_with(certified, c_knn)

    row_bytes = (d.nbytes + idx_like.nbytes + cnt_like.nbytes
                 + 2 * m * 8 + 5 * max(m, 1) * 8)
    return reach, ge, gl, ka, row_bytes


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
