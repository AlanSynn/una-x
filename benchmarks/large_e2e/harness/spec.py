"""Frozen CLI interface, argument validation and pre-work admission.

HARNESS.md freezes the command interface:

    <arm-python> <repo>/benchmarks/large_e2e/run.py \\
      --manifest <workload.json> --arm <label> --identity <arm_identity.json> \\
      --mode single|batch --jobs <K> --workers <W> --numba-threads <H> \\
      --flow-stripes <Kflow-or-default> --queue-depth <Q> --writer-limit <L> \\
      --cpu-budget <C> --memory-budget-mib <M> --timeout-s <T> \\
      --cache-root <unique-cache-root> --out <new-run-directory>

Every argument is validated here BEFORE any job execution.  Two extra,
clearly-marked optional flags exist and are forbidden in qualification:

  * --diagnostic-source-root <dir>  selects the distinctly named diagnostic
    source mode (measurement_class="diagnostic_source"); records produced in
    this mode carry qualification_valid=false and installed_qualified=false.
  * --test-fault <name>             selects a test-only fault injection for
    harness negative tests (harness.faults); measurement_class becomes
    "harness_selftest" and the run can never qualify.

Validation errors raise ValidationError; run.py maps them to exit code 2
with a structured rejection record.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import harness.dispatch as dispatch

# Declared policies (recorded into every session record).
SAMPLER_INTERVAL_S = 0.25          # fixed declared process-tree sample interval
POLL_TIMEOUT_S = 0.25              # short polling timeout for queue get()
PUT_TIMEOUT_S = 0.5                # short polling timeout for queue put()
READY_TIMEOUT_S = 120.0            # readiness barrier cap (also under --timeout-s)
SHUTDOWN_JOIN_S = 30.0             # graceful join cap per shutdown stage
RESULT_QUEUE_SLACK = 8             # extra slots above queue_depth + 2*W
DISK_MIN_FREE_BYTES = 1024**3      # 1 GiB floor for outputs+scratch
AVAILABLE_FRACTION_CAP = 0.80      # memory budget may not exceed this share of available RAM
PRESSURE_STOP_MIN_BYTES = 1024**3  # max(1 GiB, 0.10*physical) external-pressure stop
PRESSURE_STOP_FRACTION = 0.10

ENV_ALLOWLIST_INSTALLED = ("PATH", "HOME", "TMPDIR", "LANG")
# NUMBA_DISABLE_JIT is only ever accepted in diagnostic mode: it changes
# compiled behavior and must never be present for an installed qualification
# run.  The coordinator rejects it outright in installed mode.
DIAG_ONLY_ENV = ("NUMBA_DISABLE_JIT",)

SINGLE_ENV_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM",
    "NUMEXPR_NUM_THREADS",
)


class ValidationError(Exception):
    """A pre-work validation failure; mapped to exit code 2 by run.py."""


VALIDATION_EXIT = 2   # rejected before any work
RUN_EXIT = 1          # ran but INVALID (job failure, guard, watchdog, shutdown)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}")
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {parsed}")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"not a number: {value!r}")
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {parsed}")
    return parsed


def _flow_stripes(value: str) -> object:
    """'default' (case-insensitive) or a positive integer stripe count."""
    text = str(value).strip()
    if text.lower() == "default":
        return None  # None == keep production default stripe count
    return _positive_int(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "UNA large-e2e benchmark harness. Runs identical logic for the "
            "B0 baseline arm and every candidate arm; see "
            "campaigns/una_large_e2e/HARNESS.md."
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path,
                        help="verified workload manifest JSON (existing)")
    parser.add_argument("--arm", required=True,
                        help="arm label recorded into every record")
    parser.add_argument("--identity", required=True, type=Path,
                        help="arm identity JSON (harness/identity.py --make)")
    parser.add_argument("--mode", required=True, choices=["single", "batch"],
                        help="single = fresh process per job; batch = persistent pool")
    parser.add_argument("--jobs", required=True, type=_positive_int,
                        help="number of jobs K (positive)")
    parser.add_argument("--workers", required=True, type=_positive_int,
                        help="worker processes W (positive)")
    parser.add_argument("--numba-threads", required=True, type=_positive_int,
                        help="NUMBA_NUM_THREADS H requested per worker (positive)")
    parser.add_argument("--flow-stripes", required=True, type=_flow_stripes,
                        help="logical AggregateFlow stripes K, or 'default'")
    parser.add_argument("--queue-depth", required=True, type=_positive_int,
                        help="bounded input queue capacity Q (positive)")
    parser.add_argument("--writer-limit", required=True, type=_positive_int,
                        help="shared writer semaphore limit L (positive; L=W default policy)")
    parser.add_argument("--cpu-budget", required=True, type=_positive_int,
                        help="admitted CPU slot budget C (positive)")
    parser.add_argument("--memory-budget-mib", required=True, type=_positive_int,
                        help="watchdog memory budget in MiB (positive)")
    parser.add_argument("--timeout-s", required=True, type=_positive_float,
                        help="overall session watchdog timeout in seconds (>0)")
    parser.add_argument("--cache-root", required=True, type=Path,
                        help="arm-qualified JIT cache root (must be new or empty)")
    parser.add_argument("--out", required=True, type=Path,
                        help="run output directory (must be new or empty; unique)")
    parser.add_argument("--diagnostic-source-root", type=Path, default=None,
                        help="TEST/DIAGNOSTIC ONLY: run against a source tree "
                             "via this root; records are never installed-qualified")
    parser.add_argument("--test-fault", default=None,
                        help="TEST ONLY: inject a named harness fault "
                             "(see harness/faults.py); never valid qualification")
    return parser


class RunSpec:
    """Fully validated run configuration (no path defaults, nothing implicit)."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.arm = args.arm
        if not isinstance(self.arm, str) or not self.arm.strip():
            raise ValidationError("--arm must be a nonempty label")
        self.arm = self.arm.strip()

        self.manifest_path = args.manifest.expanduser().resolve()
        self.identity_path = args.identity.expanduser().resolve()
        self.out_dir = args.out.expanduser().resolve()
        self.cache_root = args.cache_root.expanduser().resolve()
        self.diagnostic_source_root = (
            args.diagnostic_source_root.expanduser().resolve()
            if args.diagnostic_source_root is not None else None
        )

        self.mode = args.mode
        self.jobs = args.jobs
        self.workers = args.workers
        self.numba_threads = args.numba_threads
        self.flow_stripes = args.flow_stripes          # None == production default
        self.queue_depth = args.queue_depth
        self.writer_limit = args.writer_limit
        self.cpu_budget = args.cpu_budget
        self.memory_budget_mib = args.memory_budget_mib
        self.timeout_s = args.timeout_s
        self.test_fault = args.test_fault

        self.diagnostic = self.diagnostic_source_root is not None
        self.measurement_class = (
            "diagnostic_source" if self.diagnostic
            else ("harness_selftest" if self.test_fault else "installed")
        )
        if self.test_fault is not None and self.diagnostic:
            raise ValidationError(
                "--test-fault and --diagnostic-source-root are mutually exclusive")

    # ------------------------------------------------------------------
    # Path / filesystem preflight (all failures are pre-work)
    # ------------------------------------------------------------------
    def validate_paths(self, input_dirs: list[Path]) -> dict:
        """Validate out/cache-root uniqueness and collision rules.

        input_dirs: directories of every manifest input file.  Outputs and
        warmups must not collide with inputs, with the cache root, or with
        any previous run's outputs.
        """
        checks: dict = {"out": str(self.out_dir), "cache_root": str(self.cache_root)}

        if self.out_dir.exists():
            if not self.out_dir.is_dir():
                raise ValidationError(f"--out exists and is not a directory: {self.out_dir}")
            if any(self.out_dir.iterdir()):
                raise ValidationError(
                    f"--out must be a new (or empty) unique directory; "
                    f"nonempty: {self.out_dir}")
        if self.cache_root.exists():
            if not self.cache_root.is_dir():
                raise ValidationError(
                    f"--cache-root exists and is not a directory: {self.cache_root}")
            if any(self.cache_root.iterdir()):
                raise ValidationError(
                    f"--cache-root must be new (or empty); nonempty: {self.cache_root}")

        # Out must not equal or contain / be contained by the cache root.
        if self.out_dir == self.cache_root:
            raise ValidationError("--out and --cache-root must differ")
        if _is_relative_to(self.out_dir, self.cache_root) or \
                _is_relative_to(self.cache_root, self.out_dir):
            raise ValidationError(
                f"--out ({self.out_dir}) and --cache-root ({self.cache_root}) "
                f"must not be nested")

        # Outputs (and warmups beneath out) must not touch input directories.
        for input_dir in input_dirs:
            input_dir = input_dir.resolve()
            if _is_relative_to(self.out_dir, input_dir) or \
                    _is_relative_to(input_dir, self.out_dir):
                raise ValidationError(
                    f"--out ({self.out_dir}) collides with input directory "
                    f"{input_dir}: outputs must never mix with inputs")
        checks["input_dirs"] = [str(d) for d in input_dirs]
        self.path_checks = checks
        return checks

    # ------------------------------------------------------------------
    # Environment policy (installed vs diagnostic)
    # ------------------------------------------------------------------
    def validate_environment(self, environ: dict) -> dict:
        """Installed mode forbids PYTHONPATH and NUMBA_DISABLE_JIT outright."""
        record = {"mode_policy": "diagnostic_source" if self.diagnostic else "installed"}
        if not self.diagnostic:
            if environ.get("PYTHONPATH"):
                raise ValidationError(
                    "installed mode refuses to run with PYTHONPATH set "
                    f"(got {environ['PYTHONPATH']!r}); unset it — source-tree "
                    "diagnostics must use --diagnostic-source-root instead")
            for var in DIAG_ONLY_ENV:
                if var in environ:
                    raise ValidationError(
                        f"installed mode refuses {var}: it changes compiled "
                        "behavior and would invalidate installed qualification")
        record["pythonpath_present"] = bool(environ.get("PYTHONPATH"))
        return record

    # ------------------------------------------------------------------
    # Resource admission (requested counts are not proof of effective counts;
    # this gate is necessary-not-sufficient and is re-verified by workers)
    # ------------------------------------------------------------------
    def validate_admission(self, system, analysis: str | None = None) -> dict:
        """system: object with cpu_count(), virtual_memory(), disk_usage().

        analysis: the manifest's analysis, when already known.  The flow
        worst-case gate (AggregateFlow runs its own K-stripe ThreadPoolExecutor
        ON TOP of the Numba pool) applies only to flow runs; accessibility and
        ODM never create that executor, so charging them for K would make the
        production-default stripe estimate refuse ordinary accessibility runs.
        """
        cpu_count = system.cpu_count()
        vm = system.virtual_memory()

        # --- CPU admission -------------------------------------------------
        if self.workers > self.cpu_budget:
            raise ValidationError(
                f"workers W={self.workers} exceeds cpu budget C={self.cpu_budget}")
        # Numba threads per worker.
        if self.workers * self.numba_threads > self.cpu_budget:
            raise ValidationError(
                f"W*H_numba = {self.workers}*{self.numba_threads} = "
                f"{self.workers * self.numba_threads} exceeds cpu budget "
                f"C={self.cpu_budget}")

        admission: dict = {
            "cpu_logical": cpu_count,
            "cpu_budget_C": self.cpu_budget,
            "workers_W": self.workers,
            "numba_threads_H": self.numba_threads,
            "flow_stripes_K_requested": self.flow_stripes,
        }
        if self.flow_stripes is None:
            # Production default topology stripes = cpu_count-1 (Topology.py).
            k_default = max(1, cpu_count - 1)
            admission["flow_stripes_K_default_estimate"] = k_default
            admission["flow_stripes_K_source"] = "production-default-estimate"
        else:
            k_default = self.flow_stripes
            admission["flow_stripes_K_source"] = "cli"

        if analysis == "flow" and self.mode == "batch":
            # Flow runs an AggregateFlow executor of K threads ON TOP of the
            # Numba pool; charge the conservative sum per worker (RESOURCES.md:
            # W*H_numba<=C is necessary but NOT sufficient for flow).
            if self.numba_threads + k_default > self.cpu_budget:
                raise ValidationError(
                    f"flow admission mismatch: per-job worst concurrency "
                    f"H_numba+K = {self.numba_threads}+{k_default} = "
                    f"{self.numba_threads + k_default} already exceeds "
                    f"C={self.cpu_budget} (e.g. NUMBA_NUM_THREADS=1 with a "
                    f"larger flow executor)")
            worst_per_worker = max(self.numba_threads, self.numba_threads + k_default)
            if self.workers * worst_per_worker > self.cpu_budget:
                raise ValidationError(
                    f"flow admission: W*(H_numba+K) = {self.workers}*"
                    f"({self.numba_threads}+{k_default}) = "
                    f"{self.workers * worst_per_worker} exceeds "
                    f"C={self.cpu_budget}")
            admission["flow_worst_concurrency_per_worker"] = worst_per_worker
            admission["flow_worst_concurrency_total"] = self.workers * worst_per_worker
        admission["flow_worst_case_gate"] = (
            "applied" if analysis == "flow" else
            "not_applied (AggregateFlow executor only exists for flow runs)")
        admission["within_budget"] = True

        # --- Memory admission ----------------------------------------------
        available = vm.available
        physical = vm.total
        budget_bytes = self.memory_budget_mib * 1024 * 1024
        if budget_bytes > AVAILABLE_FRACTION_CAP * available:
            raise ValidationError(
                f"memory budget {self.memory_budget_mib} MiB exceeds "
                f"{AVAILABLE_FRACTION_CAP:.0%} of available RAM "
                f"({available} bytes); refuse before work")
        pressure_stop = max(PRESSURE_STOP_MIN_BYTES,
                            PRESSURE_STOP_FRACTION * physical)
        admission.update({
            "memory_budget_mib": self.memory_budget_mib,
            "memory_budget_bytes": budget_bytes,
            "available_ram_bytes_at_admission": available,
            "physical_ram_bytes": physical,
            "pressure_stop_available_bytes": pressure_stop,
        })

        # --- Disk preflight -------------------------------------------------
        free_out = system.disk_usage(str(self.out_dir.parent)).free
        free_cache = system.disk_usage(str(self.cache_root.parent)).free
        admission["disk_free_out_parent_bytes"] = free_out
        admission["disk_free_cache_parent_bytes"] = free_cache
        if free_out < DISK_MIN_FREE_BYTES:
            raise ValidationError(
                f"disk free at --out parent ({free_out} bytes) below floor "
                f"{DISK_MIN_FREE_BYTES}")
        if free_cache < DISK_MIN_FREE_BYTES:
            raise ValidationError(
                f"disk free at --cache-root parent ({free_cache} bytes) below "
                f"floor {DISK_MIN_FREE_BYTES}")
        admission["disk_min_free_floor_bytes"] = DISK_MIN_FREE_BYTES
        self.admission = admission
        return admission

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        """Requested configuration echoed into every session record."""
        return {
            "arm": self.arm,
            "mode": self.mode,
            "jobs": self.jobs,
            "workers": self.workers,
            "numba_threads": self.numba_threads,
            "flow_stripes": self.flow_stripes if self.flow_stripes is not None else "default",
            "queue_depth": self.queue_depth,
            "writer_limit": self.writer_limit,
            "cpu_budget": self.cpu_budget,
            "memory_budget_mib": self.memory_budget_mib,
            "timeout_s": self.timeout_s,
            "manifest": str(self.manifest_path),
            "identity": str(self.identity_path),
            "cache_root": str(self.cache_root),
            "out": str(self.out_dir),
            "diagnostic_source_root": (
                str(self.diagnostic_source_root) if self.diagnostic else None),
            "test_fault": self.test_fault,
            "measurement_class": self.measurement_class,
            "declared_sampler_interval_s": SAMPLER_INTERVAL_S,
            "declared_poll_timeout_s": POLL_TIMEOUT_S,
            "declared_put_timeout_s": PUT_TIMEOUT_S,
            "declared_ready_timeout_s": READY_TIMEOUT_S,
            "declared_shutdown_join_s": SHUTDOWN_JOIN_S,
            "result_queue_maxsize": self.result_queue_maxsize(),
            "writer_limit_policy": (
                "explicit CLI value; final no-instrumentation timing policy "
                "expects L=W unless a reviewed pipeline preserves writer semantics"),
        }

    def result_queue_maxsize(self) -> int:
        """Bounded result channel: in-flight jobs + readies + exits + slack."""
        return self.queue_depth + 2 * self.workers + RESULT_QUEUE_SLACK


def _is_relative_to(path: Path, ancestor: Path) -> bool:
    """Resolution-based containment check (never a string-prefix match)."""
    try:
        path.resolve().relative_to(ancestor.resolve())
        return True
    except ValueError:
        return False


def parse_and_validate(argv: list[str] | None, system=None) -> RunSpec:
    """Parse argv and run every check that must happen before any work."""
    args = build_parser().parse_args(argv)
    spec = RunSpec(args)
    return spec


def supported_analyses() -> tuple[str, ...]:
    return dispatch.SUPPORTED_ANALYSES
