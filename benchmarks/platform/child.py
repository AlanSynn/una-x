"""Child-side plan execution (HARNESS.md "One real job").

The parent runner spawns a *clean* interpreter (installed wheel, scrubbed
env) with an inline bootstrap that imports this module and runs `main(plan)`.
The child:

1. asserts installed identity (before importing the candidate),
2. emits import boundary markers (flush+fsync; close is not durability),
3. dispatches the *actual requested public API exactly once*,
4. records an engagement trace (which public entry points actually ran and
   backend work counters) in a separate, un-timed leg configured by the plan,
5. writes outputs only under the plan's campaign-owned output directory,
6. reports output manifest and per-boundary child timings.

A plan may carry `mutant`: an intentionally broken candidate used by
tests/platform_harness as a negative control.  Production runs never set it.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

PUBLIC_ENTRYPOINTS = (
    "UNA.RunAccessibility", "UNA.RunODM", "UNA.RunFlow", "UNA.RunBatch",
)


class MarkerWriter:
    """Append-only JSONL marker file; flush + fsync per record."""

    def __init__(self, path: str):
        self.fh = open(path, "ab")

    def emit(self, **rec) -> None:
        rec["t_rel"] = time.perf_counter()
        self.fh.write(json.dumps(rec).encode() + b"\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())

    def close(self) -> None:
        try:
            self.fh.close()
        except OSError:
            pass


class DispatchTrace:
    """sys profile hook recording candidate public entry-point calls."""

    _ENTRY_NAMES = frozenset(n.split(".")[-1] for n in PUBLIC_ENTRYPOINTS)

    def __init__(self, package: str = "urban_network_analysis"):
        self.package = package
        self.called: list[str] = []

    def __enter__(self):
        self._old = sys.getprofile()
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *exc):
        sys.setprofile(self._old)
        return False

    def _hook(self, frame, event, args):
        if event != "call":
            return
        mod = frame.f_globals.get("__name__", "")
        if mod == self.package or mod.startswith(self.package + "."):
            if frame.f_code.co_name in self._ENTRY_NAMES:
                self.called.append(f"{mod}.{frame.f_code.co_name}")


def _build_settings(una, plan):
    """Construct Settings from the frozen manifest patch (no defaults drift:
    every patch key must exist on Settings).  Publication is redirected to
    the leg's campaign-owned output directory: workload content, settings
    and model are frozen; only the run's output location is harness-owned.
    """
    s = una.settings.__class__()
    s.Reset()
    patch = dict(plan["settings_patch"])
    for key, val in patch.items():
        if not hasattr(s, key):
            raise AttributeError(f"settings patch key {key!r} not on Settings")
        setattr(s, key, val)
    s.output_folder = plan["run"]["out_dir"]
    s.output_wStamp = False
    return s


def _dispatch_public_api(una, plan):
    """Dispatch the actual requested public API exactly once."""
    analysis = plan["model"]["analysis"]
    if analysis == "accessibility":
        una.RunAccessibility()
    elif analysis == "od":
        una.RunODM()
    elif analysis == "flow":
        una.RunFlow()
    elif analysis == "batch":
        una.RunBatch(plan["model"].get("batch_analysis", "accessibility"),
                     plan["model"]["pairing_file"])
    else:
        raise ValueError(f"unknown analysis {analysis!r}")


def _backend_counters(candidate) -> dict:
    """Substantive backend execution counters (real work, not a flag).

    The baseline package has no native/GPU backend; only 'reference' can be
    evidenced.  Native/GPU engagement later must return nonzero counters
    from their real execution paths — an empty dict fails engagement.
    """
    counters: dict = {}
    try:
        import numba
        counters["numba_version"] = numba.__version__
    except Exception:
        counters["numba_version"] = None
    sys_modules = sys.modules
    if "urban_network_analysis.backends.native" in sys_modules:
        counters["native_module_loaded"] = True
    else:
        counters["native_module_loaded"] = False
    if "urban_network_analysis.backends.gpu" in sys_modules:
        counters["gpu_module_loaded"] = True
    else:
        counters["gpu_module_loaded"] = False
    return counters


def apply_mutants(plan, markers, una) -> None:
    """Intentionally broken candidates (negative controls).  Each is a
    distinct, named defect; production plans never set `plan['mutant']`."""
    m = plan.get("mutant")
    if m == "wrong_dispatch":
        # claims flow, executes accessibility (control 1)
        plan["model"]["analysis"] = "accessibility"
    # other mutants live in parent-side checks or dedicated child flags
    # handled in run_plan below.


def run_plan(plan: dict, markers: MarkerWriter) -> None:
    t0 = time.perf_counter()
    repo = plan["harness"]["repo"]
    src_root = str(Path(repo) / "src")

    # 1. identity: import root must be an installed tree, not src shadowing
    import importlib
    pkg = plan["harness"]["package"]
    mod = importlib.import_module(pkg)
    mfile = getattr(mod, "__file__", "") or ""
    shadowed = bool(mfile) and str(Path(mfile).resolve()).startswith(
        str(Path(src_root).resolve()))
    markers.emit(marker="identity", module_file=mfile, source_shadow=shadowed)
    if shadowed:
        # negative control 2 state: source tree shadows the installed wheel
        raise SystemExit(3)

    markers.emit(marker="imported", t_import=time.perf_counter() - t0)

    # 2. build the public object exactly as a user would
    una_cls = getattr(mod, "UNA")
    una = una_cls(verbosity=0)
    s = _build_settings(una, plan)
    una.settings = s

    apply_mutants(plan, markers, una)

    out_dir = Path(plan["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. dispatch — engagement leg wraps the same public call with the trace
    mutant = plan.get("mutant")
    if plan["run"].get("engagement_leg"):
        with DispatchTrace(pkg) as trace:
            t_compute0 = time.perf_counter()
            _dispatch_public_api(una, plan)
            t_compute = time.perf_counter() - t_compute0
        markers.emit(marker="dispatch", called=sorted(set(trace.called)),
                     t_compute=t_compute)
    else:
        t_compute0 = time.perf_counter()
        _dispatch_public_api(una, plan)
        t_compute = time.perf_counter() - t_compute0
        markers.emit(marker="dispatch", called=None, t_compute=t_compute)

    markers.emit(marker="compute_done", t_compute=t_compute)

    # 4. outputs
    if mutant == "missing_output":
        declared = sorted(p.name for p in out_dir.iterdir() if p.is_file())
        if declared:
            (out_dir / declared[0]).unlink()
        markers.emit(marker="outputs", files=sorted(
            p.name for p in out_dir.iterdir() if p.is_file()))
    elif mutant == "unsynced_output":
        # report outputs but omit the export-sync marker (control 8 face 2)
        markers.emit(marker="outputs_nosync", files=sorted(
            p.name for p in out_dir.iterdir() if p.is_file()))
    else:
        files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
        markers.emit(marker="outputs", files=files)
        # export synchronization: fsync the output directory itself
        fd = os.open(out_dir, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        markers.emit(marker="export_synced", files=files)

    markers.emit(marker="backend", effective=plan["backend_reported"],
                 counters=_backend_counters(una))
    markers.emit(marker="done", t_total=time.perf_counter() - t0)


def main(argv: list[str]) -> int:
    plan_path = argv[argv.index("--plan") + 1]
    marker_path = argv[argv.index("--markers") + 1]
    markers = MarkerWriter(marker_path)
    try:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        run_plan(plan, markers)
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    except BaseException:
        markers.emit(marker="child_error", error=traceback.format_exc())
        return 1
    finally:
        markers.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
