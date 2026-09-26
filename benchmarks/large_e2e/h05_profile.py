"""H05 baseline stage/lifetime profiler (performance-owner tool, measurement only).

Runs ONE instrumented B0 public-call session (exactly ONE application window per
process, bracketing exactly one real public call: UNA.RunAccessibility or
UNA.RunFlow) and writes a raw JSON record:

  * nested diagnostic stage spans obtained by wrapping NAMED existing callables
    (no extra invocations, no extra constructors, no separately invoked cap
    pass; arguments/results/exceptions pass through unchanged);
  * an explicit partition check of the top-level stage spans against the
    application window (proof obligation: diagnostic nested spans are not added
    twice — sum of top-level inclusive spans <= window, no sibling overlaps, no
    span outside the window; every target instrumented exactly once via a patch
    registry);
  * process-tree RSS + system-available sampling (0.25 s) plus per-stage RSS
    checkpoints;
  * cold/warm JIT markers (first-call-per-stage markers, numba dispatcher
    signature growth, NUMBA_CACHE_DIR file-count before/after);
  * Numba NRT allocation counters when the process is started with
    NUMBA_NRT_STATS=1 (allocation-instrumented mode; timings from that run are
    labelled perturbed);
  * a direct per-pool thread inventory (threadpoolctl + numba threading layer +
    explicit executor objects + Python thread enumeration), captured identically
    at three points in every run (H04-N3 closure evidence);
  * verbatim logger entries + warnings per window, artifact hashes, settings
    before/after, V/E/O/D counts, resolved gravity cap;
  * od_level mode adds: per-call spans of the scipy dijkstra call sites, an
    AGGREGATE (count/sum/min/max/max-concurrent) record of every
    _accumulate_od_flow kernel invocation (per-call spans only for the first
    KERNEL_SPAN_SAMPLE calls — the OD call count can reach millions at sel-all
    scale), per-call NRT deltas, and post-window capture of the exact CSR /
    gradient / graph-engine arrays for bounded kernel probes.

This tool is diagnostic: per MODEL.md/BENCHMARKS.md it never times candidates,
never adds a second constructor or cap pass to an application total, and its
numbers are profiles, not qualification timings.

Import identity: urban_network_analysis MUST resolve under the site-packages of
the running environment (installed-wheel arm) or under an explicitly passed
--diagnostic-source-root (distinctly named diagnostic mode, never qualifying).

CLI:
  python h05_profile.py --manifest <workload.manifest.json> --cell <label>
      --mode clean|od_level --out <raw.json> --output-root <dir>
      [--expected-site-packages <dir>] [--diagnostic-source-root <dir>]
      [--origins-file-name <name>] [--pressure-stop-bytes <B>]
      [--lease-id <str>] [--run-id <str>] [--extra-settings-json <json>]
      [--capture-dir <dir>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import traceback

# ---------------------------------------------------------------------------
# Environment guards that must hold BEFORE numerical imports.
# ---------------------------------------------------------------------------
_L1 = os.environ.pop("L1_REUSE_DIR", None)
if _L1 is not None:
    print(f"[h05] refuted L1_REUSE_DIR was set ({_L1}); unset for this process", flush=True)

import numpy as np  # noqa: E402
import psutil  # noqa: E402

NRT_STATS = os.environ.get("NUMBA_NRT_STATS", "0") == "1"
SAMPLER_INTERVAL_S = 0.25
KERNEL_SPAN_SAMPLE = 200          # per-call spans kept for _accumulate_od_flow
KERNEL_NRT_EVERY = 256            # NRT read cadence inside the OD kernel wrapper

STAGE_RSS_ENABLED = False         # per-stage enter/exit RSS checkpoints (module flag)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arr_rec(a, k=8):
    if a is None:
        return None
    a = np.asarray(a)
    flat = a.reshape(-1)
    return {
        "dtype": str(a.dtype), "dtype_str": a.dtype.str, "shape": list(a.shape),
        "first_k": [float(x) if a.dtype.kind == "f" else int(x) for x in flat[:k]],
        "sha256_bytes": hashlib.sha256(a.tobytes()).hexdigest() if a.size else None,
        "n_nonfinite": int((~np.isfinite(flat)).sum()) if a.dtype.kind == "f" and a.size else 0,
    }


def rss_mib():
    return psutil.Process().memory_info().rss / (1 << 20)


def sigcounts(dispatchers):
    """numba Dispatcher -> len(signatures); plain-Python function -> None."""
    out = {}
    for k, d in dispatchers.items():
        try:
            out[k] = len(d.signatures)
        except AttributeError:
            out[k] = None
    return out


# ---------------------------------------------------------------------------
# Per-pool thread inventory (H04-N3 closure). Identical function in every run.
# ---------------------------------------------------------------------------
def inspect_pools(executor_registry):
    inv = {}
    try:
        from threadpoolctl import threadpool_info
        inv["threadpoolctl"] = threadpool_info()
    except Exception as e:
        inv["threadpoolctl"] = {"error": repr(e)}
    try:
        import numba
        initd = _threading_layer_initialized()
        inv["numba"] = {
            "version": numba.__version__,
            "get_num_threads": numba.get_num_threads(),
            "config_NUMBA_NUM_THREADS": numba.config.NUMBA_NUM_THREADS,
            "threading_layer_config": numba.config.THREADING_LAYER,
            "threading_layer_initialized": initd,
            "threading_layer": numba.threading_layer() if initd else "not_yet_initialized",
        }
    except Exception as e:
        inv["numba"] = {"error": repr(e)}
    try:
        import pyarrow as pa
        inv["pyarrow"] = {"version": pa.__version__, "cpu_count": pa.cpu_count(),
                          "io_thread_count": pa.io_thread_count()}
    except Exception as e:
        inv["pyarrow"] = {"error": repr(e)}
    ths = threading.enumerate()
    inv["python_threads"] = {
        "count": threading.active_count(),
        "names": sorted(t.name for t in ths),
        "native_ids": sorted(t.native_id for t in ths if t.native_id is not None),
    }
    try:
        inv["process_num_threads"] = psutil.Process().num_threads()
    except Exception as e:
        inv["process_num_threads"] = {"error": repr(e)}
    inv["explicit_executors"] = executor_registry.snapshot()
    inv["os_cpu_count"] = os.cpu_count()
    inv["note"] = ("macOS: psutil cpu_affinity unsupported; C=9 is a policy slot "
                   "budget, not OS pinning. Inventory = threadpoolctl-detectable BLAS/"
                   "OpenMP pools + numba threading layer + explicit "
                   "ThreadPoolExecutor objects + live Python threads.")
    return inv


def _threading_layer_initialized():
    try:
        import numba
        numba.threading_layer()
        return True
    except Exception:
        return False


class ExecutorRegistry:
    """Records every concurrent.futures.ThreadPoolExecutor created in-process,
    with creating stage tag and max_workers (explicit executor objects, per
    RESOURCES.md pool inventory; F1 admission evidence)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._execs = []
        self._next = 0

    def register(self, max_workers, tag):
        with self._lock:
            self._next += 1
            rec = {"executor_id": self._next, "max_workers": max_workers,
                   "created_under_stage": tag,
                   "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            self._execs.append(rec)
            return rec

    def snapshot(self):
        with self._lock:
            return [dict(r) for r in self._execs]


EXECUTOR_REGISTRY = ExecutorRegistry()

# ---------------------------------------------------------------------------
# Span instrumentation
# ---------------------------------------------------------------------------
class SpanRecorder:
    """Nested diagnostic spans from wrapping named callables.

    Each wrapped callable pushes its stage tag onto a THREAD-LOCAL stack; the
    span's parent is the tag underneath it in the same thread. Executor worker
    threads are seeded by the submit wrapper (stack cleared, parent tag pushed),
    so their spans nest under the dispatching stage exactly once. A callable is
    instrumented exactly once (PATCH_REGISTRY enforces this); children are
    attributed once by the enclosing context — no span is added twice.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.spans = []
        self.tls = threading.local()

    def stack(self):
        s = getattr(self.tls, "stack", None)
        if s is None:
            s = []
            self.tls.stack = s
        return s

    def current_tag(self):
        s = self.stack()
        return s[-1] if s else None

    def begin(self, stage, extra=None):
        rec = {"stage": stage, "parent": self.current_tag(),
               "tid": threading.get_native_id(), "depth": len(self.stack()),
               "t0_ns": time.perf_counter_ns()}
        if STAGE_RSS_ENABLED:
            rec["rss_in_mib"] = rss_mib()
        if extra:
            rec.update(extra)
        self.stack().append(stage)
        return rec

    def end(self, rec, extra=None):
        if rec is None:
            return
        s = self.stack()
        while s and s[-1] != rec["stage"]:
            s.pop()
        if s:
            s.pop()
        rec["t1_ns"] = time.perf_counter_ns()
        if STAGE_RSS_ENABLED:
            rec["rss_out_mib"] = rss_mib()
        if extra:
            rec.update(extra)
        with self.lock:
            self.spans.append(rec)


PATCH_REGISTRY = []  # [(id(obj), attr, stage)] — double-instrumentation guard
FIRST_CALLS_SEEN = set()
FIRST_CALLS_LOCK = threading.Lock()


def _first_call(stage):
    with FIRST_CALLS_LOCK:
        first = stage not in FIRST_CALLS_SEEN
        FIRST_CALLS_SEEN.add(stage)
        return first


def patch(obj, attr, stage, recorder, dispatchers=None, nrt=None, capture=None):
    """Wrap obj.<attr> exactly once. Optional:
      dispatchers: {name: numba_dispatcher} → per-call signature-growth record
                   (cold compile / warm cache-load markers);
      nrt: NRTCounter → per-call NRT delta;
      capture: dict → wrapper stores the call's return value under key attr
                   (post-window probe input; never re-invoked).
    """
    key = (id(obj), attr)
    for k, _, _ in PATCH_REGISTRY:
        if k == key:
            raise RuntimeError(f"double instrumentation attempt: {attr}")
    original = getattr(obj, attr)

    def wrapper(*a, **kw):
        sig_before = None
        if dispatchers:
            sig_before = {}
            for name, disp in dispatchers.items():
                try:
                    sig_before[name] = len(disp.signatures)
                except Exception:
                    sig_before[name] = None
        nb = nrt.read() if nrt else None
        rec = recorder.begin(stage, extra=({"sig_before": sig_before} if dispatchers else None))
        try:
            result = original(*a, **kw)
        finally:
            extra = {"first_call": _first_call(stage)}
            if dispatchers:
                sig_after = {}
                for name, disp in dispatchers.items():
                    try:
                        sig_after[name] = len(disp.signatures)
                    except Exception:
                        sig_after[name] = None
                extra["signatures_grew"] = {
                    n2: [sig_before[n2], sig_after[n2]]
                    for n2 in sig_before if sig_before[n2] != sig_after[n2]}
            if nrt:
                extra["nrt_delta"] = nrt.delta(nb, nrt.read())
            recorder.end(rec, extra=extra)
        if capture is not None:
            capture[attr] = result
        return result

    setattr(obj, attr, wrapper)
    PATCH_REGISTRY.append((key, attr, stage))
    wrapper.h05_capture = capture  # type: ignore[attr-defined]
    return original


class NRTCounter:
    """Numba NRT allocation counters (requires NUMBA_NRT_STATS=1 at process
    start). The counting perturbs timing, so NRT-enabled runs are labelled
    allocation-instrumented and their timings never mix with clean runs."""

    def __init__(self):
        self.ok = False
        self.error = None
        if NRT_STATS:
            try:
                from numba.core.runtime import rtsys
                rtsys.get_allocation_stats()  # raises if stats disabled
                self.ok = True
                self._rtsys = rtsys
            except Exception as e:
                self.error = repr(e)

    def read(self):
        if not self.ok:
            return None
        st = self._rtsys.get_allocation_stats()
        return {"alloc": st.alloc, "free": st.free,
                "mi_alloc": st.mi_alloc, "mi_free": st.mi_free}

    def delta(self, before, after):
        if not (self.ok and before and after):
            return None
        return {k: after[k] - before[k] for k in after}


# ---------------------------------------------------------------------------
# Sampler: recursive process-tree RSS + system available memory
# ---------------------------------------------------------------------------
class TreeSampler:
    def __init__(self, pressure_stop_bytes, breach_file=None, run_id=None):
        self.interval = SAMPLER_INTERVAL_S
        self.pressure_stop_bytes = pressure_stop_bytes
        self.breach_file = breach_file
        self.run_id = run_id
        self.samples = []
        self.peak_tree_bytes = 0
        self.breach = None
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True, name="h05-sampler")

    def _one(self):
        me = psutil.Process()
        rss = me.memory_info().rss
        try:
            children = me.children(recursive=True)
            child_rss = sum(c.memory_info().rss for c in children)
        except psutil.Error:
            children, child_rss = [], 0
        vm = psutil.virtual_memory()
        tree = rss + child_rss
        self.peak_tree_bytes = max(self.peak_tree_bytes, tree)
        s = {
            "t_ns": time.perf_counter_ns(),
            "wall_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "rss_root_bytes": rss,
            "rss_children_bytes": child_rss,
            "rss_tree_bytes": tree,
            "n_children": len(children),
            "sys_available_bytes": vm.available,
            "swap_used_bytes": psutil.swap_memory().used,
            "loadavg": os.getloadavg(),
        }
        self.samples.append(s)
        if vm.available < self.pressure_stop_bytes and self.breach is None:
            self.breach = {
                "t_ns": s["t_ns"], "wall_utc": s["wall_utc"],
                "sys_available_bytes": vm.available,
                "pressure_stop_bytes": self.pressure_stop_bytes,
                "note": "external pressure-stop threshold crossed during window "
                        "(RESOURCES.md refusal condition); campaign-owned watchdog "
                        "aborts the run",
            }
            if self.breach_file:
                try:
                    with open(self.breach_file, "w") as f:
                        json.dump({"run_id": self.run_id, "breach": self.breach}, f)
                except Exception:
                    pass
                # watchdog: abort this run (RESOURCES.md refusal semantics); the
                # supervisor detects the breach file and marks the window refused
                import signal
                os.kill(os.getpid(), signal.SIGTERM)
            return True
        return False

    def _loop(self):
        while not self.stop.is_set():
            try:
                if self._one():
                    self.stop.set()
                    break
            except psutil.Error:
                pass
            self.stop.wait(self.interval)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join(timeout=3)


def decimate(samples, max_n=4000):
    if len(samples) <= max_n:
        return samples
    step = len(samples) / max_n
    return [samples[int(i * step)] for i in range(max_n)]


# ---------------------------------------------------------------------------
# Partition check (proof obligation)
# ---------------------------------------------------------------------------
def partition_check(spans, window_ns, window_t0):
    tops = sorted((s for s in spans if s["parent"] is None), key=lambda s: s["t0_ns"])
    sum_top = sum(s["t1_ns"] - s["t0_ns"] for s in tops)
    overlaps = [[a["stage"], b["stage"], int(a["t1_ns"] - b["t0_ns"])]
                for a, b in zip(tops, tops[1:]) if b["t0_ns"] < a["t1_ns"]]
    outside = [{"stage": s["stage"], "before_window_ns": int(window_t0 - s["t0_ns"])}
               for s in spans if s["t0_ns"] < window_t0]
    by_stage = {}
    for s in spans:
        by_stage.setdefault(s["parent"], []).append(s)
    excl = []
    for s in tops:
        contained = [k for k in by_stage.get(s["stage"], [])
                     if k["t0_ns"] >= s["t0_ns"] and k["t1_ns"] <= s["t1_ns"]]
        kid_sum = sum(k["t1_ns"] - k["t0_ns"] for k in contained)
        excl.append({"stage": s["stage"],
                     "inclusive_ns": s["t1_ns"] - s["t0_ns"],
                     "children_inclusive_ns": kid_sum,
                     "exclusive_ns": (s["t1_ns"] - s["t0_ns"]) - kid_sum,
                     "n_children_contained": len(contained)})
    return {
        "window_ns": window_ns,
        "n_spans_total": len(spans),
        "n_top_level": len(tops),
        "top_level_stages_in_order": [s["stage"] for s in tops],
        "sum_top_level_inclusive_ns": sum_top,
        "window_minus_sum_top_ns": window_ns - sum_top,
        "residue_note": ("window - sum(top-level inclusive) is the uninstrumented "
                         "residue (loop glue, result assembly, interpreter overhead); "
                         "partition holds iff residue >= 0 and no overlaps/outside"),
        "top_level_sibling_overlaps_ns": overlaps,
        "spans_started_before_window": outside,
        "top_level_exclusive": excl,
        "partition_ok": bool(sum_top <= window_ns and not overlaps and not outside),
    }


# ---------------------------------------------------------------------------
# OD-kernel aggregate instrumentation (od_level)
# ---------------------------------------------------------------------------
class ODKernelStats:
    """Aggregate record of every _accumulate_od_flow invocation: count, wall
    sum/min/max, observed max concurrent executions (F1 evidence: observed
    kernel concurrency), first-K per-call spans, sampled NRT deltas."""

    def __init__(self, recorder, nrt):
        self.recorder = recorder
        self.nrt = nrt
        self.lock = threading.Lock()
        self.n_calls = 0
        self.sum_ns = 0
        self.min_ns = None
        self.max_ns = None
        self.cur_concurrent = 0
        self.max_concurrent = 0
        self.spans = []            # first KERNEL_SPAN_SAMPLE calls
        self.nrt_samples = []      # [{"call_index", "delta"}]

    def to_dict(self):
        with self.lock:
            return {
                "n_calls": self.n_calls,
                "sum_ns": self.sum_ns,
                "mean_ns": (self.sum_ns / self.n_calls) if self.n_calls else None,
                "min_ns": self.min_ns, "max_ns": self.max_ns,
                "observed_max_concurrent": self.max_concurrent,
                "n_per_call_spans_kept": len(self.spans),
                "per_call_spans_sample": self.spans,
                "nrt_samples": self.nrt_samples[:64],
                "nrt_note": f"NRT delta read every {KERNEL_NRT_EVERY} calls",
                "note": "max_concurrent updated at kernel enter/exit and sampled by a "
                        "2 ms monitor thread; >1 proves simultaneous kernel execution",
            }


def install_od_kernel_wrapper(AF_mod, recorder, nrt):
    key = (id(AF_mod), "_accumulate_od_flow")
    for k, _, _ in PATCH_REGISTRY:
        if k == key:
            raise RuntimeError("double instrumentation attempt: _accumulate_od_flow")
    original = AF_mod._accumulate_od_flow
    stats = ODKernelStats(recorder, nrt)
    stop = threading.Event()

    def monitor():
        prev = -1
        while not stop.wait(0.002):
            with stats.lock:
                cur = stats.cur_concurrent
                if cur != prev:
                    stats.max_concurrent = max(stats.max_concurrent, cur)
                    prev = cur

    threading.Thread(target=monitor, daemon=True, name="h05-od-monitor").start()

    def wrapper(*a, **kw):
        with stats.lock:
            stats.n_calls += 1
            idx = stats.n_calls
            stats.cur_concurrent += 1
            if stats.cur_concurrent > stats.max_concurrent:
                stats.max_concurrent = stats.cur_concurrent
        nb = nrt.read() if (nrt and nrt.ok and idx % KERNEL_NRT_EVERY == 0) else None
        t0 = time.perf_counter_ns()
        keep_span = idx <= KERNEL_SPAN_SAMPLE
        rec = recorder.begin("flow_od_kernel") if keep_span else None
        try:
            return original(*a, **kw)
        finally:
            dt = time.perf_counter_ns() - t0
            if rec is not None:
                recorder.end(rec, extra={"call_index": idx})
            with stats.lock:
                stats.sum_ns += dt
                stats.min_ns = dt if stats.min_ns is None else min(stats.min_ns, dt)
                stats.max_ns = dt if stats.max_ns is None else max(stats.max_ns, dt)
                stats.cur_concurrent -= 1
            if nb is not None:
                d = nrt.delta(nb, nrt.read())
                if d:
                    stats.nrt_samples.append({"call_index": idx, "delta": d})

    setattr(AF_mod, "_accumulate_od_flow", wrapper)
    PATCH_REGISTRY.append((key, "_accumulate_od_flow", "flow_od_kernel(aggregate)"))
    wrapper.h05_stats = stats  # type: ignore[attr-defined]
    wrapper.h05_stop = stop    # type: ignore[attr-defined]
    return original


def install_executor_patch(recorder):
    """Record ThreadPoolExecutor creation + seed worker-thread stage context.

    submit() wraps the submitted fn so its spans nest under the dispatching
    stage even though they run on pool threads (children attributed once —
    no double counting; worker stacks cleared to avoid cross-task bleed)."""
    import concurrent.futures as cf

    if getattr(cf.ThreadPoolExecutor, "_h05_patched", False):
        return
    orig_init = cf.ThreadPoolExecutor.__init__
    orig_submit = cf.ThreadPoolExecutor.submit

    def init(self, *a, **kw):
        max_workers = a[0] if a else kw.get("max_workers", None)
        EXECUTOR_REGISTRY.register(max_workers, recorder.current_tag())
        return orig_init(self, *a, **kw)

    def submit(self, fn, *a, **kw):
        parent_tag = recorder.current_tag()

        def run_under_context():
            recorder.stack().clear()
            if parent_tag:
                recorder.stack().append(parent_tag)
            return fn(*a, **kw)

        try:
            run_under_context.__name__ = getattr(fn, "__name__", "h05_wrapped")
        except Exception:
            pass
        return orig_submit(self, run_under_context)

    cf.ThreadPoolExecutor.__init__ = init
    cf.ThreadPoolExecutor.submit = submit
    cf.ThreadPoolExecutor._h05_patched = True


# ---------------------------------------------------------------------------
# Post-window array capture for bounded kernel probes (outside timing)
# ---------------------------------------------------------------------------
def capture_flow_arrays(REC, fe, capture_dir, captured_results):
    """fe = AggregateFlow instance; attribute names verified against B0 source
    (AggregateFlow.py: _build_csr lines 657-699, _precompute_dest_gradients
    lines 908-964)."""
    os.makedirs(capture_dir, exist_ok=True)
    d = os.path.join(capture_dir, "flow_arrays.npz")
    arrays, found, missing = {}, [], []
    candidates = {
        "csr_indptr": "_csr_indptr", "csr_indices": "_csr_indices",
        "csr_weights": "_csr_weights", "csr_edge_id": "_csr_edge_id",
        "csr_direction": "_csr_direction",
        "n_network_nodes": "_n_network_nodes", "n_destinations": "_n_destinations",
        "n_origins": "_n_origins", "first_origin_node": "_first_origin_node",
    }
    for name, attr in candidates.items():
        v = getattr(fe, attr, None)
        if v is None:
            missing.append(attr)
            continue
        arrays[name] = v if isinstance(v, np.ndarray) else np.asarray(v)
        found.append(attr)
    for csrname in ("_csr_fwd", "_csr_rev"):
        csr = getattr(fe, csrname, None)
        if csr is not None:
            tag = csrname.strip("_")
            arrays[tag + "_indptr"] = np.asarray(csr.indptr)
            arrays[tag + "_indices"] = np.asarray(csr.indices)
            arrays[tag + "_data"] = np.asarray(csr.data)
            found.append(csrname)
    grads = captured_results.get("_precompute_dest_gradients")
    if grads is not None:
        g_indptr, g_nodes, g_dist, g_pred = grads
        arrays.update({"grad_indptr": g_indptr, "grad_nodes": g_nodes,
                       "grad_dist": g_dist, "grad_pred": g_pred})
        found.append("_precompute_dest_gradients(return)")
    np.savez_compressed(d, **arrays)
    REC["capture"] = {
        "path": d, "bytes": os.path.getsize(d), "attrs_found": found,
        "attrs_missing": missing, "gradients_captured": grads is not None,
        "note": "post-window capture of the exact arrays the public call produced, "
                "for bounded kernel probes; never inside timing",
    }


def capture_access_arrays(REC, acc_engine, una, capture_dir):
    os.makedirs(capture_dir, exist_ok=True)
    d = os.path.join(capture_dir, "access_arrays.npz")
    graph = getattr(acc_engine, "graph_engine", None)
    arrays, found, missing = {}, [], []
    if graph is not None:
        candidates = {
            "adjacency_pointer": "adjacency_pointer",
            "adjacency_vector": "adjacency_vector",
            "adjacency_vector_weights": "adjacency_vector_weights",
            "adjacynct_vector_network_node": "adjacynct_vector_network_node",
            "o_terminal_idxs": "o_terminal_idxs",
            "o_terminal_weights": "o_terminal_weights",
            "d_terminal_idxs": "d_terminal_idxs",
            "d_terminal_weights": "d_terminal_weights",
        }
        for name, attr in candidates.items():
            v = getattr(graph, attr, None)
            if v is None:
                missing.append(attr)
                continue
            arrays[name] = np.asarray(v)
            found.append(attr)
    dest = getattr(getattr(una, "topology", None), "destinations", None)
    if dest is not None and getattr(dest, "node_weight", None) is not None:
        arrays["d_node_weight"] = np.asarray(dest.node_weight)
        found.append("destinations.node_weight")
    np.savez_compressed(d, **arrays)
    REC["capture"] = {
        "path": d, "bytes": os.path.getsize(d), "attrs_found": found,
        "attrs_missing": missing,
        "note": "post-window capture of the exact graph-engine arrays for bounded kernel probes",
    }


# ---------------------------------------------------------------------------
# Main session
# ---------------------------------------------------------------------------
def cache_dir_state(d):
    if not d:
        return "unset"
    if not os.path.isdir(d):
        return "missing"
    return f"exists_{sum(len(fs) for _, _, fs in os.walk(d))}_files"


def _ru_maxrss_mib():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
    except Exception:
        return None


def _write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--mode", choices=["clean", "od_level"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--output-root", required=True,
                    help="unique campaign-owned output dir for this job's exports")
    ap.add_argument("--expected-site-packages", default=None,
                    help="installed arm: site-packages dir that must contain urban_network_analysis")
    ap.add_argument("--diagnostic-source-root", default=None,
                    help="diagnostic mode: source root; record never qualifies as installed")
    ap.add_argument("--variant-id", default=None,
                    help="origin_selection variant id for multi-variant manifests (O3_FLOW: sel256|sel1024|selall)")
    ap.add_argument("--selection-manifest", default=None,
                    help="absolute override for origin_selection.selection_manifest (default: repo-root resolve)")
    ap.add_argument("--lease-id", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--pressure-stop-bytes", type=int,
                    default=max(1 << 30, int(0.10 * psutil.virtual_memory().total)))
    ap.add_argument("--extra-settings-json", default=None,
                    help="JSON object of extra settings attribute assignments, applied "
                         "AFTER manifest settings (e.g. holdout 3D leg)")
    ap.add_argument("--capture-dir", default=None,
                    help="od_level: directory for post-window observed-array capture")
    args = ap.parse_args(argv)

    REC = {
        "schema_version": 1, "task": "H05", "record": "profile_session",
        "role": "performance-owner", "cell": args.cell, "mode": args.mode,
        "measurement_class": ("diagnostic_profile_installed" if args.expected_site_packages
                              else "diagnostic_profile_source_NEVER_QUALIFIES"),
        "run_id": args.run_id, "lease_id": args.lease_id,
        "nrt_stats_enabled": NRT_STATS,
        "timing_class": ("alloc_instrumented_perturbed" if NRT_STATS else
                         ("od_instrumented" if args.mode == "od_level" else "clean")),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pressure_stop_bytes": args.pressure_stop_bytes,
        "env": {"L1_REUSE_DIR": os.environ.get("L1_REUSE_DIR", "<unset>"),
                "PYTHONPATH": os.environ.get("PYTHONPATH", "<unset>"),
                "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR"),
                "NUMBA_NRT_STATS": os.environ.get("NUMBA_NRT_STATS"),
                "numba_cache_dir_state_at_start": cache_dir_state(os.environ.get("NUMBA_CACHE_DIR"))},
    }

    # ---- import identity ---------------------------------------------------
    if args.diagnostic_source_root:
        sys.path.insert(0, args.diagnostic_source_root)
    import urban_network_analysis as una_pkg
    una_path = os.path.dirname(os.path.abspath(una_pkg.__file__))
    REC["import"] = {"urban_network_analysis_file": una_path,
                     "package_version": getattr(una_pkg, "__version__", None)}
    if args.expected_site_packages:
        real, exp = os.path.realpath(una_path), os.path.realpath(args.expected_site_packages)
        try:
            under = os.path.commonpath([real, exp]) == exp
        except ValueError:
            under = False
        if not under:
            REC["import"]["identity_assertion"] = "FAILED"
            _write_json(args.out, REC)
            sys.exit(3)
        REC["import"]["identity_assertion"] = "ok (realpath under expected site-packages)"
        REC["import"]["expected_site_packages"] = exp
    else:
        REC["import"]["identity_assertion"] = "diagnostic-source mode (never qualifying)"

    import numba
    REC["process"] = {
        "python": sys.version, "executable": sys.executable,
        "numpy": np.__version__, "numba": numba.__version__,
        "numba_num_threads": numba.get_num_threads(),
        "rss_at_import_mib": rss_mib(),
        "available_ram_bytes_at_start": psutil.virtual_memory().available,
        "swap_used_bytes_at_start": psutil.swap_memory().used,
        "loadavg_at_start": os.getloadavg(),
        "cmdline": " ".join(sys.argv),
    }

    from urban_network_analysis import UNA
    # the package __init__ rebinds `Settings`/`Topology` to the CLASSES
    # (`from .Settings import Settings`); the MODULES live in sys.modules.
    Settings_mod = sys.modules["urban_network_analysis.Settings"]
    Topology_mod = sys.modules["urban_network_analysis.Topology"]
    from urban_network_analysis.Engines import Base as EnginesBase
    from urban_network_analysis.Engines import AccessibilityWElevation as AWE_mod
    from urban_network_analysis.Engines import AggregateFlow as AF_mod
    from urban_network_analysis.Engines.AccessibilityWElevation import AccessibilityWElevation
    from urban_network_analysis.Engines.AggregateFlow import AggregateFlow

    DISPATCHERS = {
        "awe_integrated_scope_access": AWE_mod.integrated_scope_access,
        "af_accumulate_od_flow": AF_mod._accumulate_od_flow,
        # plain-Python dispatcher over per-mode njit kernels (no .signatures)
        "af_compute_trip_volumes": AF_mod._compute_trip_volumes,
    }
    REC["dispatchers_initial_sigcounts"] = sigcounts(DISPATCHERS)

    nrt = NRTCounter()
    REC["nrt_counter"] = {"enabled": nrt.ok, "error": nrt.error}
    recorder = SpanRecorder()

    # ---- instrument (each target exactly once; names verified vs B0 source) --
    patch(Settings_mod.Settings, "Validation", "settings_validation", recorder)

    patch(Topology_mod.Topology, "AddNetwork", "topology_add_network", recorder)
    patch(Topology_mod.Topology, "BuildAccessPoints", "topology_build_access_points", recorder)
    patch(Topology_mod.Topology, "AddOrigins", "topology_add_origins", recorder)
    patch(Topology_mod.Topology, "AddDestinations", "topology_add_destinations", recorder)
    patch(Topology_mod.Topology, "AddObservers", "topology_add_observers", recorder)
    patch(Topology_mod.Topology, "AddObstacles", "topology_add_obstacles", recorder)
    patch(Topology_mod.Topology, "BuildClusters", "flow_build_clusters", recorder)

    patch(AccessibilityWElevation, "__init__", "acc_engine_construct", recorder)
    patch(AccessibilityWElevation, "Centrality", "acc_centrality", recorder,
          dispatchers={"awe_integrated_scope_access": AWE_mod.integrated_scope_access})
    patch(AWE_mod, "integrated_scope_access", "acc_search_kernel_integrated", recorder,
          nrt=nrt if args.mode == "od_level" else None)

    patch(UNA, "_ResolveGravityCap", "flow_auto_cap_resolve", recorder)

    patch(AggregateFlow, "__init__", "flow_engine_construct", recorder)
    patch(AggregateFlow, "Centrality", "flow_engine_centrality", recorder,
          dispatchers={"af_accumulate_od_flow": AF_mod._accumulate_od_flow})
    patch(AggregateFlow, "_prepare_params", "flow_prepare_params", recorder)
    patch(AggregateFlow, "_build_digraph", "flow_build_digraph", recorder)
    patch(AggregateFlow, "_build_csr", "flow_build_csr", recorder)
    patch(AggregateFlow, "_precompute_dest_gradients", "flow_gradient_precompute", recorder,
          capture={} if args.mode == "od_level" else None)
    patch(AggregateFlow, "_process_origins_aggregate", "flow_origin_loop", recorder,
          dispatchers={"af_accumulate_od_flow": AF_mod._accumulate_od_flow})
    patch(AggregateFlow, "_accumulate_observer_flows", "flow_observer_flows", recorder)

    patch(EnginesBase.Base, "ExportAccessibilityResults", "export_accessibility_results", recorder)
    patch(EnginesBase.Base, "ExportFlowResult", "export_flow_result", recorder)

    if args.mode == "od_level":
        install_od_kernel_wrapper(AF_mod, recorder, nrt)
        patch(AF_mod, "_scipy_dijkstra", "scipy_dijkstra", recorder, nrt=nrt)
        patch(AF_mod, "_compute_trip_volumes", "flow_trip_volumes", recorder,
              dispatchers={"af_compute_trip_volumes": AF_mod._compute_trip_volumes})
    install_executor_patch(recorder)

    REC["patch_registry"] = [{"attr": a, "stage": s} for _, a, s in PATCH_REGISTRY]
    REC["pool_inventory_post_patch"] = inspect_pools(EXECUTOR_REGISTRY)

    # ---- manifest + input rehash --------------------------------------------
    with open(args.manifest) as f:
        manifest = json.load(f)
    rehash = {}
    for key, spec in manifest["inputs"].items():
        p = spec["local_path"]
        h, b = sha256_file(p), os.path.getsize(p)
        rehash[key] = {"sha256": h, "bytes": b,
                       "match": h == spec["sha256"] and b == spec["bytes"]}
    REC["input_rehash"] = rehash
    if not all(v["match"] for v in rehash.values()):
        REC["fatal"] = "input_rehash_mismatch"
        _write_json(args.out, REC)
        sys.exit(2)
    REC["manifest"] = {"path": os.path.abspath(args.manifest),
                       "sha256": sha256_file(args.manifest),
                       "workload_id": manifest.get("workload_id"),
                       "settings_overrides": manifest.get("settings_overrides")}

    # ---- origin selection (manifest-anchored; H04 fixture semantics) ---------
    # Resolve manifest.origin_selection -> pinned selection manifest -> slice the
    # verified source origins into a campaign-owned fixture BEFORE the window.
    # Identity selections use the source file directly (no rewrite).
    def _fatal(msg):
        REC["fatal"] = msg
        _write_json(args.out, REC)
        sys.exit(2)

    sel_spec = manifest.get("origin_selection")
    if not sel_spec:
        _fatal("manifest has no origin_selection block")
    if "variants" in sel_spec:
        vids = [v["variant_id"] for v in sel_spec["variants"]]
        if args.variant_id not in vids:
            _fatal(f"--variant-id must be one of {vids} for a multi-variant manifest")
        sel = next(v for v in sel_spec["variants"] if v["variant_id"] == args.variant_id)
    else:
        sel = sel_spec
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(args.manifest),
                                             "..", "..", ".."))
    sel_path = (os.path.abspath(args.selection_manifest) if args.selection_manifest
                else os.path.join(REPO_ROOT, sel["selection_manifest"]))
    sel_sha = sha256_file(sel_path)
    if sel_sha != sel.get("selection_manifest_sha256"):
        _fatal("selection manifest sha mismatch vs parent workload manifest "
               f"({sel_sha} != {sel.get('selection_manifest_sha256')})")
    with open(sel_path) as f:
        selman = json.load(f)
    if selman.get("source_sha256") != manifest["inputs"]["origins"]["sha256"]:
        _fatal("selection manifest source_sha256 != manifest inputs.origins.sha256")
    ordered = selman["ordered_indices"]
    n_src = manifest["inputs"]["origins"]["feature_count"]
    if (len(ordered) != sel["selected_count"] or len(set(ordered)) != len(ordered)
            or any(not isinstance(i, int) or not (0 <= i < n_src) for i in ordered)):
        _fatal("selection manifest ordered_indices invalid vs selected_count/feature_count")
    sel_rec = {"selection_manifest": sel_path, "selection_manifest_sha256": sel_sha,
               "selected_count": sel["selected_count"],
               "workload_variant": selman.get("workload_variant")}
    if args.variant_id:
        sel_rec["chosen_variant"] = args.variant_id
    if ordered == list(range(n_src)):
        origins_path = manifest["inputs"]["origins"]["local_path"]
        sel_rec["fixture"] = ("none — ordered_indices is the identity; verified "
                              "source file used directly, no rewrite")
    else:
        import geopandas as gpd
        src = gpd.read_file(manifest["inputs"]["origins"]["local_path"])
        if len(src) != n_src:
            _fatal(f"origins source has {len(src)} rows, manifest says {n_src}")
        fx = src.iloc[ordered].reset_index(drop=True)
        rows = selman.get("selected_row_attributes") or []
        if rows:
            for i, want in enumerate(rows):
                got = fx.iloc[i]
                for col in ("fid", "id", "lon", "lat"):
                    wv, gv = want.get(col), got.get(col)
                    same = (wv == gv) if isinstance(wv, str) else (
                        (wv != wv and gv != gv) or wv == gv)  # NaN==NaN
                    if not same:
                        _fatal(f"fixture row {i} attr {col}: {gv!r} != selection manifest {wv!r}")
            sel_rec["attribute_crosscheck"] = "fid/id/lon/lat row-by-row vs selection manifest: ok"
        data_dir = os.path.dirname(manifest["inputs"]["network"]["local_path"])
        fixture = os.path.join(
            data_dir, f"h05_origins_fixture_{args.run_id or 'run'}_{sel['selected_count']}.geojson")
        fx.to_file(fixture, driver="GeoJSON")
        origins_path = fixture
        sel_rec["fixture"] = {"path": fixture, "sha256": sha256_file(fixture),
                              "rows": int(len(fx)),
                              "note": "derived in-run from verified selection manifest; "
                                      "attributes sliced untouched, row order = ordered_indices"}
    origins_name = os.path.basename(origins_path)
    REC["origin_selection"] = sel_rec

    # ---- construct UNA (outside windows; counted once) -----------------------
    t0 = time.perf_counter_ns()
    una = UNA(verbosity=1)
    REC["constructor"] = {"call": "UNA(verbosity=1)", "wall_ns": time.perf_counter_ns() - t0,
                          "note": "outside application windows; counted once (protocol)"}
    REC["settings_pristine"] = una.settings.ToDict(compact=False)

    # ---- settings assembly from frozen manifest ------------------------------
    st = una.settings
    PATH_KEYS = {"data_folder", "network_file", "origins_file", "destinations_file",
                 "output_folder"}
    for k, v in manifest["settings_before"].items():
        if k in PATH_KEYS or k.startswith("_"):
            continue
        if hasattr(st, k):
            setattr(st, k, v)
    for k, v in manifest.get("settings_overrides", {}).items():
        setattr(st, k, v)
    st.data_folder = os.path.dirname(manifest["inputs"]["network"]["local_path"])
    st.network_file = os.path.basename(manifest["inputs"]["network"]["local_path"])
    origins_name = os.path.basename(origins_path)
    st.origins_file = origins_name
    st.destinations_file = os.path.basename(manifest["inputs"]["destinations"]["local_path"])
    if args.extra_settings_json:
        extra = json.loads(args.extra_settings_json)
        for k, v in extra.items():
            setattr(st, k, v)
        REC["extra_settings_applied"] = extra
    os.makedirs(args.output_root, exist_ok=True)
    st.output_folder = args.output_root

    assembled = una.settings.ToDict(compact=False)
    REC["settings"] = {
        "assembled": assembled,
        "assembly_diff_vs_manifest_settings_before": {
            k: [assembled.get(k), v] for k, v in manifest["settings_before"].items()
            if k in assembled and assembled.get(k) != v},
        "origins_file_effective": origins_name,
        "diff_policy": "only path-locational keys + explicit extra-settings may differ",
    }

    is_flow = manifest.get("workload_id", "").endswith("_FLOW")
    public_call = "una.RunFlow()" if is_flow else "una.RunAccessibility()"
    REC["public_call"] = public_call
    REC["pool_inventory_pre_window"] = inspect_pools(EXECUTOR_REGISTRY)
    nrt_before_window = nrt.read()

    # ---- THE application window: exactly ONE real public call ----------------
    log_start = len(una.topology.logger.log_list)
    REC["settings_before_window"] = una.settings.ToDict(compact=False)
    globals()["STAGE_RSS_ENABLED"] = True
    sampler = TreeSampler(args.pressure_stop_bytes,
                          breach_file=os.path.join(
                              os.path.dirname(os.path.abspath(args.out)),
                              f"pressure_breach_{args.run_id or 'run'}.json"),
                          run_id=args.run_id)
    sampler.__enter__()
    w0 = time.perf_counter_ns()
    error = None
    try:
        if is_flow:
            una.RunFlow()
        else:
            una.RunAccessibility()
    except Exception:
        error = traceback.format_exc()
        raise
    finally:
        w1 = time.perf_counter_ns()
        sampler.__exit__()
        globals()["STAGE_RSS_ENABLED"] = False
    window_ns = w1 - w0
    nrt_after_window = nrt.read()

    od_stats = None
    if args.mode == "od_level":
        w = getattr(AF_mod, "_accumulate_od_flow", None)
        if w is not None and hasattr(w, "h05_stop"):
            w.h05_stop.set()
            od_stats = w.h05_stats.to_dict()

    REC["window"] = {
        "public_call": public_call,
        "application_window_ns": window_ns,
        "boundary": "perf_counter_ns immediately before the public call through return "
                    "(incl. synchronous exports); exactly ONE real public call; constructor "
                    "outside; gravity cap never pre-resolved outside RunFlow",
        "raised": error is not None,
        "traceback": error,
        "nrt_delta_window": nrt.delta(nrt_before_window, nrt_after_window),
        "peak_rss_tree_sampled_mib": sampler.peak_tree_bytes / (1 << 20),
        "pressure_breach": sampler.breach,
    }
    if od_stats is not None:
        REC["od_kernel_aggregate"] = od_stats

    # ---- post-window facts (outside timing) ----------------------------------
    REC["settings_after_window"] = una.settings.ToDict(compact=False)
    REC["pool_inventory_post_window"] = inspect_pools(EXECUTOR_REGISTRY)
    REC["env"]["numba_cache_dir_state_at_end"] = cache_dir_state(os.environ.get("NUMBA_CACHE_DIR"))
    REC["dispatchers_final_sigcounts"] = sigcounts(DISPATCHERS)
    ent = list(una.topology.logger.log_list)[log_start:]
    REC["logger_entries"] = ent
    REC["verbatim_warnings"] = [
        e for e in ent
        if "WARN" in str(e.get("event", "")).upper()
        or "WARN" in str(e.get("details", "")).upper()]
    REC["counts"] = {
        "V_nodes": int(una.topology.network.node_points.shape[0]),
        "E_edges": int(una.topology.network.lengths.shape[0]),
        "O_rows": int(len(una.topology.origins.geometry)),
        "D_rows": int(len(una.topology.destinations.geometry)),
    }
    if is_flow:
        cap = getattr(una.settings, "flow_gravity_cap", None)
        REC["gravity_cap"] = {
            "before": manifest["settings_before"].get("flow_gravity_cap"),
            "after": cap,
            "note": "resolved INSIDE the public call when the settings value was a "
                    "percentile string (never pre-resolved)",
        }
        fe = una.flow
        REC["flow_arrays"] = {
            "edge_flow": arr_rec(getattr(fe, "edge_flow", None)),
            "edge_flow_AB": arr_rec(getattr(fe, "edge_flow_AB", None)),
            "edge_flow_BA": arr_rec(getattr(fe, "edge_flow_BA", None)),
            "node_flow": arr_rec(getattr(fe, "node_flow", None)),
            "assigned_stats": getattr(fe, "_assigned_stats", None),
        }
        if args.mode == "od_level" and args.capture_dir:
            capt = getattr(AggregateFlow._precompute_dest_gradients, "h05_capture", None)
            capture_flow_arrays(REC, fe, args.capture_dir, capt or {})
    else:
        acc = una.accessibility
        REC["accessibility_arrays"] = {
            m: arr_rec(getattr(acc, m, None))
            for m in ("reach", "gravity_exponential", "gravity_logistic", "knn_access")}
        if args.mode == "od_level" and args.capture_dir:
            capture_access_arrays(REC, acc, una, args.capture_dir)
            # per-origin engine metric values at the probe sample (post-window,
            # outside timing) — validation targets for the A2 kernel probe
            rng = np.random.default_rng(20260925)
            n_o = REC["counts"]["O_rows"]
            sample = np.sort(rng.choice(n_o, size=min(32, n_o), replace=False))
            metrics = {}
            for o in sample:
                metrics[str(int(o))] = {
                    "reach": float(acc.reach[o]),
                    "gravity_exponential": float(acc.gravity_exponential[o]),
                    "gravity_logistic": float(acc.gravity_logistic[o]),
                    "knn_access": float(acc.knn_access[o]),
                }
            with open(os.path.join(args.capture_dir, "access_engine_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=1)
            REC["sampled_origin_metrics"] = {
                "path": os.path.join(args.capture_dir, "access_engine_metrics.json"),
                "seed": 20260925, "n": int(len(sample)),
            }

    artifacts = {}
    for dirpath, _, files in os.walk(args.output_root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            artifacts[os.path.relpath(p, args.output_root)] = {
                "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    REC["artifacts"] = artifacts
    REC["rss_after_window_mib"] = rss_mib()
    REC["ru_maxrss_mib"] = _ru_maxrss_mib()

    # ---- spans + partition check ---------------------------------------------
    REC["spans"] = recorder.spans
    REC["first_call_stages"] = sorted(FIRST_CALLS_SEEN)
    REC["partition_check"] = partition_check(recorder.spans, window_ns, w0)
    REC["sampler"] = {
        "interval_s": SAMPLER_INTERVAL_S, "n_samples": len(sampler.samples),
        "samples": decimate(sampler.samples),
        "peak_rss_tree_bytes": sampler.peak_tree_bytes,
        "note": "summed RSS is conservative (shared pages may double count); "
                "macOS non-root has no USS/PSS",
    }
    REC["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    REC["no_double_instrumentation_note"] = (
        "Diagnostic nested spans are not added twice: every wrapped callable is "
        "instrumented exactly once (PATCH_REGISTRY raises on repeat), children attach "
        "to the enclosing stage via thread-local context (executor worker stacks are "
        "cleared and seeded with the dispatching stage's tag exactly once per task), "
        "and partition_check verifies sum(top-level) <= window, no sibling overlaps, "
        "and no span outside the window. Kernels invoked only inside compiled code "
        "(compact_vector_node_view_scope, adjust_destination_distances, "
        "reach_gravity_knn_access, _find_arc, _decay) are NOT wrapped — their time "
        "appears inside their compiled parent's span and is decomposed only by "
        "separate bounded kernel probes outside application windows.")
    _write_json(args.out, REC)
    print(f"[h05] session written: {args.out} window={window_ns/1e9:.3f}s "
          f"partition_ok={REC['partition_check']['partition_ok']}", flush=True)


if __name__ == "__main__":
    main()
