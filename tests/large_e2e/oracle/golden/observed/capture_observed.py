"""H03-N4 supplement: capture OBSERVED-typed-array kernel-boundary
fixtures from a real B0 O2 observed pass (accessibility + flow with the
auto gravity cap), for later candidate-vs-B0 kernel comparisons.

Bounded execution per the campaign constraints:
  - RAM gate: abort unless psutil available >= max(1 GiB, 10% physical).
  - Single run, no sweeps: one UNA construction, one RunAccessibility,
    one RunFlow (the same public sequence the H04 reviewer recorded).
  - Caller must set PYTHONPATH=<wt-b0>/src and
    NUMBA_CACHE_DIR=<campaign_data>/nbc_oracle and apply a subprocess
    timeout.

Capture method (no B0 source edits; wt-b0 is read-only):
  Python-level monkeypatching of the module attributes that hold the
  numba dispatchers, plus the one Python-method gradient boundary.
  njit->njit inner calls bind at compile time and are NOT intercepted;
  the captured boundaries are exactly the Python-visible kernel
  boundaries a candidate must reproduce:

    AccessibilityWElevation.integrated_scope_access   (O2 acc kernel;
        called by RunAccessibility AND again inside RunFlow by
        _ResolveGravityCap for the auto p95 cap)
    AggregateFlow._accumulate_od_flow                 (per-OD via-arc
        loading kernel; zero-buffer replay captures the per-call delta)
    AggregateFlow._compute_trip_volumes               (per-origin trip
        generation kernel)
    AggregateFlow.AggregateFlow._precompute_dest_gradients  (scipy-based
        reverse-gradient assembly; F3 boundary)
  Coverage ledgers are also kept for the kernels the pass does NOT
  call (base Accessibility family, OD-matrix wrapper).

Every stored array is deduplicated by (dtype, shape, sha256(bytes));
per-call records reference array keys. The selected _accumulate_od_flow
calls (first N_SEL_PER_ORIGIN eligible destinations per origin) are
replayed immediately after the live call with byte-identical inputs and
FRESH ZERO output buffers, so the fixture stores the exact per-call
output delta (the live stripe buffers are cumulative and are NOT
stored per call). delivered must be bit-identical between the live
call and the zero-buffer replay (the kernel's return does not read the
out buffers).

Outputs (in this directory):
  observed_o2_kernels.npz          all stored arrays (compressed)
  observed_o2_kernels.hashes.json  sidecar: env, provenance, per-array
                                   dtype/shape/contiguity/sha256, call
                                   ledgers, npz sha256
Both files are chmod 444 after writing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time

import numpy as np

B0_SRC = "/Users/alansynn/orca/workspaces/una-x/wt-b0/src"
CAMPAIGN_DATA = "/Users/alansynn/orca/workspaces/una-x/campaign_data"
FIXTURE = os.path.join(CAMPAIGN_DATA, "O2_origins_h04review.geojson")
FIXTURE_SHA = "afaf4bf4651a1c9d76a8328ab8e78d99bb35699b3fd378c297c54661dea09492"
O2_MANIFEST = ("/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/"
               "tests/large_e2e/inputs/O2.manifest.json")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(OUT_DIR, "observed_o2_kernels.npz")
SIDECAR_PATH = os.path.join(OUT_DIR, "observed_o2_kernels.hashes.json")
N_SEL_PER_ORIGIN = 2

B0_COMMIT = "361928e4ba38f34622cafe065b0025244db61368"

OD_ARG_NAMES = [
    "indptr", "indices", "weights", "edge_id_of_arc", "dir_of_arc",
    "d_o", "d_d", "pred_o", "pred_d",
    "origin_virtual_node", "dest_virtual_node", "o_edge_id", "d_edge_id",
    "d_shortest", "budget",
    "decay_curve_id", "decay_beta", "decay_midpoint",
    "trip_volume", "n_net",
    "out_AB", "out_BA", "out_node_flow",
]
ISA_ARG_NAMES = [
    "o_terminal_idxs", "o_terminal_weights",
    "adjacency_pointer", "adjacency_vector", "adjacency_vector_weights",
    "adjacynct_vector_network_node",
    "d_terminal_idxs", "d_terminal_weights", "d_weights",
    "gravity_beta", "gravity_plateau", "gravity_logistic_midpoint",
    "gravity_growth_rate", "knn_decay", "knn_weights", "cutoff",
]
CTV_ARG_NAMES = [
    "o_weight", "dest_weights", "d_shortest_arr", "radius", "gravity_beta",
    "decay_on", "decay_curve", "use_nearest", "use_d_weights",
    "decay_method", "gravity_cap",
]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# RAM gate (abort BEFORE any heavy work)
# --------------------------------------------------------------------------
import psutil  # noqa: E402

vm = psutil.virtual_memory()
RAM = {
    "available_mib": round(vm.available / (1 << 20), 1),
    "phys_mib": round(vm.total / (1 << 20), 1),
    "threshold_mib": round(max(1 << 30, vm.total // 10) / (1 << 20), 1),
}
if vm.available < max(1 << 30, vm.total // 10):
    print(f"[capture] ABORT: available RAM {RAM['available_mib']} MiB < "
          f"threshold {RAM['threshold_mib']} MiB", flush=True)
    sys.exit(3)

# --------------------------------------------------------------------------
# Provenance checks BEFORE importing B0
# --------------------------------------------------------------------------
assert sha256_file(FIXTURE) == FIXTURE_SHA, "O2 fixture hash mismatch"
assert os.path.abspath(sys.argv[0]) == os.path.join(OUT_DIR, "capture_observed.py")

# --------------------------------------------------------------------------
# Capture store
# --------------------------------------------------------------------------
class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.arrays = {}        # key -> np.ndarray (copied, owned)
        self.meta = {}          # key -> dict
        self.by_hash = {}       # dedup key -> key
        self._n = 0

    def _key(self):
        k = f"a{self._n:05d}"
        self._n += 1
        return k

    def put(self, arr):
        """Dedup-store a COPY of arr; return (key, meta)."""
        a = np.asarray(arr)
        if not a.flags.c_contiguous and not a.flags.f_contiguous:
            a = np.ascontiguousarray(a)
        digest = sha256_bytes(a.dtype.str.encode() +
                              str(a.shape).encode() + a.tobytes())
        dk = (a.dtype.str, a.shape, digest)
        with self.lock:
            key = self.by_hash.get(dk)
            if key is None:
                key = self._key()
                self.arrays[key] = a.copy()
                self.by_hash[dk] = key
                self.meta[key] = {
                    "dtype": a.dtype.str, "dtype_name": str(a.dtype),
                    "shape": list(a.shape), "ndim": a.ndim,
                    "nbytes": int(a.nbytes),
                    "c_contiguous": bool(a.flags.c_contiguous),
                    "f_contiguous": bool(a.flags.f_contiguous),
                    "sha256": digest,
                }
        return key, self.meta[key]

    def ref(self, arr):
        """Hash WITHOUT storing (cheap ledger reference)."""
        a = np.asarray(arr)
        return sha256_bytes(a.dtype.str.encode() +
                            str(a.shape).encode() + a.tobytes())


STORE = Store()
LEDGER = {
    "integrated_scope_access": [],
    "od_compact_vector_node_view_scope": [],
    "accumulate_od_flow": [],
    "compute_trip_volumes": [],
    "precompute_dest_gradients": [],
    "cutoff_for_shortest": [],
    "not_called": {},
}


def wrap_module_kernel(mod, name, recorder):
    orig = getattr(mod, name)

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter_ns()
        out = orig(*args, **kwargs)
        wall = time.perf_counter_ns() - t0
        recorder(args, kwargs, out, wall)
        return out

    wrapper._h03_unwrapped = orig
    setattr(mod, name, wrapper)
    return orig


def rec_isa(module_name, output_names=None):
    out_names = output_names or ["reach", "gravity_exponential",
                                 "gravity_logistic", "knn_access"]

    def recorder(args, kwargs, out, wall):
        bound = dict(zip(ISA_ARG_NAMES, args))
        bound.update(kwargs)
        rec = {"module": module_name, "wall_ns": wall, "args": {}, "scalars": {}}
        for k, v in bound.items():
            if isinstance(v, np.ndarray):
                key, _ = STORE.put(v)
                rec["args"][k] = key
            else:
                rec["scalars"][k] = v if not isinstance(v, np.generic) else v.item()
        rec["outputs"] = {}
        for i, o in enumerate(out):
            key, _ = STORE.put(o)
            rec["outputs"][out_names[i] if i < len(out_names) else f"out{i}"] = key
        LEDGER["integrated_scope_access"].append(rec)
    return recorder


def rec_od(args, kwargs, out, wall):
    a = dict(zip(OD_ARG_NAMES, args))
    ovn = a["origin_virtual_node"]
    with SELECT.lock:
        SELECT.group_calls[ovn] = SELECT.group_calls.get(ovn, 0) + 1
        selected = SELECT.group_calls[ovn] <= N_SEL_PER_ORIGIN
    rec = {
        "seq": SELECT.next_seq(),
        "selected": selected,
        "scalars": {
            "origin_virtual_node": int(a["origin_virtual_node"]),
            "dest_virtual_node": int(a["dest_virtual_node"]),
            "o_edge_id": int(a["o_edge_id"]),
            "d_edge_id": int(a["d_edge_id"]),
            "d_shortest": float(a["d_shortest"]),
            "budget": float(a["budget"]),
            "decay_curve_id": int(a["decay_curve_id"]),
            "decay_beta": float(a["decay_beta"]),
            "decay_midpoint": float(a["decay_midpoint"]),
            "trip_volume": float(a["trip_volume"]),
            "n_net": int(a["n_net"]),
            "out_node_size": int(np.asarray(a["out_node_flow"]).shape[0]),
        },
        "d_o_ref": STORE.ref(a["d_o"]),
        "wall_ns": wall,
        "delivered_live": float(out),
    }
    if selected:
        # Stash copies of every input BEFORE the zero-buffer replay (the
        # driver resets dd_buf/pd_buf right after the live call).
        stash = {k: np.array(a[k], copy=True)
                 for k in ("indptr", "indices", "weights", "edge_id_of_arc",
                           "dir_of_arc", "d_o", "d_d", "pred_o", "pred_d")}
        rec["args"] = {}
        for k in ("indptr", "indices", "weights", "edge_id_of_arc",
                  "dir_of_arc", "d_o", "d_d", "pred_o", "pred_d"):
            key, _ = STORE.put(stash[k])
            rec["args"][k] = key
        # Zero-buffer replay: exact per-call output delta.
        z_AB = np.zeros_like(a["out_AB"])
        z_BA = np.zeros_like(a["out_BA"])
        z_node = np.zeros_like(a["out_node_flow"])
        t0 = time.perf_counter_ns()
        delivered_zero = SELECT.orig(
            stash["indptr"], stash["indices"], stash["weights"],
            stash["edge_id_of_arc"], stash["dir_of_arc"],
            stash["d_o"], stash["d_d"], stash["pred_o"], stash["pred_d"],
            int(a["origin_virtual_node"]), int(a["dest_virtual_node"]),
            int(a["o_edge_id"]), int(a["d_edge_id"]),
            float(a["d_shortest"]), float(a["budget"]),
            int(a["decay_curve_id"]), float(a["decay_beta"]),
            float(a["decay_midpoint"]), float(a["trip_volume"]),
            int(a["n_net"]),
            z_AB, z_BA, z_node,
        )
        rec["replay_wall_ns"] = time.perf_counter_ns() - t0
        if float(delivered_zero) != float(out):
            raise AssertionError(
                f"delivered mismatch live={out!r} zero-replay={delivered_zero!r}")
        rec["delivered_zero"] = float(delivered_zero)
        rec["outputs"] = {}
        for name, arr in (("out_AB", z_AB), ("out_BA", z_BA),
                          ("out_node_flow", z_node)):
            key, _ = STORE.put(arr)
            rec["outputs"][name] = key
    LEDGER["accumulate_od_flow"].append(rec)


class _Sel:
    lock = threading.Lock()
    group_calls = {}
    _seq = [0]
    orig = None

    @classmethod
    def next_seq(cls):
        with cls.lock:
            cls._seq[0] += 1
            return cls._seq[0] - 1


SELECT = _Sel()


def rec_ctv(args, kwargs, out, wall):
    bound = dict(zip(CTV_ARG_NAMES, args))
    rec = {"wall_ns": wall, "scalars": {}, "args": {}, "output": None}
    for k, v in bound.items():
        if isinstance(v, np.ndarray):
            key, _ = STORE.put(v)
            rec["args"][k] = key
        elif isinstance(v, str):
            rec["scalars"][k] = v
        else:
            rec["scalars"][k] = v if not isinstance(v, np.generic) else v.item()
    key, _ = STORE.put(out)
    rec["output"] = key
    LEDGER["compute_trip_volumes"].append(rec)


def rec_cutoff(args, kwargs, out, wall):
    LEDGER["cutoff_for_shortest"].append({
        "args": [a if isinstance(a, (int, float, str)) else repr(a) for a in args],
        "result": float(out), "wall_ns": wall,
    })


def main():
    t_start = time.perf_counter()
    import numba
    import scipy
    import scipy.sparse.csgraph as _csg

    # ---- import B0 (PYTHONPATH set by caller) -----------------------------
    import urban_network_analysis
    from urban_network_analysis import UNA
    from urban_network_analysis.Engines import Accessibility as MOD_ACC
    from urban_network_analysis.Engines import (  # noqa: F401
        AccessibilityWElevation as MOD_ACCE)
    from urban_network_analysis.Engines import AggregateFlow as MOD_FLOW

    una_pkg_dir = os.path.dirname(os.path.abspath(urban_network_analysis.__file__))
    assert una_pkg_dir.startswith(B0_SRC), f"UNA imported outside B0: {una_pkg_dir}"

    try:
        layer = numba.threading_layer()
    except Exception as exc:  # not set until a parallel kernel compiled
        layer = f"unavailable: {exc!r}"
    env = {
        "python": sys.version.split()[0],
        "numba": numba.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba_num_threads": numba.get_num_threads(),
        "numba_threading_layer": layer,
        "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR"),
        "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
    }

    # ---- attach capture wrappers ------------------------------------------
    # Accessibility family (base module): expected ZERO calls at O2
    # (UNA dispatches non-turn runs to AccessibilityWElevation, whose
    # kernel family is a separate, duplicated njit family).
    wrap_module_kernel(MOD_ACC, "integrated_scope_access", rec_isa("Accessibility"))
    wrap_module_kernel(MOD_ACC, "od_compact_vector_node_view_scope",
                       rec_isa("Accessibility.od_compact_vector_node_view_scope",
                               output_names=["od_matrix"]))
    # AccessibilityWElevation family: the O2 accessibility kernel.
    wrap_module_kernel(MOD_ACCE, "integrated_scope_access", rec_isa("AccessibilityWElevation"))
    wrap_module_kernel(MOD_ACCE, "od_compact_vector_node_view_scope",
                       rec_isa("AccessibilityWElevation.od_compact_vector_node_view_scope",
                               output_names=["od_matrix"]))
    # AggregateFlow kernels.
    SELECT.orig = wrap_module_kernel(MOD_FLOW, "_accumulate_od_flow", rec_od)
    wrap_module_kernel(MOD_FLOW, "_compute_trip_volumes", rec_ctv)
    wrap_module_kernel(MOD_FLOW, "_cutoff_for_shortest", rec_cutoff)

    orig_pdg = MOD_FLOW.AggregateFlow._precompute_dest_gradients

    def pdg_wrapper(self, ns):
        t0 = time.perf_counter_ns()
        indptr, nodes, dist, pred = orig_pdg(self, ns)
        wall = time.perf_counter_ns() - t0
        rec = {"wall_ns": wall, "args": {}, "outputs": {}, "scalars": {
            "n_net": int(self._n_network_nodes),
            "n_dest": int(self._n_destinations),
            "limit": float(MOD_FLOW.AggregateFlow._gradient_limit_for(ns)),
        }}
        for name in ("_csr_indptr", "_csr_indices", "_csr_weights",
                     "_csr_edge_id", "_csr_direction"):
            key, _ = STORE.put(getattr(self, name))
            rec["args"][name] = key
        for name in ("_csr_rev",):
            csr = getattr(self, name)
            for part in ("indptr", "indices", "data"):
                key, _ = STORE.put(getattr(csr, part))
                rec["args"][f"{name}_{part}"] = key
        dest_nodes = np.arange(int(self._n_network_nodes),
                               int(self._n_network_nodes) + int(self._n_destinations),
                               dtype=np.int64)
        key, _ = STORE.put(dest_nodes)
        rec["args"]["dest_nodes"] = key
        for name, arr in (("indptr", indptr), ("nodes", nodes),
                          ("dist", dist), ("pred", pred)):
            key, _ = STORE.put(arr)
            rec["outputs"][name] = key
        LEDGER["precompute_dest_gradients"].append(rec)
        return indptr, nodes, dist, pred

    MOD_FLOW.AggregateFlow._precompute_dest_gradients = pdg_wrapper

    # ---- settings: EXACTLY the H04 reviewer's O2 assembly ------------------
    una = UNA(verbosity=1)
    s = una.settings
    s.data_folder = CAMPAIGN_DATA
    s.network_file = "inputs/20260703_PercLenNetwork_InnerCore.geojson"
    s.origins_file = "O2_origins_h04review.geojson"
    s.destinations_file = "inputs/MA_bus_stops.geojson"
    s.network_weight_column = "Geometric"
    s.origin_weight_column = "Count"
    s.destination_weight_column = "weekly_departures"
    s.search_radius = 500
    s.turns = False
    s.elevation = False
    s.calculate_reach = True
    s.calculate_exponential_gravity = True
    s.calculate_logistic_gravity = True
    s.calculate_knn_access = True
    s.output_csv = True
    s.output_geojson = True
    s.output_feather = True
    out_root = os.path.join(CAMPAIGN_DATA, "h03_capture_out")
    os.makedirs(os.path.join(out_root, "acc"), exist_ok=True)
    s.output_folder = os.path.join(out_root, "acc")
    settings_echo = {
        k: getattr(s, k) for k in (
            "network_file", "origins_file", "destinations_file",
            "network_weight_column", "origin_weight_column",
            "destination_weight_column", "search_radius", "turns",
            "elevation", "calculate_reach", "calculate_exponential_gravity",
            "calculate_logistic_gravity", "calculate_knn_access",
            "flow_engine", "flow_detour_ratio", "flow_detour_buffer",
            "flow_detour_mode", "flow_decay", "flow_decay_curve",
            "flow_decay_method", "flow_gravity_cap", "gravity_beta",
            "gravity_logistic_midpoint", "gravity_plateau",
            "gravity_decay_constant", "knn_decay", "knn_weights")
    }

    # ---- RUN 1: RunAccessibility ------------------------------------------
    t0 = time.perf_counter()
    una.RunAccessibility()
    acc_wall = time.perf_counter() - t0
    acc_out = {}
    for name in ("reach", "gravity_exponential", "gravity_logistic", "knn_access"):
        key, _ = STORE.put(getattr(una.accessibility, name))
        acc_out[name] = key

    # ---- flow settings: EXACTLY the H04 reviewer's call-2 assembly ---------
    s.flow_engine = "aggregate_flow"
    s.flow_decay = True
    s.flow_decay_method = "gravity_cap"
    s.flow_gravity_cap = "p95"   # string -> RunFlow MUST derive the cap internally
    flow_out_root = os.path.join(CAMPAIGN_DATA, "h03_capture_out", "flow")
    os.makedirs(flow_out_root, exist_ok=True)
    s.output_folder = flow_out_root
    flow_settings_changed = {
        "flow_engine": "aggregate_flow", "flow_decay": True,
        "flow_decay_method": "gravity_cap", "flow_gravity_cap": "p95",
        "output_folder": flow_out_root,
    }

    # ---- RUN 2: RunFlow (auto gravity cap inside) --------------------------
    t0 = time.perf_counter()
    una.RunFlow()
    flow_wall = time.perf_counter() - t0
    flow_out = {}
    for name in ("edge_flow_AB", "edge_flow_BA", "node_flow"):
        arr = getattr(una.flow, name, None)
        if arr is not None:
            key, _ = STORE.put(arr)
            flow_out[name] = key
    resolved_cap = getattr(una, "resolved_gravity_cap", None)
    try:
        env["numba_threading_layer_after_runs"] = numba.threading_layer()
    except Exception as exc:
        env["numba_threading_layer_after_runs"] = f"unavailable: {exc!r}"

    # ---- assemble npz + sidecar --------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    arrays = STORE.arrays
    np.savez_compressed(NPZ_PATH, **arrays)
    npz_sha = sha256_file(NPZ_PATH)
    with open(O2_MANIFEST, "rb") as fh:
        o2_manifest_sha = hashlib.sha256(fh.read()).hexdigest()

    sidecar = {
        "task": "H03-N4 observed kernel-boundary fixture capture",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "capture_wall_s": round(time.perf_counter() - t_start, 2),
        "provenance": {
            "b0_root": B0_SRC,
            "b0_commit": B0_COMMIT,
            "una_imported_from": una_pkg_dir,
            "o2_fixture": FIXTURE,
            "o2_fixture_sha256": FIXTURE_SHA,
            "o2_fixture_derivation": ("/Users/alansynn/orca/workspaces/una-x/"
                                      "wt-large-e2e/campaigns/una_large_e2e/"
                                      "evidence/H04/o2_fixture_derivation.json"),
            "o2_manifest": O2_MANIFEST,
            "o2_manifest_sha256": o2_manifest_sha,
            "reviewer_l2_record": ("/Users/alansynn/orca/workspaces/una-x/"
                                   "campaign_data/h04_observed_l2_raw.json"),
            "settings_echo": settings_echo,
            "flow_settings_changed_before_runflow": flow_settings_changed,
            "resolved_gravity_cap": resolved_cap,
        },
        "ram_gate": RAM,
        "env": env,
        "capture_method": __doc__.strip().split("Outputs (in this directory):")[0].strip(),
        "selection_rule": (f"first {N_SEL_PER_ORIGIN} eligible _accumulate_od_flow "
                           "calls per origin group (origin_virtual_node); "
                           "zero-buffer replay per selected call"),
        "n_sel_per_origin": N_SEL_PER_ORIGIN,
        "walls": {"run_accessibility_s": round(acc_wall, 3),
                  "run_flow_s": round(flow_wall, 3)},
        "arrays": STORE.meta,
        "calls": LEDGER,
        "engine_outputs": {"accessibility": acc_out, "flow": flow_out},
        "npz_bytes": os.path.getsize(NPZ_PATH),
        "npz_sha256": npz_sha,
    }
    with open(SIDECAR_PATH, "w") as fh:
        json.dump(sidecar, fh, indent=1, default=str)
    os.chmod(NPZ_PATH, 0o444)
    os.chmod(SIDECAR_PATH, 0o444)

    n_od = len(LEDGER["accumulate_od_flow"])
    n_sel = sum(1 for r in LEDGER["accumulate_od_flow"] if r["selected"])
    n_arr = len(arrays)
    n_bytes = sum(a.nbytes for a in arrays.values())
    print(f"[capture] RUNNER OK: {n_od} OD kernel calls "
          f"({n_sel} selected+replayed), "
          f"{len(LEDGER['integrated_scope_access'])} integrated_scope_access "
          f"calls, {len(LEDGER['compute_trip_volumes'])} trip-volume calls, "
          f"{len(LEDGER['precompute_dest_gradients'])} gradient captures; "
          f"{n_arr} unique arrays, {n_bytes/1e6:.1f} MB uncompressed; "
          f"npz {os.path.getsize(NPZ_PATH)/1e6:.1f} MB sha256={npz_sha[:16]}...; "
          f"total {time.perf_counter()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
