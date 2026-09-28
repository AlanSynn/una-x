"""A3R paired-screen bare job driver — executed BY AN ARM VENV's python.

Clone of evidence/A1R/a1r_bare_job.py (the binding accessibility bare-driver
skeleton) with the A3R deltas, each marked "# [A3R-delta]" so the
diff-vs-skeleton audit is mechanical. Inherited unchanged: import-identity
realpath assertion, manifest rehash, selection identity shortcut, settings
assembly with PATH_KEYS, exactly ONE public call inside perf_counter_ns,
observation strictly after the window (arrays, counts, warnings, artifacts,
ru_maxrss).

A3R deltas (none touch the timed window):
  1. --arm {b0,cand} + per-arm module content-sha pin table (asserted
     in-run, post-window; the cross-arm INEQUALITY of the scratch/engine
     pins is the screened change itself, per the committed A3 set).
  2. Atomic record filing: parent-dir creation in the writing branch +
     TEMP + os.replace (item-16 class).
  3. Route receipts (G5): symbol PRESENCE via hasattr only (never
     executed from this driver) + a numpy replay of the A3 guard's
     admission conditions on inputs reconstructed from the same topology
     tables the engine builds d_terminal_idxs from
     (Accessibility.py :648-649, cand blob); labeled
     replay_on_reconstructed_inputs, corroborative-only.
  4. leg-C init microbench (post-window, after ru_maxrss is read):
     one njit(cache=False) replica of the production init expression
     np.ones(n, float64) + cutoff, timed at n = nd (= V + D, full init,
     b0's production form) and n = V (tail-free init, cand's production
     form); 3 warmups + up to 2000 timed reps under a wall deadline;
     shares vs this run's own window computed in-record.
  5. PYTHONDONTWRITEBYTECODE recorded (supervisor sets it); cache entry
     COUNTS remain a supervisor obligation — this driver records paths
     only.

One process = one run. The supervisor (a3r_run_pairs.py) owns scheduling,
cache-root freshness/wipe, (W,H) env, lease windows, timeouts, census.
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--cell", required=True)
ap.add_argument("--expected-site-packages", required=True)
ap.add_argument("--out-json", required=True)
ap.add_argument("--output-root", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--arm", required=True, choices=["b0", "cand"])  # [A3R-delta 1]
args = ap.parse_args()

# [A3R-delta 1] per-arm module content pins (sha256 of installed file bytes;
# values are the in-session git cat-file blob derivations recorded in
# screen_spec.json source_identity_and_arms; b0 = 9b1340e, cand = f17184b).
ARM_MODULE_PINS = {
    "b0": {
        "_large_access_scratch.py":
            "2d3b77363a5d42050024e335b5e2bedf8c8088a312d6e163666596ce42894e23",
        "Accessibility.py":
            "5fec7ad5bf00b6b033747ae41ae2a7eec57f53748395ee7b5c0836536b805f75",
        "AccessibilityWElevation.py":
            "83ed0bfdd0e2b0b7d413dc39a2e84899a1b145f6980ddd9a30243f6e76944a9c",
    },
    "cand": {
        "_large_access_scratch.py":
            "a9679b653e6c55e5b88ccbdc79c5576aec1735afdeb994b72d850a2d8579a92b",
        "Accessibility.py":
            "fa3e550b0cd51076a8323b7e45facc395ebf6f2f6816565d8d5d7cb7595c90e6",
        "AccessibilityWElevation.py":
            "bfd65ab6609ad133dd12b25d92f80d237bdd819e66ac9f4ebed63630848e1400",
    },
}

REC = {
    "schema_version": 1, "task": "A3R", "record": "bare_complete_job",
    "role": "screen-executor", "cell": args.cell, "run_id": args.run_id,
    "arm": args.arm,  # [A3R-delta 1]
    "timing_class": "bare_complete_job_no_leaf_profiler",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "env": {"NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
            "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR"),
            "PYTHONDONTWRITEBYTECODE":
                os.environ.get("PYTHONDONTWRITEBYTECODE"),  # [A3R-delta 5]
            "L1_REUSE_DIR": os.environ.get("L1_REUSE_DIR", "<unset>"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", "<unset>")},
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_out():  # [A3R-delta 2] parent-dir + TEMP + atomic replace
    REC["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(out_dir, exist_ok=True)
    tmp = args.out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(REC, f, indent=1)
    os.replace(tmp, args.out_json)


def fatal(msg):
    REC["fatal"] = msg
    write_out()
    print(f"[a3r-job] FATAL: {msg}", flush=True)
    sys.exit(2)


# ---- import identity (before anything else) --------------------------------
import numba  # noqa: E402
import numpy as np  # noqa: E402
import urban_network_analysis as u  # noqa: E402

una_dir = os.path.dirname(os.path.abspath(u.__file__))
exp_real = os.path.realpath(args.expected_site_packages)
una_real = os.path.realpath(una_dir)
identity_ok = una_real.startswith(exp_real + os.sep)
REC["import"] = {
    "urban_network_analysis_file": una_dir,
    "expected_site_packages": args.expected_site_packages,
    "identity_assertion": ("ok (realpath under expected site-packages)" if identity_ok
                           else f"FAILED: {una_real} not under {exp_real}"),
    "package_version": getattr(u, "__version__", None),
}
if not identity_ok:
    fatal("import identity assertion failed")
REC["process"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                  "numba": numba.__version__,
                  "numba_num_threads_config": numba.config.NUMBA_NUM_THREADS,
                  "numba_get_num_threads": numba.get_num_threads(),
                  "numba_threading_layer": numba.threading_layer()}

# ---- manifest + input rehash (outside window) ------------------------------
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
    fatal("input_rehash_mismatch")
REC["manifest"] = {"path": os.path.abspath(args.manifest),
                   "sha256": sha256_file(args.manifest),
                   "workload_id": manifest.get("workload_id"),
                   "settings_overrides": manifest.get("settings_overrides")}

# ---- origin selection: verify + identity shortcut ---------------------------
sel_spec = manifest.get("origin_selection")
if not sel_spec:
    fatal("manifest has no origin_selection block")
sel = sel_spec
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(args.manifest),
                                         "..", "..", ".."))
sel_path = os.path.join(REPO_ROOT, sel["selection_manifest"])
sel_sha = sha256_file(sel_path)
if sel_sha != sel.get("selection_manifest_sha256"):
    fatal("selection manifest sha mismatch vs parent workload manifest")
with open(sel_path) as f:
    selman = json.load(f)
if selman.get("source_sha256") != manifest["inputs"]["origins"]["sha256"]:
    fatal("selection manifest source_sha256 != manifest inputs.origins.sha256")
ordered = selman["ordered_indices"]
n_src = manifest["inputs"]["origins"]["feature_count"]
if (len(ordered) != sel["selected_count"] or len(set(ordered)) != len(ordered)
        or any(not isinstance(i, int) or not (0 <= i < n_src) for i in ordered)):
    fatal("selection manifest ordered_indices invalid")
if ordered != list(range(n_src)):
    fatal("non-identity selection not supported by bare driver; this screen "
          "uses sel_all identity cells only")
REC["origin_selection"] = {
    "selection_manifest": sel_path, "selection_manifest_sha256": sel_sha,
    "selected_count": sel["selected_count"],
    "workload_variant": selman.get("workload_variant"),
    "fixture": "none — ordered_indices is the identity; verified source file "
               "used directly, no rewrite"}

# ---- construct UNA + settings assembly (outside window) ---------------------
from urban_network_analysis import UNA  # noqa: E402

una = UNA(verbosity=1)
REC["constructor"] = {"call": "UNA(verbosity=1)"}
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
st.origins_file = os.path.basename(manifest["inputs"]["origins"]["local_path"])
st.destinations_file = os.path.basename(
    manifest["inputs"]["destinations"]["local_path"])
os.makedirs(args.output_root, exist_ok=True)
st.output_folder = args.output_root
REC["settings_effective_search_radius"] = getattr(st, "search_radius", None)

# ---- THE complete-job window: exactly ONE public call, nothing else ---------
t0 = time.perf_counter_ns()
una.RunAccessibility()
t1 = time.perf_counter_ns()
REC["window"] = {"application_window_ns": t1 - t0,
                 "boundaries": "perf_counter_ns around the single "
                               "una.RunAccessibility() call incl. synchronous "
                               "exports; no instrumentation inside"}

# ---- observation (all AFTER the window) -------------------------------------
def arr_rec(a, k=8):
    if a is None:
        return None
    a = np.asarray(a)
    flat = a.reshape(-1)
    return {
        "dtype": str(a.dtype), "dtype_str": a.dtype.str, "shape": list(a.shape),
        "first_k": [float(x) if a.dtype.kind == "f" else int(x)
                    for x in flat[:k]],
        "sha256_bytes": hashlib.sha256(a.tobytes()).hexdigest() if a.size else None,
        "n_nonfinite": int((~np.isfinite(flat)).sum()) if a.dtype.kind == "f" and a.size else 0,
    }


acc = una.accessibility
REC["accessibility_arrays"] = {
    m: arr_rec(getattr(acc, m, None))
    for m in ("reach", "gravity_exponential", "gravity_logistic", "knn_access")}
REC["counts"] = {
    "V_nodes": int(una.topology.network.node_points.shape[0]),
    "E_edges": int(una.topology.network.lengths.shape[0]),
    "O_rows": int(len(una.topology.origins.geometry)),
    "D_rows": int(len(una.topology.destinations.geometry)),
}
ent = list(una.topology.logger.log_list)
REC["verbatim_warnings"] = [
    e for e in ent
    if "WARN" in str(e.get("event", "")).upper()
    or "WARN" in str(e.get("details", "")).upper()]
artifacts = {}
for dirpath, _, files in os.walk(args.output_root):
    for fn in files:
        p = os.path.join(dirpath, fn)
        artifacts[os.path.relpath(p, args.output_root)] = {
            "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
REC["artifacts"] = artifacts
REC["peak_rss_ru_maxrss_mib"] = resource.getrusage(
    resource.RUSAGE_SELF).ru_maxrss / (1 << 20)  # macOS: bytes

# ---- [A3R-delta 1] in-run module pin assertion (post-window) ----------------
import importlib  # noqa: E402
import importlib.util  # noqa: E402

engines_dir = os.path.join(una_real, "Engines")
pin_checks = {}
for fname, expect in ARM_MODULE_PINS[args.arm].items():
    p = os.path.join(engines_dir, fname)
    got = sha256_file(p) if os.path.isfile(p) else None
    pin_checks[fname] = {"expected": expect, "observed": got,
                         "match": got == expect}
REC["module_pins"] = {"engines_dir": engines_dir, "checks": pin_checks,
                      "all_match": all(c["match"] for c in pin_checks.values())}
if not REC["module_pins"]["all_match"]:
    fatal("module pin mismatch for arm " + args.arm)

# ---- [A3R-delta 3] route receipts (presence + replay; nothing executed) -----
scratch = importlib.import_module(
    "urban_network_analysis.Engines._large_access_scratch")
presence = {sym: hasattr(scratch, sym)
            for sym in ("_a3_tail_admits", "_a3_scope_search_tailless",
                        "_a1_scope_admits", "_a1_scope_search")}
net = una.topology.network
dst = una.topology.destinations
V = REC["counts"]["V_nodes"]
D = REC["counts"]["D_rows"]
replay = {"note": "replay_on_reconstructed_inputs — d_terminal_idxs is not "
                  "persisted post-window; reconstructed from the same "
                  "topology tables the engine builds it from "
                  "(Engines/Accessibility.py :648-649, cand blob)",
          "has_edge_start_node": hasattr(dst, "edge_start_node"),
          "has_edge_end_node": hasattr(dst, "edge_end_node")}
if replay["has_edge_start_node"] and replay["has_edge_end_node"]:
    ds = np.asarray(dst.edge_start_node)
    de = np.asarray(dst.edge_end_node)
    shape_ok = (ds.ndim == 1 and de.ndim == 1 and ds.shape[0] == de.shape[0]
                and int(ds.shape[0]) == D)
    range_ok = bool(ds.size and de.size and int(ds.min()) >= 0
                    and int(de.min()) >= 0
                    and int(ds.max()) < V and int(de.max()) < V)
    replay.update({"replay_D": int(ds.shape[0]), "shape_ok": shape_ok,
                   "range_ok": range_ok,
                   "a3_guard_would_admit": bool(shape_ok and range_ok),
                   "replay_min": [int(ds.min()), int(de.min())],
                   "replay_max": [int(ds.max()), int(de.max())],
                   "node_count_used": V})
REC["route_receipts"] = {"symbol_presence": presence,
                         "execution_from_driver": "none — hasattr only",
                         "guard_conditions_replay": replay}

# ---- [A3R-delta 4] leg-C init microbench (post-window, after ru_maxrss) -----
import numba as nb  # noqa: E402

LEGC_REPS_CAP = 2000
LEGC_BUDGET_S = 5.0


@nb.njit(cache=False)
def _legc_init(n, cutoff):
    v = np.ones(n, dtype=np.float64) + cutoff
    return v[0] + v[n - 1]


def legc_measure(n, cutoff):
    for _ in range(3):
        _legc_init(n, cutoff)
    xs = []
    deadline = time.perf_counter() + LEGC_BUDGET_S
    for _ in range(LEGC_REPS_CAP):
        a = time.perf_counter_ns()
        _legc_init(n, cutoff)
        b = time.perf_counter_ns()
        xs.append(b - a)
        if time.perf_counter() > deadline:
            break
    xs_sorted = sorted(xs)
    n_r = len(xs)
    mean = sum(xs_sorted) / n_r
    def p(q):
        return xs_sorted[min(n_r - 1, int(q * n_r))]
    return {"n": n_r, "mean_s": mean / 1e9, "median_s": xs_sorted[n_r // 2] / 1e9,
            "p10_s": p(0.10) / 1e9, "p90_s": p(0.90) / 1e9,
            "min_s": xs_sorted[0] / 1e9, "max_s": xs_sorted[-1] / 1e9}


cutoff = float(REC["settings_effective_search_radius"])
V_n = REC["counts"]["V_nodes"]
nd_n = REC["counts"]["V_nodes"] + REC["counts"]["D_rows"]
win_s = REC["window"]["application_window_ns"] / 1e9
O_rows = REC["counts"]["O_rows"]
full = legc_measure(nd_n, cutoff)
tailfree = legc_measure(V_n, cutoff)
for tag, m in (("full", full), ("tailfree", tailfree)):
    m["share_serial_equiv"] = m["mean_s"] * O_rows / win_s
REC["a3_legc"] = {
    "replica": "np.ones(n, dtype=np.float64) + cutoff via one "
               "njit(cache=False) function; n selects the size",
    "dtype_pinned": "float64",
    "dtype_grounding": "A3I t2 record dtype_observed float64; replica vs real "
                       "kernel array-equality 32/32 (admission.json)",
    "cutoff": cutoff,
    "n_full_nd": nd_n, "n_tailfree_V": V_n,
    "nd_identity": "nd = V + D (proof.md section 2) from this record's counts",
    "reps_cap": LEGC_REPS_CAP, "budget_s": LEGC_BUDGET_S,
    "full_nd_init": full, "tailfree_V_init": tailfree,
    "production_expression": {"b0": "full_nd_init", "cand": "tailfree_V_init"}[args.arm],
    "window_s": win_s, "O_rows": O_rows,
    "note": "instrument add-on, strictly post-window and post-ru_maxrss; "
            "realized cross-arm saving is a supervisor-summary quantity",
}

write_out()
print(f"[a3r-job] ok arm={args.arm} win={win_s:.3f}s "
      f"peak={REC['peak_rss_ru_maxrss_mib']:.1f}MiB "
      f"legc_full={full['mean_s'] * 1e6:.1f}us "
      f"legc_tf={tailfree['mean_s'] * 1e6:.1f}us", flush=True)
