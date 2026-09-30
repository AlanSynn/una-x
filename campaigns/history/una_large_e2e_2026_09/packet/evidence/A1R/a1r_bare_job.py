"""A1R paired-screen bare job driver — executed BY AN ARM VENV's python.

Complete-job timing per h04-reviewer's screen spec: the manifest's public
RunAccessibility window with NO leaf instrumentation inside the timed window
(no sampler thread, no span patches, no comparator). Identity is asserted,
inputs are rehashed, and all observation (arrays, artifacts, logger, peak RSS)
happens AFTER the window.

One process = one run. The supervisor (a1r_run_pairs.py) owns scheduling,
cache freshness, (W,H) env, lease windows, and timeouts.
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
args = ap.parse_args()

REC = {
    "schema_version": 1, "task": "A1R", "record": "bare_complete_job",
    "role": "screen-executor", "cell": args.cell, "run_id": args.run_id,
    "timing_class": "bare_complete_job_no_leaf_profiler",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "env": {"NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
            "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR"),
            "L1_REUSE_DIR": os.environ.get("L1_REUSE_DIR", "<unset>"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", "<unset>")},
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_out():
    REC["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(args.out_json, "w") as f:
        json.dump(REC, f, indent=1)


def fatal(msg):
    REC["fatal"] = msg
    write_out()
    print(f"[a1r-job] FATAL: {msg}", flush=True)
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
write_out()
print(f"[a1r-job] ok win={(t1 - t0) / 1e9:.3f}s "
      f"peak={REC['peak_rss_ru_maxrss_mib']:.1f}MiB", flush=True)
