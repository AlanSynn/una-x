"""F1R paired-screen bare job driver — executed BY AN ARM VENV's python.

Vehicle per screen_spec rev 4 (P4b): authored on the evidence/A1R/a1r_bare_job.py
skeleton, h04-reviewer-verified before any gate-eligible window, coordinator-
committed. One process = one run; the supervisor (f1r_run_pairs.py, authored
next) owns scheduling, cache freshness, the (W,K,H) env, lease windows, the
watchdog, and timeouts.

Flow adaptations against the A1R skeleton (screen_spec rev 4 obligations):
  - --flow-stripes wiring: una.topology.num_threads = K BEFORE the public call
    (public attribute the AggregateFlow engine reads at construction via
    Base.num_threads = topology.num_threads; mirrors harness worker.py:498-500;
    NOT a source change).
  - Three-field engine_config_observed block recorded from the LIVE objects
    (flow_stripes_requested echo + topology_num_threads + engine_num_threads,
    plus topology_num_clusters), re-read AFTER the run — the worker.py:522-531
    equivalents, which the bare path must emit itself.
  - ONE public call in the window: una.RunFlow() incl. synchronous exports and
    the in-call auto gravity-cap resolution (never pre-resolved).
  - sel1024 is a NON-identity selection: pinned selection manifest -> geopandas
    fixture derivation written INTO settings.data_folder (campaign_data/inputs,
    h05_profile.py precedent — Topology.py:198 joins data_folder + origins_file
    inside RunFlow), outside the window; all three input paths are
    existence-checked pre-window so a resolution failure can never burn a
    timed window.
  - Input rehash, byte hashes (arrays + artifacts), ru_maxrss, loadavg per run;
    numba H-pin verified in-run (config and get_num_threads == 1).
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
ap.add_argument("--cell", required=True, choices=["O3_FLOW"],
                help="screen cell (rev-4 spec driver parameter)")
ap.add_argument("--variant-id", required=True,
                help="origin_selection variant (F1R screen: sel1024)")
ap.add_argument("--expected-site-packages", required=True)
ap.add_argument("--out-json", required=True)
ap.add_argument("--output-root", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--flow-stripes", type=int, required=True,
                help="K: topology.num_threads / AggregateFlow stripe executor")
ap.add_argument("--wheel-sha256", required=True)
ap.add_argument("--numba-threads", type=int, required=True,
                help="H pin; verified in-run against numba config/get_num_threads")
ap.add_argument("--block", type=int, required=True, choices=[0, 1, 2, 3],
                help="paired-block number (schedule label; 0 = diagnostics-"
                     "class canary, never a gate record)")
ap.add_argument("--order", required=True, choices=["AB", "BA", "CANARY"],
                help="block arm order (schedule label; A = b0 runs first in "
                     "AB; CANARY = diagnostics-class canary run)")
args = ap.parse_args()

REC = {
    "schema_version": 1, "task": "F1R", "record": "bare_complete_job_flow",
    "role": "screen-executor", "cell": args.cell, "arm": args.arm,
    "run_id": args.run_id,
    "block": args.block, "order": args.order,
    "gate_eligible": args.block != 0,
    "diagnostics_class": ("memory_canary" if args.block == 0 else None),
    "cold_warm": ("diagnostic (memory canary, throwaway cache)" if args.block == 0
                  else "cold (fresh per-arm cache)" if args.block == 1
                  else "warm (reused cache)"),
    "timing_class": "bare_complete_job_no_leaf_profiler",
    "frozen_triple": {"W": 1, "K": args.flow_stripes, "H": args.numba_threads},
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
    print(f"[f1r-job] FATAL: {msg}", flush=True)
    sys.exit(2)


# ---- import identity + H pin (before numerical work) ------------------------
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
    "wheel_sha256": args.wheel_sha256,
}
if not identity_ok:
    fatal("import identity assertion failed")
h_config = int(numba.config.NUMBA_NUM_THREADS)
h_live = int(numba.get_num_threads())
REC["process"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                  "numba": numba.__version__,
                  "numba_num_threads_config": h_config,
                  "numba_get_num_threads": h_live,
                  "numba_threading_layer": numba.threading_layer()}
if h_config != args.numba_threads or h_live != args.numba_threads:
    fatal(f"numba H pin mismatch: config={h_config} live={h_live} "
          f"expected={args.numba_threads}")

# ---- manifest + pinned sha + input rehash (outside window) ------------------
FROZEN_MANIFEST_SHA = "c2e6f2669cf334e4060d3930d23d44baf41efa5c54f36908cd2b72b25ec96e2a"
with open(args.manifest) as f:
    manifest = json.load(f)
man_sha = sha256_file(args.manifest)
if man_sha != FROZEN_MANIFEST_SHA:
    fatal(f"manifest sha {man_sha} != frozen O3_FLOW workload sha")
if manifest.get("settings_overrides", {}).get("flow_engine") != "aggregate_flow":
    fatal("manifest is not an aggregate_flow workload")
rehash = {}
for key, spec in manifest["inputs"].items():
    p = spec["local_path"]
    h, b = sha256_file(p), os.path.getsize(p)
    rehash[key] = {"sha256": h, "bytes": b,
                   "match": h == spec["sha256"] and b == spec["bytes"]}
REC["input_rehash"] = rehash
if not all(v["match"] for v in rehash.values()):
    fatal("input_rehash_mismatch")
REC["manifest"] = {"path": os.path.abspath(args.manifest), "sha256": man_sha,
                   "workload_id": manifest.get("workload_id"),
                   "settings_overrides": manifest.get("settings_overrides")}

# ---- origin selection: pinned sel1024 -> fixture (outside window) -----------
sel_spec = manifest.get("origin_selection")
vids = [v["variant_id"] for v in sel_spec.get("variants", [])]
if args.variant_id not in vids:
    fatal(f"--variant-id must be one of {vids}")
sel = next(v for v in sel_spec["variants"] if v["variant_id"] == args.variant_id)
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
    fatal("selection manifest ordered_indices invalid vs selected_count/feature_count")
sel_rec = {"selection_manifest": sel_path, "selection_manifest_sha256": sel_sha,
           "selected_count": sel["selected_count"], "chosen_variant": args.variant_id,
           "workload_variant": selman.get("workload_variant")}
# settings.data_folder (same single source the settings assembly below uses):
# the fixture MUST live there — Topology.py:198 resolves origins as
# os.path.join(settings.data_folder, settings.origins_file) INSIDE RunFlow.
data_folder = os.path.dirname(manifest["inputs"]["network"]["local_path"])
if ordered == list(range(n_src)):
    origins_path = manifest["inputs"]["origins"]["local_path"]
    sel_rec["fixture"] = "none — ordered_indices is the identity"
else:
    import geopandas as gpd
    src = gpd.read_file(manifest["inputs"]["origins"]["local_path"])
    if len(src) != n_src:
        fatal(f"origins source has {len(src)} rows, manifest says {n_src}")
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
                    fatal(f"fixture row {i} attr {col}: {gv!r} != selection manifest {wv!r}")
        sel_rec["attribute_crosscheck"] = "fid/id/lon/lat row-by-row vs selection manifest: ok"
    else:
        sel_rec["attribute_crosscheck"] = (
            "not available: sel1024 selection manifest carries no "
            "selected_row_attributes; fixture identity anchored by selection "
            "manifest sha pin + source_sha256 chain + ordered-index validity")
    fixture_dir = data_folder  # h05_profile.py precedent: fixtures in inputs dir
    os.makedirs(fixture_dir, exist_ok=True)
    fixture = os.path.join(fixture_dir, f"f1r_origins_{args.run_id}_{len(fx)}.geojson")
    fx.to_file(fixture, driver="GeoJSON")
    origins_path = fixture
    sel_rec["fixture"] = {"path": fixture, "sha256": sha256_file(fixture),
                          "rows": int(len(fx)),
                          "note": "derived in-run from pinned selection manifest into "
                                  "settings.data_folder (campaign_data/inputs, h05 "
                                  "precedent); attributes sliced untouched, row order "
                                  "= ordered_indices"}
REC["origin_selection"] = sel_rec

# ---- construct UNA + settings assembly (OUTSIDE window) ---------------------
from urban_network_analysis import UNA  # noqa: E402

una = UNA(verbosity=1)
REC["constructor"] = {"call": "UNA(verbosity=1)", "note": "outside window; counted once"}
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
st.data_folder = data_folder
st.network_file = os.path.basename(manifest["inputs"]["network"]["local_path"])
st.origins_file = os.path.basename(origins_path)
st.destinations_file = os.path.basename(
    manifest["inputs"]["destinations"]["local_path"])
os.makedirs(args.output_root, exist_ok=True)
st.output_folder = args.output_root
REC["settings_effective"] = {"search_radius": getattr(st, "search_radius", None),
                             "flow_engine": getattr(st, "flow_engine", None),
                             "flow_decay_method": getattr(st, "flow_decay_method", None),
                             "flow_gravity_cap": getattr(st, "flow_gravity_cap", None)}

# ---- K wiring: public topology attribute, BEFORE the public call ------------
# Mirrors harness worker.py:498-500. The engine (constructed inside RunFlow)
# inherits num_threads from topology at construction (Base.__init__).
una.topology.num_threads = int(args.flow_stripes)
REC["pre_window_wiring"] = {
    "flow_stripes_requested": int(args.flow_stripes),
    "topology_num_threads_after_wiring": int(una.topology.num_threads),
    "mechanism": "public attribute una.topology.num_threads = K before RunFlow; "
                 "AggregateFlow inherits it at construction (worker.py:498-500 "
                 "equivalent; not a source change)"}
if int(una.topology.num_threads) != int(args.flow_stripes):
    fatal("topology.num_threads wiring did not stick")

# ---- pre-window input resolution (Topology.py:198 semantics) ----------------
# os.path.join(data_folder, <file>) must resolve for ALL THREE inputs BEFORE
# t0; a missing input fatals out here instead of burning a window inside
# RunFlow's AddOrigins.
pre_res = {}
for label, fn in (("network", st.network_file), ("origins", st.origins_file),
                  ("destinations", st.destinations_file)):
    resolved = os.path.join(st.data_folder, fn)
    pre_res[label] = resolved
    if not os.path.isfile(resolved):
        fatal(f"pre-window input resolution failed ({label}): {resolved}")
REC["pre_window_input_resolution"] = pre_res

# ---- THE complete-job window: exactly ONE public call, nothing else ---------
t0 = time.perf_counter_ns()
una.RunFlow()
t1 = time.perf_counter_ns()
REC["window"] = {"application_window_ns": t1 - t0,
                 "boundaries": "perf_counter_ns around the single "
                               "una.RunFlow() call incl. synchronous exports and "
                               "in-call auto gravity-cap resolution; no "
                               "instrumentation inside"}

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


fe = una.flow
REC["flow_arrays"] = {
    "edge_flow": arr_rec(getattr(fe, "edge_flow", None)),
    "edge_flow_AB": arr_rec(getattr(fe, "edge_flow_AB", None)),
    "edge_flow_BA": arr_rec(getattr(fe, "edge_flow_BA", None)),
    "node_flow": arr_rec(getattr(fe, "node_flow", None)),
}
REC["engine_config_observed"] = {
    "flow_stripes_requested": int(args.flow_stripes),
    "topology_num_threads": getattr(getattr(una, "topology", None),
                                    "num_threads", None),
    "engine_num_threads": getattr(getattr(una, "flow", None),
                                  "num_threads", None),
    "topology_num_clusters": getattr(getattr(una, "topology", None),
                                     "num_clusters", None),
}
REC["resolved_gravity_cap"] = getattr(una, "resolved_gravity_cap", None)
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
    resource.RUSAGE_SELF).ru_maxrss / (1 << 20)  # macOS reports bytes
REC["loadavg_after_window"] = os.getloadavg()
REC["numba_threads_after_run"] = {"config": int(numba.config.NUMBA_NUM_THREADS),
                                  "get": int(numba.get_num_threads())}
write_out()
print(f"[f1r-job] ok win={(t1 - t0) / 1e9:.3f}s "
      f"peak={REC['peak_rss_ru_maxrss_mib']:.1f}MiB "
      f"config_observed={REC['engine_config_observed']}", flush=True)
