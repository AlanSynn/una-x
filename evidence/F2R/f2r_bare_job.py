"""F2R paired-screen bare job driver — executed BY THE SINGLE WHEEL VENV's python.

Vehicle per evidence/F2R/screen_spec.json (P3/P4b): authored on the
campaigns/una_large_e2e/evidence/F1R/f1r_bare_job.py skeleton (BINDING
SKELETON per screen_spec vehicle_and_review_chain.P3), h04-reviewer-verified
line-by-line before any gate-eligible window, coordinator-committed. One
process = one run; the supervisor (f2r_run_pairs.py) owns scheduling, the
(W,K,H) env, per-arm cache roots, REV-9 quiet-window gating, lease windows,
the watchdog, and timeouts. Same-binary differential (central ruling): ONE
wheel from git archive of 61f3db6 in ONE fresh venv; BOTH arms run the same
install; ONE fresh interpreter per arm per block.

CLONE-ROT vs f1r_bare_job.py (341 lines), stated per the P3 requirement:
  KEPT (production path unchanged): import-identity realpath fatal; H pin
    (config AND get_num_threads) before numerical work; frozen manifest sha
    constant + full input rehash outside window; aggregate_flow guard;
    sel1024 non-identity selection with pinned selection-manifest sha chain,
    geopandas fixture derived INTO settings.data_folder, row-attribute
    crosscheck; settings assembly; K wiring via public una.topology.num_threads;
    pre-window isfile resolution of all three inputs; window = perf_counter_ns
    around the SINGLE una.RunFlow() call; post-window-only observation
    (array records, engine_config_observed three-field block, counts,
    verbatim_warnings, artifacts walk, ru_maxrss, loadavg).
  CHANGED: UNA(verbosity=1) -> verbosity=2 (the [F2]-era log line rides and
    the full log is persisted); --arm choices b0|cand (same-binary arms, not
    per-arm builds); fixture prefix f1r_origins_ -> f2r_origins_; task/record
    strings F1R -> F2R; run-id pattern f2r_b<N>_<ORDER>_<arm>.
  ADDED (F2R obligations absent from the skeleton):
    - MODULE PIN: installed AggregateFlow.py file sha256 asserted == the
      frozen pin 1f3cf4d2... in-run, BOTH arms (condition 4), alongside the
      package realpath-under-site-packages assertion.
    - LAUNCHER (b0 arm only): --launcher-path/--launcher-sha256; launcher
      file sha asserted BEFORE loading; install() BEFORE the package import;
      install receipt recorded; PRE-WINDOW invoke_real alive receipt on
      declared args (True, 1, 1); POST-WINDOW live-inputs invoke_real
      differential receipt from self._f2_stats + live node count (see the
      launcher docstring's P4b note). Cand arm: no launcher, default import.
    - ROUTE RECEIPTS (G7): self._f2_stats recorded both arms (fast_calls,
      overflow_events, fallback_calls, local_route_ok, local_slice_max);
      b0 contradiction fast_calls != 0 = FATAL (launcher not engaged);
      cand fast_calls == 0 = FATAL (stop-and-preserve); overflow/fallback
      recorded WITHOUT auto-reject (G7 of-policy adjudication is upstream).
    - [F2]-LINE ENFORCEMENT: the [F2] log line must be present in BOTH arms;
      a cand record with NO [F2] line = instrument defect, immediate stop
      (driver obligation); the full logger log_list is persisted.
    - NUMBA_CACHE_DIR CENSUS: cache-root entry count AND sorted name-set
      recorded pre-window and post-window (cold condition 5 evidence; the
      supervisor owns the wipe and the per-arm roots).
  REMOVED: nothing (no skeleton capability dropped).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--cell", required=True, choices=["O3_FLOW"],
                help="screen cell (screen_spec driver parameter)")
ap.add_argument("--variant-id", required=True,
                help="origin_selection variant (F2R screen: sel1024)")
ap.add_argument("--expected-site-packages", required=True)
ap.add_argument("--out-json", required=True)
ap.add_argument("--output-root", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--arm", required=True, choices=["b0", "cand"],
                help="same-binary differential arm: b0 = launcher-forced "
                     "route OFF; cand = default import, route ON")
ap.add_argument("--launcher-path", default=None,
                help="b0 ONLY: f2r_route_launcher.py path")
ap.add_argument("--launcher-sha256", default=None,
                help="b0 ONLY: expected launcher file sha256")
ap.add_argument("--module-pin-sha256",
                default="1f3cf4d2725320569c1b19bf339a39d0bf013d1cb7b6f8e5d72526a77531797c",
                help="frozen AggregateFlow module pin (screen_spec)")
ap.add_argument("--flow-stripes", type=int, required=True,
                help="K: topology.num_threads / AggregateFlow stripe executor")
ap.add_argument("--wheel-sha256", required=True)
ap.add_argument("--numba-threads", type=int, required=True,
                help="H pin; verified in-run against numba config/get_num_threads")
ap.add_argument("--block", type=int, required=True, choices=[1, 2, 3],
                help="paired-block number (1 = cold, fresh per-arm cache)")
ap.add_argument("--order", required=True, choices=["AB", "BA"],
                help="block arm order (A = b0 runs first in AB)")
args = ap.parse_args()

REC = {
    "schema_version": 1, "task": "F2R", "record": "bare_complete_job_flow",
    "role": "screen-executor", "cell": args.cell, "arm": args.arm,
    "run_id": args.run_id,
    "block": args.block, "order": args.order,
    "gate_eligible": True,
    "cold_warm": ("cold (fresh per-arm cache)" if args.block == 1
                  else "warm (reused cache)"),
    "timing_class": "bare_complete_job_no_leaf_profiler",
    "frozen_triple": {"W": 1, "K": args.flow_stripes, "H": args.numba_threads},
    "os_file_cache_control": "UNCONTROLLED (ambient page cache; declared per "
                             "cold_definition.layers.os_file_cache)",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "env": {"NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
            "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR"),
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
    print(f"[f2r-job] FATAL: {msg}", flush=True)
    sys.exit(2)


def cache_census(root):
    """Entry count + sorted name-set of a NUMBA_CACHE_DIR (condition 5)."""
    if not root or not os.path.isdir(root):
        return {"count": None, "names": [], "note": "root missing"}
    names = sorted(os.listdir(root))
    return {"count": len(names), "names": names}


# ---- b0 launcher: sha-assert, load, install BEFORE the package import ------
LAUNCHER_STATE = None
if args.arm == "b0":
    if not args.launcher_path or not args.launcher_sha256:
        fatal("b0 arm requires --launcher-path and --launcher-sha256")
    if not os.path.isfile(args.launcher_path):
        fatal(f"launcher file missing: {args.launcher_path}")
    lsha = sha256_file(args.launcher_path)
    if lsha != args.launcher_sha256:
        fatal(f"launcher sha {lsha} != pinned {args.launcher_sha256}")
    spec = importlib.util.spec_from_file_location("f2r_route_launcher",
                                                  args.launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    install_receipt = launcher.install()  # wraps _use_local_route pre-import
    if install_receipt["aggregateflow_sha256_at_install"] != args.module_pin_sha256:
        fatal("launcher installed against a module whose sha != the frozen pin")
    if not install_receipt["no_other_attribute_touched"]:
        fatal("launcher receipt reports an attribute beyond _use_local_route "
              "was changed")
    alive = launcher.invoke_real(True, 1, 1)  # PRE-WINDOW, declared args
    if alive["return"] is not True:
        fatal(f"launcher pre-window alive receipt returned {alive!r}, "
              "expected True from the retained real function")
    LAUNCHER_STATE = {"launcher": launcher, "install_receipt": install_receipt,
                      "prewindow_alive_receipt": alive}
    REC["launcher"] = {"path": os.path.abspath(args.launcher_path),
                       "sha256": lsha,
                       "install_receipt": install_receipt,
                       "prewindow_alive_receipt": alive,
                       "mechanism": "route-patch on _use_local_route, DECLARED "
                                    "ONCE, consistent across every block "
                                    "(condition 2); precondition NOT patched"}

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
    "single_binary_note": "ONE wheel from git archive of 61f3db6 in ONE venv; "
                          "both arms same install (central ruling)",
}
if not identity_ok:
    fatal("import identity assertion failed")

# MODULE PIN (condition 4): installed AggregateFlow.py file sha, BOTH arms.
af_path = os.path.join(una_dir, "Engines", "AggregateFlow.py")
af_sha = sha256_file(af_path)
REC["module_pin"] = {"path": af_path, "sha256": af_sha,
                     "expected": args.module_pin_sha256,
                     "match": af_sha == args.module_pin_sha256}
if af_sha != args.module_pin_sha256:
    fatal(f"installed AggregateFlow sha {af_sha} != module pin")

h_config = int(numba.config.NUMBA_NUM_THREADS)
h_live = int(numba.get_num_threads())
REC["process"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                  "numba": numba.__version__,
                  "numba_num_threads_config": h_config,
                  "numba_get_num_threads": h_live}
if h_config != args.numba_threads or h_live != args.numba_threads:
    fatal(f"numba H pin mismatch: config={h_config} live={h_live} "
          f"expected={args.numba_threads}")

# ---- cache census PRE-WINDOW (post-import; imports compile nothing) ---------
nbc_root = os.environ.get("NUMBA_CACHE_DIR")
REC["numba_cache_census"] = {"root": nbc_root, "pre_window": cache_census(nbc_root),
                             "rule": "per-arm roots, never shared; block 1 "
                                     "wiped to 0 files by the SUPERVISOR "
                                     "before pair 1 (declared cold state)"}

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
for key, spec_in in manifest["inputs"].items():
    p = spec_in["local_path"]
    h, b = sha256_file(p), os.path.getsize(p)
    rehash[key] = {"sha256": h, "bytes": b,
                   "match": h == spec_in["sha256"] and b == spec_in["bytes"]}
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
    fixture_dir = data_folder
    os.makedirs(fixture_dir, exist_ok=True)
    fixture = os.path.join(fixture_dir, f"f2r_origins_{args.run_id}_{len(fx)}.geojson")
    fx.to_file(fixture, driver="GeoJSON")
    origins_path = fixture
    sel_rec["fixture"] = {"path": fixture, "sha256": sha256_file(fixture),
                          "rows": int(len(fx)),
                          "note": "derived in-run from pinned selection manifest into "
                                  "settings.data_folder; attributes sliced untouched, "
                                  "row order = ordered_indices"}
REC["origin_selection"] = sel_rec

# ---- construct UNA + settings assembly (OUTSIDE window) ---------------------
from urban_network_analysis import UNA  # noqa: E402

una = UNA(verbosity=2)
REC["constructor"] = {"call": "UNA(verbosity=2)", "note": "outside window; "
                      "[F2]-era log line rides v>=1 and the full log is "
                      "persisted (driver obligation)"}
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
una.topology.num_threads = int(args.flow_stripes)
REC["pre_window_wiring"] = {
    "flow_stripes_requested": int(args.flow_stripes),
    "topology_num_threads_after_wiring": int(una.topology.num_threads),
    "mechanism": "public attribute una.topology.num_threads = K before RunFlow; "
                 "AggregateFlow inherits it at construction"}
if int(una.topology.num_threads) != int(args.flow_stripes):
    fatal("topology.num_threads wiring did not stick")

# ---- pre-window input resolution (Topology path-join semantics) -------------
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


REC["numba_cache_census"]["post_window"] = cache_census(nbc_root)

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

# ---- [F2] log line + full log persistence -----------------------------------
ent = list(una.topology.logger.log_list)
REC["log_lines"] = {"count": len(ent), "entries": ent}
f2_lines = [e for e in ent if "[F2]" in str(e.get("details", ""))]
REC["f2_log_line"] = {"present": bool(f2_lines), "entries": f2_lines}
if not f2_lines:
    fatal("[F2] log line ABSENT — log-persistence instrument defect, "
          "immediate stop (driver obligation; the line is emitted "
          "unconditionally at AggregateFlow.py:1730, both arms)")

# ---- route receipts (G7) ----------------------------------------------------
f2s = getattr(fe, "_f2_stats", None)
REC["route_state"] = {"f2_stats": f2s, "arm": args.arm}
if f2s is None:
    fatal("engine _f2_stats missing — route receipt unavailable, "
          "stop-and-preserve")
if args.arm == "b0":
    if int(f2s["fast_calls"]) != 0:
        fatal(f"b0 CONTRADICTION: fast_calls={f2s['fast_calls']} != 0 — "
              "launcher-forced OFF not engaged, stop-and-preserve")
    # Integrity: like-for-like alive proof across the window (the real
    # function, trivially-valid args, must still say True). Data: the
    # run-param probe is RECORDED, never asserted - _use_local_route is a
    # stateless function of its args (local_ok and slice_len <=
    # max(1, n_total // _F2_LOCAL_SLICE_FRAC)); at the observed slice_max
    # it can deterministically decline (bound 49159//8 = 6144 < 7105 on
    # this fixture) with no exclusion implied (h04 diag-service ruling
    # 12027b04(c)/51ad93d4 chain, mirrored here).
    alive_post = LAUNCHER_STATE["launcher"].invoke_real(True, 1, 1)
    REC["route_state"]["postwindow_alive_receipt"] = alive_post
    if alive_post["return"] is not True:
        fatal("post-window alive receipt returned "
              f"{alive_post['return']!r} — real-function integrity "
              "lost across the window; stop-and-preserve")
    live = LAUNCHER_STATE["launcher"].invoke_real(
        bool(f2s["local_route_ok"]), int(f2s["local_slice_max"]),
        REC["counts"]["V_nodes"])
    REC["route_state"]["postwindow_live_receipt"] = live
    REC["route_state"]["differential_reading"] = (
        "real(_use_local_route) returned "
        f"{live['return']} on the engine's OWN live constants "
        f"(local_route_ok={f2s['local_route_ok']}, "
        f"local_slice_max={f2s['local_slice_max']}, "
        f"n_total={REC['counts']['V_nodes']}) while the wrapper forced "
        f"False for every OD (fast_calls=0): launcher-forced, not "
        "data-driven")
else:
    if int(f2s["fast_calls"]) <= 0:
        fatal(f"cand CONTRADICTION: fast_calls={f2s['fast_calls']} <= 0 — "
              "production route did not engage, stop-and-preserve")
    REC["route_state"]["expected_regime"] = (
        "fallback=none expected at this cell (slices far below "
        "local_slice_max); nonzero overflow/fallback is NOT auto-reject — "
        "of-policy is output-identical by committed tests; G7 adjudication "
        "reads these counters")

# ---- warnings, artifacts, ambient (post-window) -----------------------------
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
print(f"[f2r-job] ok arm={args.arm} win={(t1 - t0) / 1e9:.3f}s "
      f"peak={REC['peak_rss_ru_maxrss_mib']:.1f}MiB "
      f"f2_stats={REC['route_state']['f2_stats']}", flush=True)
