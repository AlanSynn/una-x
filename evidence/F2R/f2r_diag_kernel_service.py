"""F2R diagnostics kernel service — instrumented probe, NEVER a gate run.

Vehicle per evidence/F2R/screen_spec.json (P3/P4b + session_discrepancy_design
+ lease_and_diagnostics_class.diagnostics_uncharged): fired ONLY by
campaign_data/f2r_lease_pass.py (diagnostics entries: charged_s 0.0,
gate_eligible false), on THROWAWAY NUMBA_CACHE_DIRs, never the gate roots,
never the gate run-id pattern (f2r_diag_*). Runs on the SAME shared wheel
venv as the screen (same-binary discipline).

Skeleton chain (P3 clone-rot): campaigns/una_large_e2e/evidence/F1R/
f1r_bare_job.py (BINDING skeleton) -> evidence/F2R/f2r_bare_job.py (this
bundle) -> THIS FILE. Diff stated vs f2r_bare_job.py:
  KEPT: the entire production-path scaffold — launcher sha-assert + install()
    BEFORE package import (b0), import-identity realpath fatal, module pin,
    H pin, frozen manifest sha + input rehash, pinned sel1024 fixture
    derivation + attribute crosscheck, settings assembly, K wiring,
    pre-window input resolution, [F2]-line + _f2_stats enforcement with the
    same FATAL contradictions, UNA(verbosity=2), artifacts walk, ru_maxrss.
  CHANGED: run-id must start f2r_diag_; NUMBA_CACHE_DIR is the EXPLICIT
    throwaway --cache-root (lease-pass created it; asserted equal to env);
    window semantics --mode dependent (below); [F2]/route enforcement runs
    after the MEASURED run's engine only.
  ADDED (the diagnostics legs, spec session_discrepancy_design):
    (i)   warm-root precondition census: warm mode asserts census AFTER
          warmup == census AFTER measured (measured window adds nothing);
    (ii)  FULL span retention on _accumulate_od_flow: every call kept
          (call_index, thread_id, phase, t0_ns, dt_ns), 2 ms concurrency
          monitor; post-run distribution min/median/p90/p99/p999/max + TOP-10
          spans with indices;
    (iii) reconciliation: measured span-sum vs measured window wall vs
          process_time_ns CPU delta;
    (iv)  cold companion: --mode cold on a WIPED throwaway root; the
          preregistered cascade prediction (constants below, spec
          cascade_prediction.arithmetic) is recorded alongside the measured
          top-10/sum/closure as DATA for h04 — confirmation/refutation is
          the reviewer's, not this instrument's;
    (v)   both route states: --arm b0 (launcher) | cand, receipts identical
          to the bare job;
    (vi)  module pin re-asserted in-run (same as bare job, restated here);
    (vii) per-dispatcher call counts, declared set: _accumulate_od_flow
          (span-wrapped; count from stats), _accumulate_od_flow_local,
          Accessibility.reach_gravity_knn_access, _ordered_csr.
          _try_build_ordered_csr (count-wrapped; wrapper overhead DECLARED,
          windows not gate-comparable);
    (viii) raw flow arrays + span table preserved to campaign_data (npz +
          record JSON).
  REMOVED: the bare job's single-run gate-window semantics (replaced by
    warmup+measured in warm mode); nothing else.

OVERHEAD DISCLOSURE: the span/count wrappers are Python-level and inflate
kernel dispatch cost; diag windows are NOT comparable to gate windows and
are never used as admission inputs.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import sys
import threading
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--cell", required=True, choices=["O3_FLOW"])
ap.add_argument("--variant-id", required=True)
ap.add_argument("--expected-site-packages", required=True)
ap.add_argument("--out-json", required=True)
ap.add_argument("--output-root", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--arm", required=True, choices=["b0", "cand"])
ap.add_argument("--mode", required=True, choices=["cold", "warm"])
ap.add_argument("--cache-root", required=True,
                help="throwaway NUMBA_CACHE_DIR (lease-pass created it)")
ap.add_argument("--launcher-path", default=None)
ap.add_argument("--launcher-sha256", default=None)
ap.add_argument("--module-pin-sha256",
                default="1f3cf4d2725320569c1b19bf339a39d0bf013d1cb7b6f8e5d72526a77531797c")
ap.add_argument("--flow-stripes", type=int, required=True)
ap.add_argument("--wheel-sha256", required=True)
ap.add_argument("--numba-threads", type=int, required=True)
args = ap.parse_args()

if not args.run_id.startswith("f2r_diag_"):
    print("[f2r-diag] FATAL: run id must start f2r_diag_ (outside the gate "
          "pattern)", flush=True)
    sys.exit(2)
if os.environ.get("NUMBA_CACHE_DIR") != args.cache_root:
    print("[f2r-diag] FATAL: --cache-root != NUMBA_CACHE_DIR env (the "
          "lease-pass owns the throwaway root)", flush=True)
    sys.exit(2)

REC = {
    "schema_version": 1, "task": "F2R", "record": "diagnostics_kernel_service",
    "role": "screen-executor", "cell": args.cell, "arm": args.arm,
    "mode": args.mode, "run_id": args.run_id,
    "gate_eligible": False,
    "not_a_gate_run": "instrumented diagnostics probe; windows are NOT "
                      "gate-comparable (wrapper overhead declared); charged_s "
                      "0.0 by the lease-pass",
    "frozen_triple": {"W": 1, "K": args.flow_stripes, "H": args.numba_threads},
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "env": {"NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
            "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR")},
    # preregistered cascade prediction (spec session_discrepancy_design.
    # cascade_prediction.arithmetic) — recorded as DATA for h04's
    # confirm/refute; this instrument computes, it does not adjudicate
    "cascade_prediction_preregistered": {
        "source": "evidence/F2R/screen_spec.json session_discrepancy_design/"
                  "cascade_prediction (committed P1 bytes 085684ec)",
        "predicted_top_block_spans": 9,
        "h05_session_A_span_sum_ns": 17578805432,
        "h05_session_B_span_sum_ns": 638797485,
        "h05_compile_span_ns": 1929925833,
        "predicted_sum_A_ns": 18008129982,
        "closure_ratio_h05": 1.0244228512375744,
        "prediction_text": "a cold-cache companion shows a BLOCK of ~9 top "
                           "spans each ~the compile duration (compile-lock "
                           "serialization), span-sum ~ 9 x compile_span + "
                           "steady-state sum",
    },
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
    print(f"[f2r-diag] FATAL: {msg}", flush=True)
    sys.exit(2)


def cache_census(root):
    if not os.path.isdir(root):
        return {"count": None, "names": [], "note": "root missing"}
    names = sorted(os.listdir(root))
    return {"count": len(names), "names": names}


def census_fingerprint(c):
    return {"count": c["count"], "names": c["names"]}


# ---- b0 launcher: sha-assert, load, install BEFORE the package import ------
LAUNCHER_STATE = None
if args.arm == "b0":
    if not args.launcher_path or not args.launcher_sha256:
        fatal("b0 arm requires --launcher-path and --launcher-sha256")
    if not os.path.isfile(args.launcher_path):
        fatal(f"launcher file missing: {args.launcher_path}")
    if sha256_file(args.launcher_path) != args.launcher_sha256:
        fatal("launcher sha != pinned")
    spec = importlib.util.spec_from_file_location("f2r_route_launcher",
                                                  args.launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    install_receipt = launcher.install()
    if install_receipt["aggregateflow_sha256_at_install"] != args.module_pin_sha256:
        fatal("launcher installed against a module whose sha != the frozen pin")
    if not install_receipt["no_other_attribute_touched"]:
        fatal("launcher receipt reports an attribute beyond _use_local_route "
              "was changed")
    alive = launcher.invoke_real(True, 1, 1)
    if alive["return"] is not True:
        fatal(f"launcher pre-window alive receipt returned {alive!r}")
    LAUNCHER_STATE = {"launcher": launcher, "install_receipt": install_receipt,
                      "prewindow_alive_receipt": alive}
    REC["launcher"] = {"path": os.path.abspath(args.launcher_path),
                       "sha256": args.launcher_sha256,
                       "install_receipt": install_receipt,
                       "prewindow_alive_receipt": alive}

# ---- import identity + module pin + H pin -----------------------------------
import numba  # noqa: E402
import urban_network_analysis as u  # noqa: E402

una_dir = os.path.dirname(os.path.abspath(u.__file__))
exp_real = os.path.realpath(args.expected_site_packages)
identity_ok = os.path.realpath(una_dir).startswith(exp_real + os.sep)
REC["import"] = {"urban_network_analysis_file": una_dir,
                 "expected_site_packages": args.expected_site_packages,
                 "identity_assertion": ("ok (realpath under expected site-packages)"
                                        if identity_ok else "FAILED"),
                 "wheel_sha256": args.wheel_sha256}
if not identity_ok:
    fatal("import identity assertion failed")
af_path = os.path.join(una_dir, "Engines", "AggregateFlow.py")
af_sha = sha256_file(af_path)
REC["module_pin"] = {"path": af_path, "sha256": af_sha,
                     "expected": args.module_pin_sha256,
                     "match": af_sha == args.module_pin_sha256}
if af_sha != args.module_pin_sha256:
    fatal("installed AggregateFlow sha != module pin")
h_config = int(numba.config.NUMBA_NUM_THREADS)
h_live = int(numba.get_num_threads())
REC["process"] = {"python": sys.version.split()[0], "numba": numba.__version__,
                  "numpy": np.__version__,
                  "numba_num_threads_config": h_config,
                  "numba_get_num_threads": h_live}
if h_config != args.numba_threads or h_live != args.numba_threads:
    fatal(f"numba H pin mismatch: config={h_config} live={h_live}")

# ---- cache census (pre-anything) --------------------------------------------
REC["numba_cache_census"] = {"root": args.cache_root,
                             "pre_run": cache_census(args.cache_root),
                             "rule": "throwaway root; cold = wiped to 0 files "
                                     "pre-spawn by the lease-pass; warm = "
                                     "populated by THIS service's warmup"}

# ---- manifest + pinned sha + input rehash (outside any window) --------------
FROZEN_MANIFEST_SHA = "c2e6f2669cf334e4060d3930d23d44baf41efa5c54f36908cd2b72b25ec96e2a"
with open(args.manifest) as f:
    manifest = json.load(f)
if sha256_file(args.manifest) != FROZEN_MANIFEST_SHA:
    fatal("manifest sha != frozen O3_FLOW workload sha")
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

# ---- origin selection: pinned sel1024 -> fixture (outside any window) -------
sel_spec = manifest.get("origin_selection")
sel = next((v for v in sel_spec.get("variants", [])
            if v["variant_id"] == args.variant_id), None)
if sel is None:
    fatal(f"--variant-id must be one of "
          f"{[v['variant_id'] for v in sel_spec.get('variants', [])]}")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(args.manifest),
                                         "..", "..", ".."))
sel_path = os.path.join(REPO_ROOT, sel["selection_manifest"])
if sha256_file(sel_path) != sel.get("selection_manifest_sha256"):
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
sel_rec = {"selection_manifest": sel_path, "selected_count": sel["selected_count"],
           "chosen_variant": args.variant_id}
data_folder = os.path.dirname(manifest["inputs"]["network"]["local_path"])
import geopandas as gpd  # noqa: E402
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
                (wv != wv and gv != gv) or wv == gv)
            if not same:
                fatal(f"fixture row {i} attr {col}: {gv!r} != selection "
                      f"manifest {wv!r}")
    sel_rec["attribute_crosscheck"] = "fid/id/lon/lat row-by-row: ok"
fixture = os.path.join(data_folder, f"f2r_origins_{args.run_id}_{len(fx)}.geojson")
os.makedirs(data_folder, exist_ok=True)
fx.to_file(fixture, driver="GeoJSON")
sel_rec["fixture"] = {"path": fixture, "sha256": sha256_file(fixture),
                      "rows": int(len(fx))}
REC["origin_selection"] = sel_rec

# ---- UNA + settings builder (factored: warm mode builds TWICE) --------------
from urban_network_analysis import UNA  # noqa: E402


def build_una(tag):
    una = UNA(verbosity=2)
    st = una.settings
    PATH_KEYS = {"data_folder", "network_file", "origins_file",
                 "destinations_file", "output_folder"}
    for k, v in manifest["settings_before"].items():
        if k in PATH_KEYS or k.startswith("_"):
            continue
        if hasattr(st, k):
            setattr(st, k, v)
    for k, v in manifest.get("settings_overrides", {}).items():
        setattr(st, k, v)
    st.data_folder = data_folder
    st.network_file = os.path.basename(
        manifest["inputs"]["network"]["local_path"])
    st.origins_file = os.path.basename(fixture)
    st.destinations_file = os.path.basename(
        manifest["inputs"]["destinations"]["local_path"])
    out_root = os.path.join(args.output_root, tag)
    os.makedirs(out_root, exist_ok=True)
    st.output_folder = out_root
    una.topology.num_threads = int(args.flow_stripes)
    if int(una.topology.num_threads) != int(args.flow_stripes):
        fatal(f"[{tag}] topology.num_threads wiring did not stick")
    return una


# ---- INSTRUMENTATION (legs ii, vii) — installed ONCE, guarded ---------------
import urban_network_analysis.Engines.AggregateFlow as AF  # noqa: E402
import urban_network_analysis.Engines.Accessibility as AC  # noqa: E402
import urban_network_analysis.Engines._ordered_csr as OCR  # noqa: E402

PHASE = {"label": "setup"}
_lock = threading.Lock()
spans = []            # FULL retention: {call_index, thread_id, phase, t0_ns, dt_ns}
counts = {"_accumulate_od_flow": 0,
          "_accumulate_od_flow_local": 0,
          "Accessibility.reach_gravity_knn_access": 0,
          "_ordered_csr._try_build_ordered_csr": 0}
concur = {"cur": 0, "max": 0}
_stop = threading.Event()


def _monitor():
    prev = -1
    while not _stop.wait(0.002):
        with _lock:
            if concur["cur"] != prev:
                concur["max"] = max(concur["max"], concur["cur"])
                prev = concur["cur"]


_real_od = AF._accumulate_od_flow
if getattr(_real_od, "_f2r_diag_is_wrapper", False):
    fatal("double instrumentation attempt (_accumulate_od_flow already "
          "wrapped)")


def _od_span_wrapper(*a, **kw):
    with _lock:
        counts["_accumulate_od_flow"] += 1
        idx = counts["_accumulate_od_flow"]
        concur["cur"] += 1
        concur["max"] = max(concur["max"], concur["cur"])
        phase = PHASE["label"]
        tid = threading.get_ident()
        t0 = time.perf_counter_ns()
    try:
        return _real_od(*a, **kw)
    finally:
        dt = time.perf_counter_ns() - t0
        with _lock:
            concur["cur"] -= 1
            spans.append({"call_index": idx, "thread_id": tid,
                          "phase": phase, "t0_ns": t0, "dt_ns": dt})


_od_span_wrapper._f2r_diag_is_wrapper = True
setattr(AF, "_accumulate_od_flow", _od_span_wrapper)


def _count_wrap(module, attr, key):
    real = getattr(module, attr)
    if getattr(real, "_f2r_diag_is_wrapper", False):
        fatal(f"double instrumentation attempt: {key}")

    def wrapper(*a, **kw):
        with _lock:
            counts[key] += 1
        return real(*a, **kw)

    wrapper._f2r_diag_is_wrapper = True
    setattr(module, attr, wrapper)


_count_wrap(AF, "_accumulate_od_flow_local", "_accumulate_od_flow_local")
_count_wrap(AC, "reach_gravity_knn_access",
            "Accessibility.reach_gravity_knn_access")
_count_wrap(OCR, "_try_build_ordered_csr",
            "_ordered_csr._try_build_ordered_csr")
REC["instrumentation"] = {
    "span_wrapped": ["_accumulate_od_flow (FULL retention)"],
    "count_wrapped": ["_accumulate_od_flow_local",
                      "Accessibility.reach_gravity_knn_access",
                      "_ordered_csr._try_build_ordered_csr"],
    "monitor_s": 0.002,
    "overhead_disclosure": "python-level wrappers inflate dispatch cost; diag "
                           "windows are NOT gate-comparable",
    "declared_census_set": sorted(counts.keys()),
}
_mon = threading.Thread(target=_monitor, daemon=True)
_mon.start()

# ---- warmup (warm mode only) + census leg (i) -------------------------------
if args.mode == "warm":
    una_w = build_una("warmup")
    PHASE["label"] = "warmup"
    t = time.perf_counter_ns()
    una_w.RunFlow()
    REC["warmup"] = {"wall_ns": time.perf_counter_ns() - t,
                     "note": "untimed gate-wise; populates the throwaway "
                             "cache root; engine state DISCARDED (fresh una "
                             "built for the measured run)"}
    REC["numba_cache_census"]["post_warmup"] = cache_census(args.cache_root)
    del una_w

# ---- MEASURED window --------------------------------------------------------
una = build_una("measured")
REC["pre_window_input_resolution"] = {
    "network": os.path.isfile(os.path.join(
        una.settings.data_folder, una.settings.network_file)),
    "origins": os.path.isfile(os.path.join(
        una.settings.data_folder, una.settings.origins_file)),
    "destinations": os.path.isfile(os.path.join(
        una.settings.data_folder, una.settings.destinations_file))}
if not all(REC["pre_window_input_resolution"].values()):
    fatal("pre-window input resolution failed")
PHASE["label"] = "measured"
wall_ns_start = time.perf_counter_ns()
cpu_ns_start = time.process_time_ns()
una.RunFlow()
cpu_ns_delta = time.process_time_ns() - cpu_ns_start
wall_ns = time.perf_counter_ns() - wall_ns_start
PHASE["label"] = "post"
REC["window"] = {"measured_wall_ns": wall_ns, "measured_cpu_ns": cpu_ns_delta,
                 "boundaries": "perf_counter_ns + process_time_ns around the "
                               "single measured una.RunFlow() (instrumented; "
                               "NOT gate-comparable)"}

# ---- leg (i): warm-mode census equality -------------------------------------
REC["numba_cache_census"]["post_measured"] = cache_census(args.cache_root)
if args.mode == "warm":
    pre = census_fingerprint(REC["numba_cache_census"]["post_warmup"])
    post = census_fingerprint(REC["numba_cache_census"]["post_measured"])
    REC["leg_i_warm_census_equal"] = {"equal": pre == post,
                                      "post_warmup": pre, "post_measured": post}
    if pre != post:
        fatal("leg (i) FAILED: measured window added cache entries on a "
              "pre-warmed root — warm precondition violated, stop-and-preserve")

# ---- legs (ii)+(iii): spans, distribution, top-10, reconciliation -----------
with _lock:
    measured_spans = [s for s in spans if s["phase"] == "measured"]
    all_spans_n = len(spans)
    dts = sorted(s["dt_ns"] for s in measured_spans)
    span_sum_ns = sum(dts)


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


top10 = sorted(measured_spans, key=lambda s: -s["dt_ns"])[:10]
REC["leg_ii_spans"] = {
    "n_spans_all_phases": all_spans_n,
    "n_spans_measured": len(measured_spans),
    "max_concurrent_observed": concur["max"],
    "distribution_measured_ns": {
        "min": dts[0] if dts else None,
        "median": pct(dts, 0.50), "p90": pct(dts, 0.90),
        "p99": pct(dts, 0.99), "p999": pct(dts, 0.999),
        "max": dts[-1] if dts else None},
    "top10_measured": top10,
    "full_span_table_file": f"{args.out_json}.spans.json",
    "npz_file": f"{args.out_json}.spans.npz",
}
os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
with open(f"{args.out_json}.spans.json", "w") as f:
    json.dump(spans, f)
np.savez(f"{args.out_json}.spans.npz",
         t0_ns=np.array([s["t0_ns"] for s in spans], dtype=np.int64),
         dt_ns=np.array([s["dt_ns"] for s in spans], dtype=np.int64),
         call_index=np.array([s["call_index"] for s in spans], dtype=np.int64),
         thread_id=np.array([s["thread_id"] for s in spans], dtype=np.int64),
         phase=np.array([s["phase"] for s in spans]))
REC["leg_iii_reconciliation"] = {
    "measured_span_sum_ns": span_sum_ns,
    "measured_wall_ns": wall_ns,
    "measured_process_cpu_ns": cpu_ns_delta,
    "span_sum_over_wall": (round(span_sum_ns / wall_ns, 6) if wall_ns else None),
    "span_sum_over_cpu": (round(span_sum_ns / cpu_ns_delta, 6)
                          if cpu_ns_delta else None),
    "note": "spans are CONCURRENCY-INCLUSIVE latency (h05 ruling), not "
            "per-call service; closure figures are data for h04",
}

# ---- leg (vii): dispatcher counts -------------------------------------------
REC["leg_vii_dispatcher_counts"] = {"counts": counts,
                                    "declared_set": sorted(counts.keys())}

# ---- leg (v): route receipts ([F2] + _f2_stats, measured engine) ------------
ent = list(una.topology.logger.log_list)
REC["log_lines"] = {"count": len(ent), "entries": ent}
f2_lines = [e for e in ent if "[F2]" in str(e.get("details", ""))]
REC["f2_log_line"] = {"present": bool(f2_lines), "entries": f2_lines}
if not f2_lines:
    fatal("[F2] log line ABSENT in the diagnostics run — log-persistence "
          "instrument defect, immediate stop")
fe = una.flow
f2s = getattr(fe, "_f2_stats", None)
REC["route_state"] = {"f2_stats": f2s, "arm": args.arm}
if f2s is None:
    fatal("engine _f2_stats missing in diagnostics run")
if args.arm == "b0":
    if int(f2s["fast_calls"]) != 0:
        fatal(f"b0 CONTRADICTION: fast_calls={f2s['fast_calls']} != 0 in "
              "diagnostics — launcher not engaged, stop-and-preserve")
    live = LAUNCHER_STATE["launcher"].invoke_real(
        bool(f2s["local_route_ok"]), int(f2s["local_slice_max"]),
        int(una.topology.network.node_points.shape[0]))
    REC["route_state"]["postwindow_live_receipt"] = live
    if live["return"] is not True:
        fatal("diagnostics live receipt returned False — data-driven "
              "exclusion signature; stop-and-preserve")
else:
    if int(f2s["fast_calls"]) <= 0:
        fatal(f"cand CONTRADICTION: fast_calls={f2s['fast_calls']} <= 0 in "
              "diagnostics — production route did not engage")

# ---- leg (viii): raw flow arrays -> campaign_data ---------------------------
def arr_meta(a):
    if a is None:
        return None
    a = np.asarray(a)
    return {"dtype": str(a.dtype), "shape": list(a.shape),
            "sha256_bytes": hashlib.sha256(a.tobytes()).hexdigest()
                            if a.size else None}


arrays = {"edge_flow": getattr(fe, "edge_flow", None),
          "edge_flow_AB": getattr(fe, "edge_flow_AB", None),
          "edge_flow_BA": getattr(fe, "edge_flow_BA", None),
          "node_flow": getattr(fe, "node_flow", None)}
np.savez(f"{args.out_json}.arrays.npz",
         **{k: np.asarray(v) for k, v in arrays.items() if v is not None})
REC["leg_viii_raw_arrays"] = {
    "npz_file": f"{args.out_json}.arrays.npz",
    "arrays": {k: arr_meta(v) for k, v in arrays.items()},
    "counts": {"V_nodes": int(una.topology.network.node_points.shape[0]),
               "E_edges": int(una.topology.network.lengths.shape[0]),
               "O_rows": int(len(una.topology.origins.geometry)),
               "D_rows": int(len(una.topology.destinations.geometry))},
}

# ---- ambient -----------------------------------------------------------------
artifacts = {}
for dirpath, _, files in os.walk(args.output_root):
    for fn in files:
        p = os.path.join(dirpath, fn)
        artifacts[os.path.relpath(p, args.output_root)] = {
            "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
REC["artifacts"] = artifacts
REC["peak_rss_ru_maxrss_mib"] = resource.getrusage(
    resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
REC["loadavg_after_measured"] = os.getloadavg()
_stop.set()
_mon.join(timeout=2)
write_out()
print(f"[f2r-diag] ok arm={args.arm} mode={args.mode} "
      f"wall={wall_ns / 1e9:.3f}s spans={len(measured_spans)} "
      f"counts={counts}", flush=True)
