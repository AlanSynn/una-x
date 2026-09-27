"""F3R paired-screen bare job driver — executed BY AN ARM VENV's python.

Vehicle per screen_spec (P4b): authored on the evidence/F1R/f1r_bare_job.py
skeleton (which itself descends from the A1R skeleton), h04-reviewer-verified
before any gate-eligible window, coordinator-committed. One process = one
run; the supervisor (f3r_run_pairs.py, authored next) owns scheduling, cache
freshness, the (W,K,H) env, lease windows, the watchdog, and timeouts.

F3R adaptations against the F1R skeleton (spec obligations + rulings):
  - UNA(verbosity=2): the ruled driver obligation — the v=2 [F3] gradient-
    workspace line prints AND lands in log_list (Logger.py appends every
    entry regardless of verbosity; verbosity gates printing only).
  - [F3] tail capture + parse: the candidate arm must emit EXACTLY ONE
    "[F3] cap=...B margin=...B chunks=... schedule=[...] fallback=...
    floor_fired=..." tail per gradient-precompute execution (AggregateFlow
    calls _precompute_dest_gradients once per RunFlow at this workload);
    zero [F3] lines on cand = instrument defect = stop. The b0 arm must
    emit ZERO [F3] lines (identity check: F3 module absent).
  - Mechanism receipt (spec G3_mechanism_receipt, ruled leg structure at
    spec rev 6: form resim_perinstance_v1), recomputed IN-RECORD from
    MEASURED values and PUBLISHED arrays — never hardcoded:
      V' = _csr_indptr.shape[0] - 1 (the CSR row width, n_total);
      n_dest = _n_destinations (gradient rows = destinations — NOT the
      origin selection count);
      fixed_live = both CSR directions (data+indices+indptr) + dest_nodes
      (arange int64) + counts (zeros int64), replicating the caller's
      allocation expressions;
      per_source = V'*(dist8+pred4+mask1) + row_temporaries(8*V') +
      source_index_cost(dest dtype itemsize).
    Gradient publication stash: the engine builds dest_grad_sparse inside
    RunFlow and drops it after the origin loop (consumed at
    AggregateFlow.py:828, never stored on self), so a class-level
    pass-through wrapper around AggregateFlow._precompute_dest_gradients —
    installed PRE-window, on BOTH arms identically — stashes the returned
    (indptr, nodes, dist, pred) REFERENCE for post-window receipt use.
    Zero per-slice capture (the prohibited class); in-window footprint is
    one call indirection + one reference store per gradient-precompute
    execution; every receipt computation happens POST-window; the recorded
    ru_maxrss is captured BEFORE the receipt block reads the stash, so the
    stash cannot inflate it. DISCLOSED in every record for h04's P4b
    ruling.
    Leg (c) re-simulation: counts = diff(published indptr); the retained
    trajectory is indptr[s_k] x per_entry at each observed slice start
    (parts are appended PER-ROW IN SLICE ORDER, AggregateFlow.py
    :1032-1044 at b387f90); the driver replays _lfws.select_chunk from the
    installed venv over n_dest and asserts ELEMENT-WISE equality with the
    observed schedule. Leg (d) is SUBSUMED by that equality (uniform
    non-final widths modeled retained~0 fixtures — spec rev 6).
    Leg (e) exact per-instance form: for EVERY slice k,
    indptr[s_k] x per_entry + schedule[k] x per_source + margin <= cap,
    with per_entry = nodes.itemsize + dist.itemsize + pred.itemsize read
    from the published arrays and the identity
    nodes.nbytes + dist.nbytes + pred.nbytes == indptr[-1] x per_entry
    asserted (per_entry = 20 B by construction and verified anyway).
    Q2: retained_total in the record = the EXACT published bytes
    (nodes.nbytes + dist.nbytes + pred.nbytes); the finite-count-x-20
    multiply is the CHECKED IDENTITY, not the accounting source.
    The aggregate-form retained check (total retained + widest slice) is
    REJECTED (spec rev 6): its two maxima are anti-correlated, so it
    false-fires behaving mechanisms.
  - In-run module identity: cand imports the private
    Engines._large_flow_workspace from the INSTALLED venv and asserts the
    frozen F5 constants (DEFAULT_CAP_BYTES 268435456, MARGIN_BYTES
    67108864); b0 asserts the module is ABSENT (find_spec None + import
    raises).
  - fallback != none or floor_fired != none, or any binding-leg mismatch =
    G7 stop-and-preserve: the run COMPLETES (full record written), then
    exits nonzero so the supervisor stops the matrix immediately (PD-E).
"""
import argparse
import hashlib
import json
import os
import re
import resource
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--cell", required=True, choices=["O3_FLOW"],
                help="screen cell (spec driver parameter)")
ap.add_argument("--variant-id", required=True,
                help="origin_selection variant (F3R screen: sel1024)")
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
    "schema_version": 1, "task": "F3R", "record": "bare_complete_job_flow_f3r",
    "role": "screen-executor", "cell": args.cell, "arm": args.arm,
    "run_id": args.run_id,
    "block": args.block, "order": args.order,
    "gate_eligible": args.block != 0,
    "diagnostics_class": ("memory_canary" if args.block == 0 else None),
    "cold_warm": ("diagnostic (memory canary, throwaway cache)" if args.block == 0
                  else "cold (fresh per-arm cache)" if args.block == 1
                  else "warm (reused cache)"),
    "timing_class": "bare_complete_job_no_leaf_profiler",
    "capacity_only": "DECISION.md line 11: wall time recorded as capacity_only; "
                     "no throughput or performance claim made or admissible "
                     "from this record",
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
    print(f"[f3r-job] FATAL: {msg}", flush=True)
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

# ---- F3 arm identity: _large_flow_workspace present/absent (in-run) ---------
# Ruled constants (F5 freeze): DEFAULT_CAP_BYTES = CAP_256 = 268435456,
# MARGIN_BYTES = 67108864. Any value change = new ruling + delta cycle.
RULED_CAP_BYTES = 268435456
RULED_MARGIN_BYTES = 67108864
lfws_modname = "urban_network_analysis.Engines._large_flow_workspace"
if args.arm == "cand":
    import importlib
    lfws = importlib.import_module(lfws_modname)
    lfws_file = os.path.realpath(lfws.__file__)
    lfws_ok = lfws_file.startswith(exp_real + os.sep)
    REC["f3_module_identity"] = {
        "module": lfws_modname, "file": lfws_file,
        "identity_assertion": ("ok (realpath under expected site-packages)"
                               if lfws_ok else f"FAILED: {lfws_file}"),
        "default_cap_bytes": int(lfws.DEFAULT_CAP_BYTES),
        "margin_bytes": int(lfws.MARGIN_BYTES),
        "cap_matches_ruled": int(lfws.DEFAULT_CAP_BYTES) == RULED_CAP_BYTES,
        "margin_matches_ruled": int(lfws.MARGIN_BYTES) == RULED_MARGIN_BYTES,
    }
    if not lfws_ok:
        fatal("f3 module import identity assertion failed")
    if not (REC["f3_module_identity"]["cap_matches_ruled"]
            and REC["f3_module_identity"]["margin_matches_ruled"]):
        fatal("f3 module frozen constants deviate from ruled F5 values")
else:
    import importlib.util
    spec_found = importlib.util.find_spec(lfws_modname)
    raised = None
    try:
        __import__(lfws_modname)
    except Exception as exc:  # ModuleNotFoundError expected on b0
        raised = type(exc).__name__
    REC["f3_module_identity"] = {
        "module": lfws_modname,
        "find_spec": "None (absent)" if spec_found is None else repr(spec_found),
        "import_attempt": ("raised " + str(raised)) if raised
                          else "IMPORTED (UNEXPECTED on b0)",
        "absence_assertion": ("ok (F3 module absent)" if spec_found is None
                              and raised is not None else "FAILED"),
    }
    if REC["f3_module_identity"]["absence_assertion"] != "ok (F3 module absent)":
        fatal("b0 arm identity failure: _large_flow_workspace is importable")

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
           "workload_variant": selman.get("workload_variant"),
           "note": "sel1024 selects ORIGINS; the gradient row count is the "
                   "DESTINATION count (manifest inputs.destinations."
                   "feature_count) — recorded post-window as n_dest_measured"}
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
    fixture = os.path.join(fixture_dir, f"f3r_origins_{args.run_id}_{len(fx)}.geojson")
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

una = UNA(verbosity=2)
REC["constructor"] = {"call": "UNA(verbosity=2)",
                      "note": "ruled F3R driver obligation (v=2 [F3] capture); "
                              "outside window; counted once; Logger appends "
                              "every entry to log_list regardless of verbosity, "
                              "so capture does not depend on this — verbosity "
                              "gates console printing only"}
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
pre_res = {}
for label, fn in (("network", st.network_file), ("origins", st.origins_file),
                  ("destinations", st.destinations_file)):
    resolved = os.path.join(st.data_folder, fn)
    pre_res[label] = resolved
    if not os.path.isfile(resolved):
        fatal(f"pre-window input resolution failed ({label}): {resolved}")
REC["pre_window_input_resolution"] = pre_res

# ---- gradient publication stash (PRE-window, BOTH arms, disclosed) ----------
# dest_grad_sparse is a RunFlow local: consumed by _process_origins_aggregate
# and never stored on self, so it is unreachable post-window without a stash.
# The wrapper is installed BEFORE the window at CLASS level (the engine object
# is constructed inside RunFlow), on both arms identically for G4 symmetry.
# Pass-through only: one call indirection + one reference store per
# gradient-precompute execution; no per-slice capture, no array copies, no
# numerical effect (the engine's own return value is returned untouched).
# All receipt computation is POST-window, and ru_maxrss is recorded BEFORE
# the receipt block reads the stash.
from urban_network_analysis.Engines.AggregateFlow import AggregateFlow \
    as _AFClass  # noqa: E402

_grad_stash = {"n_calls": 0, "grad": None}
_orig_pdg = _AFClass._precompute_dest_gradients


def _pdg_with_stash(self, ns):
    result = _orig_pdg(self, ns)
    _grad_stash["n_calls"] += 1
    _grad_stash["grad"] = result
    return result


_AFClass._precompute_dest_gradients = _pdg_with_stash
REC["gradient_publication_stash"] = {
    "installed": True,
    "when": "pre-window, class-level",
    "target": "AggregateFlow._precompute_dest_gradients (identical on both "
              "arms; method exists at both pins, returns (indptr, nodes, "
              "dist, pred) at c37cf22:961 and b387f90:1068)",
    "mechanism": "return-REFERENCE stash; pass-through wrapper",
    "disclosure": "in-window footprint = one call indirection + one "
                  "reference store per gradient-precompute execution; zero "
                  "per-slice capture; every receipt computation is "
                  "post-window; flagged for h04 P4b ruling (spec rev 6 legs "
                  "read nbytes/itemsize off the returned arrays, which die "
                  "with the RunFlow frame without this stash); RULED at P4b: "
                  "admitted with five conditions (h04 ruling 2026-09-27); "
                  "the gradient_publication_stash is not consulted by any "
                  "in-window code path; all receipt computation is "
                  "post-window. within-window unchanged: RunFlow's local "
                  "already holds "
                  "the reference through the window, the stash extends only "
                  "post-window lifetime, and ru_maxrss is a high-water mark "
                  "read before the stash is consumed (this sentence resolves "
                  "condition (iv))",
}

# ---- THE complete-job window: exactly ONE public call, nothing else ---------
t0 = time.perf_counter_ns()
una.RunFlow()
t1 = time.perf_counter_ns()
REC["window"] = {"application_window_ns": t1 - t0,
                 "boundaries": "perf_counter_ns around the single "
                               "una.RunFlow() call incl. synchronous exports and "
                               "in-call auto gravity-cap resolution; no "
                               "instrumentation inside except the disclosed "
                               "gradient_publication_stash pass-through "
                               "wrapper (see its record field)"}

# ---- [F3] tail capture (AFTER the window) -----------------------------------
# The engine's logger IS the topology logger (Base.__init__:124
# self.logger = self.topology.logger); Logger.log appends EVERY entry to
# log_list regardless of verbosity, so scanning entry["details"] is the
# capture path. cand: EXACTLY ONE [F3] line per run (one gradient-precompute
# execution per RunFlow at this workload) — zero = instrument defect = stop.
# b0: ZERO [F3] lines — any hit = arm identity failure = stop.
ent = list(una.topology.logger.log_list)
f3_entries = [e for e in ent if "[F3] cap=" in str(e.get("details", ""))]
F3_RE = re.compile(
    r"\[F3\] cap=(\d+)B margin=(\d+)B chunks=(\d+) "
    r"schedule=\[([^\]]*)\] fallback=(\w+) floor_fired=(\S+)")
FE_RE = re.compile(r"Gradient storage: ([\d,]+) finite entries")


def parse_f3(entry):
    details = str(entry.get("details", ""))
    m = F3_RE.search(details)
    fe = FE_RE.search(details)
    if not m or not fe:
        return None
    sched = [int(x) for x in m.group(4).split(",")] if m.group(4).strip() else []
    return {"verbatim_details": details,
            "finite_entries": int(fe.group(1).replace(",", "")),
            "cap_bytes": int(m.group(1)), "margin_bytes": int(m.group(2)),
            "chunks": int(m.group(3)), "schedule": sched,
            "fallback": m.group(5), "floor_fired": m.group(6)}


REC["f3_tail_capture"] = {
    "n_f3_lines": len(f3_entries),
    "source": "una.topology.logger.log_list (engine logger == topology logger)",
    "expected": ("exactly 1 (cand arm)" if args.arm == "cand"
                 else "exactly 0 (b0 arm identity check)"),
}
if args.arm == "cand" and len(f3_entries) != 1:
    REC["f3_tail_capture"]["verdict"] = "INSTRUMENT DEFECT"
    fatal(f"cand arm emitted {len(f3_entries)} [F3] lines, expected exactly 1 "
          "(instrument defect — ruled stop condition)")
if args.arm == "b0" and len(f3_entries) != 0:
    REC["f3_tail_capture"]["verdict"] = "ARM IDENTITY FAILURE"
    REC["f3_tail_capture"]["lines"] = [str(e.get("details", "")) for e in f3_entries]
    fatal(f"b0 arm emitted {len(f3_entries)} [F3] lines, expected 0 "
          "(F3 module must be absent on b0)")
if args.arm == "cand":
    f3 = parse_f3(f3_entries[0])
    if f3 is None:
        REC["f3_tail_capture"]["verdict"] = "INSTRUMENT DEFECT"
        REC["f3_tail_capture"]["verbatim_details"] = str(
            f3_entries[0].get("details", ""))
        fatal("[F3] line present but unparseable against the pinned emission "
              "format (instrument defect)")
    REC["f3_tail_capture"]["verdict"] = "ok"
    REC["f3_tail_capture"]["parsed"] = f3

# ---- post-window engine observations + mechanism receipt --------------------
fe_obj = getattr(una, "flow", None)
if fe_obj is None:
    fatal("una.flow is None after RunFlow — engine not constructed")
csr_indptr = getattr(fe_obj, "_csr_indptr", None)
if csr_indptr is None:
    fatal("engine _csr_indptr missing after RunFlow")
n_total_measured = int(csr_indptr.shape[0]) - 1          # = V' (row width)
n_net_measured = int(fe_obj._n_network_nodes)
n_dest_measured = int(fe_obj._n_destinations)            # gradient rows
# D4 in-run anchor assertions (h04 amendment round, both fatal): a printed
# profile value is grounded only when it cites its defining artifact, so
# the anchors are CHECKED at the vehicle, not just printed.
if n_dest_measured != int(manifest["inputs"]["destinations"]["feature_count"]):
    fatal("anchor assertion: n_dest_measured != manifest inputs."
          "destinations.feature_count")
if n_net_measured + n_dest_measured + sel["selected_count"] != n_total_measured:
    fatal(f"anchor assertion: decomposition identity failed: "
          f"{n_net_measured} + {n_dest_measured} + {sel['selected_count']} "
          f"!= {n_total_measured} (n_network + n_dest + n_origins_sel = V')")
fwd, rev = fe_obj._csr_fwd, fe_obj._csr_rev
# fixed_live recomputed by REPLICATING the caller's allocation expressions
# (AggregateFlow._precompute_dest_gradients): dest_nodes arange int64 +
# counts zeros int64, plus both CSR directions' three arrays.
dest_nodes_like = np.arange(n_net_measured,
                            n_net_measured + n_dest_measured, dtype=np.int64)
counts_like = np.zeros(n_dest_measured, dtype=np.int64)
fixed_live_measured = int(
    fwd.data.nbytes + fwd.indices.nbytes + fwd.indptr.nbytes
    + rev.data.nbytes + rev.indices.nbytes + rev.indptr.nbytes
    + dest_nodes_like.nbytes + counts_like.nbytes)
# Sizing contract dtypes (caller expressions; NOT literals):
dist_itemsize = np.dtype(np.float64).itemsize    # 8
pred_itemsize = np.dtype(np.int32).itemsize      # 4
mask_itemsize = np.dtype(np.bool_).itemsize      # 1
row_temporaries = 8 * n_total_measured
source_index_cost = int(dest_nodes_like.dtype.itemsize)
per_source_recomputed = int(
    n_total_measured * (dist_itemsize + pred_itemsize + mask_itemsize)
    + row_temporaries + source_index_cost)
REC["engine_shape_measured"] = {
    "v_prime_row_width_n_total": n_total_measured,
    "n_network_nodes": n_net_measured,
    "n_dest_measured": n_dest_measured,
    "note": "V' is the CSR row width (n_network_nodes + n_destinations + "
            "n_origins); gradient rows are DESTINATIONS, not the origin "
            "selection count",
    "csr_nnz_fwd": int(fwd.nnz), "csr_nnz_rev": int(rev.nnz),
    "csr_data_dtype": str(fwd.data.dtype),
    "csr_indices_dtype": str(fwd.indices.dtype),
    "csr_indptr_dtype": str(fwd.indptr.dtype),
}
REC["mechanism_receipt_inputs_measured"] = {
    "fixed_live_bytes": fixed_live_measured,
    "row_temporaries_bytes": row_temporaries,
    "source_index_cost_bytes": source_index_cost,
    "per_source_bytes": per_source_recomputed,
    "dtypes": {"dist": f"float64/{dist_itemsize}", "pred": f"int32/{pred_itemsize}",
               "mask": f"bool/{mask_itemsize}"},
}

# Receipt evaluation is deferred until AFTER the full observation block below,
# so a G7 stop-and-preserve record is complete for adjudication (PD-E: the run
# completes, then the matrix stops). See "mechanism receipt evaluation" below.

# ---- observation (all AFTER the window) ------------------------------------
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


REC["flow_arrays"] = {
    "edge_flow": arr_rec(getattr(fe_obj, "edge_flow", None)),
    "edge_flow_AB": arr_rec(getattr(fe_obj, "edge_flow_AB", None)),
    "edge_flow_BA": arr_rec(getattr(fe_obj, "edge_flow_BA", None)),
    "node_flow": arr_rec(getattr(fe_obj, "node_flow", None)),
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

# ---- mechanism receipt evaluation (cand; AFTER full observation) ------------
if args.arm == "cand":
    p = REC["f3_tail_capture"]["parsed"]
    cap_obs, margin_obs = p["cap_bytes"], p["margin_bytes"]
    sched, chunks_obs = p["schedule"], p["chunks"]
    legs = {}
    # (a) cap/margin exactly the ruled frozen values, observed == in-run module
    legs["a_cap_margin_exact"] = {
        "cap_observed": cap_obs, "margin_observed": margin_obs,
        "cap_expected": RULED_CAP_BYTES, "margin_expected": RULED_MARGIN_BYTES,
        "pass": (cap_obs == RULED_CAP_BYTES == int(lfws.DEFAULT_CAP_BYTES)
                 and margin_obs == RULED_MARGIN_BYTES == int(lfws.MARGIN_BYTES))}
    # (b) fallback=none and floor_fired=none (G7 stop-and-preserve otherwise)
    legs["b_fallback_none"] = {
        "fallback": p["fallback"], "floor_fired": p["floor_fired"],
        "pass": p["fallback"] == "none" and p["floor_fired"] == "none"}
    legs_pass_b = legs["b_fallback_none"]["pass"]
    # published gradient arrays from the disclosed stash (exactly 1 execution)
    if _grad_stash["n_calls"] != 1 or _grad_stash["grad"] is None:
        fatal("gradient publication stash expected exactly 1 gradient-precompute "
              f"execution with a returned tuple; got n_calls="
              f"{_grad_stash['n_calls']} (instrument defect)")
    g_indptr, g_nodes, g_dist, g_pred = _grad_stash["grad"]
    g_indptr = np.asarray(g_indptr)
    per_entry = int(g_nodes.itemsize + g_dist.itemsize + g_pred.itemsize)
    # byte identity (asserted): published part nbytes == indptr[-1] x per_entry
    byte_identity = {
        "lhs_published_bytes": int(g_nodes.nbytes + g_dist.nbytes + g_pred.nbytes),
        "rhs_indptr_last_x_per_entry": int(g_indptr[-1]) * per_entry,
        "per_entry_bytes": per_entry,
        "per_entry_note": "read from the PUBLISHED arrays; pinned by the "
                          "caller's astype calls (cols int64, dist float64, "
                          "preds int32) regardless of scipy's returns",
        "pass": int(g_nodes.nbytes + g_dist.nbytes + g_pred.nbytes)
                == int(g_indptr[-1]) * per_entry,
    }
    if not byte_identity["pass"]:
        fatal("published-gradient byte identity failed: "
              f"{byte_identity['lhs_published_bytes']} != "
              f"{byte_identity['rhs_indptr_last_x_per_entry']}")
    # Q2 accounting form: published bytes are the source; the finite-count
    # multiply is the CHECKED IDENTITY, not the accounting source
    retained_total_published = int(g_nodes.nbytes + g_dist.nbytes + g_pred.nbytes)
    q2_checked_identity = {
        "finite_entries_from_f3_line": p["finite_entries"],
        "finite_entries_x_per_entry": p["finite_entries"] * per_entry,
        "equal_to_published_bytes": (p["finite_entries"] * per_entry
                                     == retained_total_published),
    }
    # (c) re-simulation leg: replay the pinned selector POST-window from
    # published artifacts; retained at slice start = indptr[s] x per_entry
    # (parts appended PER-ROW IN SLICE ORDER — AggregateFlow :1032-1044 at
    # b387f90); element-wise equality with the observed schedule is the check
    sim_schedule = []
    sim_nofit_at_row = None
    if legs_pass_b:
        s_row = 0
        while s_row < n_dest_measured:
            retained_k = int(g_indptr[s_row]) * per_entry
            c_k = int(lfws.select_chunk(
                n_total_measured, n_dest_measured - s_row,
                dist_itemsize, pred_itemsize, mask_itemsize,
                fixed_live_measured, retained_k,
                row_temporaries, source_index_cost,
                margin_obs, cap_obs))
            if c_k == int(lfws.NO_FIT):
                sim_nofit_at_row = s_row
                break
            sim_schedule.append(c_k)
            s_row += c_k
    resim_equal = bool(legs_pass_b) and sim_schedule == list(sched)
    legs["c_resim_elementwise"] = {
        "resim_inputs": {
            "v_prime_row_width": n_total_measured,
            "n_dest": n_dest_measured,
            "fixed_live_bytes": fixed_live_measured,
            "per_source_bytes": per_source_recomputed,
            "per_entry_bytes": per_entry,
            "row_temporaries_bytes": row_temporaries,
            "source_index_cost_bytes": source_index_cost,
            "cap_bytes": cap_obs, "margin_bytes": margin_obs,
            "retained_source": "published indptr x per_entry at each "
                               "observed slice start",
            "selector": "_lfws.select_chunk imported from the installed venv",
        },
        "resim_schedule": sim_schedule,
        "observed_schedule": list(sched),
        "elementwise_equal": resim_equal,
        "chunks_equal": (len(sim_schedule) == chunks_obs) if legs_pass_b else None,
        "sim_nofit_at_row": sim_nofit_at_row,
        "pass": bool(legs_pass_b and resim_equal
                     and len(sim_schedule) == chunks_obs),
    }
    # (d) subsumed by leg (c)'s element-wise equality (spec rev 6): uniform
    # non-final widths modeled retained~0 fixtures; retention makes the real
    # schedule non-uniform downward. Recorded, not independently checked.
    legs["d_subsumed"] = {
        "pass": legs["c_resim_elementwise"]["pass"],
        "note": "subsumed by leg (c): no independent check - element-wise "
                "simulated==observed equality implies the tail property for "
                "whatever shape retention produces (spec rev 6) "
                "(h04 ruling 2026-09-27)",
    }
    # (e) exact per-instance reconstruction: for EVERY observed slice k,
    # indptr[s_k]*per_entry + schedule[k]*per_source + margin <= cap
    if legs_pass_b:
        instances = []
        s_row = 0
        for k, w in enumerate(sched):
            retained_k = int(g_indptr[s_row]) * per_entry
            lhs_k = retained_k + int(w) * per_source_recomputed + margin_obs
            instances.append({"slice": k, "start_row": s_row,
                              "retained_bytes": retained_k,
                              "lhs_bytes": lhs_k,
                              "slack_bytes": cap_obs - lhs_k,
                              "pass": lhs_k <= cap_obs})
            s_row += int(w)
        worst = min(instances, key=lambda r: r["slack_bytes"])
        legs["e_perinstance_reconstruction"] = {
            "form": "indptr[s_k] x per_entry + schedule[k] x per_source "
                    "+ margin <= cap, every observed slice",
            "n_slices_checked": len(instances),
            "pass_count": sum(1 for r in instances if r["pass"]),
            "all_instances_pass": all(r["pass"] for r in instances),
            "worst_instance": worst,
            "pass": bool(instances) and all(r["pass"] for r in instances),
        }
    else:
        legs["e_perinstance_reconstruction"] = {
            "pass": None,
            "note": "not evaluated: leg (b) failed"}
    all_pass = all(l.get("pass") is True for l in legs.values())
    REC["mechanism_receipt"] = {
        "form": "resim_perinstance_v1",
        "binding_legs": legs,
        "all_binding_legs_pass": all_pass,
        "byte_identity": byte_identity,
        "q2_retained_total_bytes": retained_total_published,
        "q2_checked_identity": q2_checked_identity,
        "aggregate_form_rejected": {
            "note": "aggregate form (total retained + widest slice x "
                    "per_source + margin) REJECTED at spec rev 6: the two "
                    "maxima are anti-correlated (widest slice at smallest "
                    "retained), so the aggregate passes only if "
                    "retained_total <= 3167080 B and its FAIL carries zero "
                    "per-instance information",
        },
        "self_validation_note": "observed schedule IS the real selector ground "
                                "truth; receipt divergence surfaces as "
                                "mismatch = the correct alarm",
        "orientation_nonbinding": {
            "c_first_slice": (sim_schedule[0] if sim_schedule else None),
            "note": "first-slice width from the full formula, printed for "
                    "orientation only, explicitly NON-binding and never the "
                    "G7 referent (spec ORIENTATION at rev 6); the binding "
                    "check is the re-simulation equality above"},
    }
    if not all_pass:
        REC["g7_stop_and_preserve"] = {
            "trigger": ("fallback=" + str(p["fallback"]) + " floor_fired="
                        + str(p["floor_fired"])) if not legs_pass_b
            else "binding receipt leg mismatch",
            "action": "run COMPLETED (full record incl. flow arrays, artifacts "
                      "and peak written); supervisor must stop the matrix "
                      "immediately per PD-E and preserve everything for h04 "
                      "adjudication"}
        write_out()
        print(f"[f3r-job] G7 STOP-AND-PRESERVE: "
              f"{REC['g7_stop_and_preserve']['trigger']}", flush=True)
        sys.exit(3)

write_out()
print(f"[f3r-job] ok win={(t1 - t0) / 1e9:.3f}s "
      f"peak={REC['peak_rss_ru_maxrss_mib']:.1f}MiB "
      f"f3_lines={REC['f3_tail_capture']['n_f3_lines']} "
      f"config_observed={REC['engine_config_observed']}", flush=True)
