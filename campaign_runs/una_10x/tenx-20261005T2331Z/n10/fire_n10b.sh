#!/bin/zsh
# N10 continuation fire (fix wheel: install shim force-include, then
# finish the ladder S30b/S40b/S50b/S60b/S70/S80).
#   S30b rebuild wheel+sdist, census (shim present, native-free)  (rc 30/31)
#   S40b reinstall wheel into venv_install                        (rc 40)
#   S50b identity probes rerun                                    (rc 50)
#   S60b loadavg gate + installed wheel leg                       (rc 60/61)
#   S70  native fresh-build cost                                  (rc 70)
#   S80  qualification JSON                                       (rc 81)
set -u
cd /Users/alansynn/orca/workspaces/una-x/main || exit 1

WS=/Users/alansynn/orca/workspaces/una-x
RUN=campaign_runs/una_10x/tenx-20261005T2331Z
N10=$RUN/n10
SCRATCH=/tmp/n10
DIST=$SCRATCH/dist_fixed
CONSOLE=$SCRATCH/console_fire2.log
: > "$CONSOLE"
exec > "$CONSOLE" 2>&1

say() { print -r -- "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
sha() { shasum -a 256 "$1" | awk '{print $1}'; }

# ---------- S30b rebuild + census ----------
say "S30b begin (fixed wheel from HEAD $(git rev-parse HEAD))"
rm -rf dist "$DIST"
"$WS/venv-n10wheel/bin/python" -m hatchling build -t wheel -t sdist || exit 30
mkdir -p "$DIST"
mv dist/*.whl dist/*.tar.gz "$DIST"/ || exit 30
rmdir dist
WHEEL=$(ls "$DIST"/*.whl | head -1)
SDIST=$(ls "$DIST"/*.tar.gz | head -1)
"$WS/venv-p01m/bin/python" - "$WHEEL" "$SDIST" "$N10" <<'PYEOF' || exit 31
import hashlib, json, sys, tarfile, zipfile
wheel, sdist, n10 = sys.argv[1], sys.argv[2], sys.argv[3]
wnames = zipfile.ZipFile(wheel).namelist()
snames = tarfile.open(sdist).getnames()
shim = [n for n in wnames if n == "una_batch_worker_main.py"]
native = [n for n in wnames if "/native/" in n or n.endswith(".so")
          or "sitecustomize" in n or "una_n00_trace_producer" in n]
sdist_shim = [n for n in snames if n.endswith("src/una_batch_worker_main.py")]
out = {
  "wheel": {"path": wheel, "sha256": None, "entries": len(wnames),
            "spawn_shim_at_root": bool(shim),
            "native_or_ext_or_sitecustomize_entries": native,
            "extension_modules": [n for n in wnames if n.endswith(".so")]},
  "sdist": {"path": sdist, "sha256": None, "entries": len(snames),
            "spawn_shim_in_src": bool(sdist_shim)},
  "law_note": "fixed wheel: top-level spawn shim force-included at wheel "
              "root (defect fix for ModuleNotFoundError at pool.start); "
              "still native/ext/sitecustomize-free by construction",
}
for sec, path in (("wheel", wheel), ("sdist", sdist)):
    out[sec]["sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
json.dump(out, open(f"{n10}/packaging_census_fixed.json", "w"), indent=1)
assert shim, "spawn shim MISSING from fixed wheel"
assert not native, f"native/ext leaked into wheel: {native}"
print("WHEEL FIXED: entries", len(wnames), "shim-at-root:", bool(shim))
PYEOF
WHEEL_SHA=$(sha "$WHEEL"); SDIST_SHA=$(sha "$SDIST")
say "S30b ok wheel_sha=$WHEEL_SHA sdist_sha=$SDIST_SHA"

# ---------- S40b reinstall ----------
say "S40b begin"
"$SCRATCH/venv_install/bin/python" -m pip install -q --no-deps \
    --force-reinstall "$WHEEL" || exit 40
"$SCRATCH/venv_install/bin/python" -m pip show urban-network-analysis | head -2 || exit 41
say "S40b ok"

# ---------- S50b identity probes ----------
say "S50b begin"
UNA_NATIVE_TRACE=1 PYTHONDONTWRITEBYTECODE=1 \
"$SCRATCH/venv_install/bin/python" - "$N10" "$WHEEL_SHA" "$SDIST_SHA" <<'PYEOF' || exit 51
import json, os, sys
n10, wheel_sha, sdist_sha = sys.argv[1], sys.argv[2], sys.argv[3]
import urban_network_analysis as u
pkg_file = u.__file__
assert "/venv_install/" in pkg_file, f"not installed context: {pkg_file}"
assert not os.path.exists(os.path.join(os.path.dirname(pkg_file), "sitecustomize.py"))
try:
    import una_n00_trace_producer
    raise SystemExit(f"native extension importable: {una_n00_trace_producer.__file__}")
except ImportError as e:
    ext_err = str(e)
import una_batch_worker_main as shim   # must now resolve from the WHEEL
shim_file = shim.__file__
assert "/venv_install/" in shim_file, f"shim not from wheel: {shim_file}"
assert getattr(shim, "pool_main", None) is not None
import sitecustomize as sc
sc_file = getattr(sc, "__file__", None)
native_dir_abs = "/Users/alansynn/orca/workspaces/una-x/main/native"
assert not (sc_file and sc_file.startswith(native_dir_abs))
from urban_network_analysis import UNA
UNA(verbosity=0)
out = {
  "probe_env": {"UNA_NATIVE_TRACE": os.environ.get("UNA_NATIVE_TRACE"),
                "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE")},
  "package_file": pkg_file,
  "spawn_shim_file": shim_file,
  "native_extension_importable": False,
  "native_extension_import_error": ext_err,
  "sitecustomize_file": sc_file,
  "wheel_sha256": wheel_sha, "sdist_sha256": sdist_sha,
  "executable": sys.executable,
  "una_constructed": True,
}
json.dump(out, open(f"{n10}/install_identity.json", "w"), indent=1)
print("IDENTITY OK shim:", shim_file)
PYEOF
say "S50b ok"

# ---------- S60b gate + wheel leg ----------
say "S60b begin (gate: 1-min loadavg < 8, 90-min bounded)"
GATE_TRIES=0
while true; do
  LA=$(sysctl -n vm.loadavg | awk '{print $2}')
  ok=$(python3 -c "print(1 if float('$LA') < 8.0 else 0)")
  [ "$ok" = "1" ] && break
  GATE_TRIES=$((GATE_TRIES+1))
  [ "$GATE_TRIES" -gt 90 ] && { say "GATE TIMEOUT loadavg=$LA"; exit 61; }
  say "gate hold #$GATE_TRIES loadavg=$LA"
  sleep 60
done
PRE_LA=$(sysctl -n vm.loadavg | awk '{print $2}')
say "gate open pre_leg_loadavg=$PRE_LA"
rm -rf "$SCRATCH/numba_wheelleg" "$SCRATCH/out_wheelleg"
mkdir -p "$SCRATCH/numba_wheelleg" "$SCRATCH/out_wheelleg"
PAIRING=/Users/alansynn/orca/workspaces/una-x/main/campaign_data/inputs/upstream_Boston/Boston/local_audit_manifest.csv
DATAFOLDER=/Users/alansynn/orca/workspaces/una-x/main/campaign_data/inputs/upstream_Boston/Boston
NUMBA_CACHE_DIR=$SCRATCH/numba_wheelleg PYTHONDONTWRITEBYTECODE=1 \
UNA_BATCH_PHASES= UNA_NATIVE_TRACE= \
"$SCRATCH/venv_install/bin/python" "$RUN/p05s/run_screen_leg.py" \
  --tree /Users/alansynn/orca/workspaces/una-x/main \
  --pairing "$PAIRING" --data-folder "$DATAFOLDER" \
  --outdir "$SCRATCH/out_wheelleg" --tag wheel_installed_numba \
  --out "$N10/legs/wheel_installed_numba.json" \
  --parallel --workers 4 --threads 1 --queue-depth 64 --cache-mode off \
  || exit 60
say "S60b leg done"

# ---------- S70 native build cost ----------
say "S70 begin"
rm -rf "$SCRATCH/native_copy"
mkdir -p "$SCRATCH/native_copy"
rsync -a --exclude '_build' --exclude '__pycache__' native/ "$SCRATCH/native_copy/"
T0=$(python3 -c "import time; print(time.perf_counter())")
"$WS/venv-n00/bin/python" "$SCRATCH/native_copy/build.py" || exit 70
T1=$(python3 -c "import time; print(time.perf_counter())")
BUILD_S=$(python3 -c "print(round(float('$T1')-float('$T0'),4))")
ls "$SCRATCH/native_copy/_build" || exit 71
say "S70 ok native_build_s=$BUILD_S"

# ---------- S80 qualification JSON ----------
say "S80 begin"
"$WS/venv-p01m/bin/python" - "$N10" "$WHEEL_SHA" "$SDIST_SHA" "$BUILD_S" "$PRE_LA" <<'PYEOF' || exit 81
import json, re, sys
n10, wheel_sha, sdist_sha, build_s, pre_la = \
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

leg = json.load(open(f"{n10}/legs/wheel_installed_numba.json"))
ident = json.load(open(f"{n10}/install_identity.json"))
pkg = json.load(open(f"{n10}/packaging_census_fixed.json"))

req = [f for f in leg["output_files"] if not f["rel"].split("/")[0].startswith(".")]
req_files = len(req); req_bytes = sum(f["bytes"] for f in req)
assert (req_files, req_bytes) == (64, 147829892), (req_files, req_bytes)
assert leg["zero_failed_jobs"], leg["rows"]

ref = json.load(open("campaign_runs/una_10x/tenx-20261005T2331Z/n00/legs/L2A1_numba.json"))
ts = re.compile(r"accessibility_\d{4}-\d{2}-\d{2}_\d{4}")
def census(j):
    return {ts.sub("accessibility_TS", f["rel"]): f["sha256"]
            for f in j["output_files"]
            if not f["rel"].split("/")[0].startswith(".")}
bitwise_equal = census(leg) == census(ref)

matrix_walls = {}
for t in ("L2A1_numba", "L1B1_native", "L2A2_numba", "L1B2_native"):
    matrix_walls[t] = json.load(open(
        f"campaign_runs/una_10x/tenx-20261005T2331Z/n00/legs/{t}.json")
        )["timing"]["public_call_wall_s"]

q = {
 "schema": "una-10x-n10-native-qualification-1",
 "task": "N10 qualify native region and install cost (performance-owner)",
 "inputs": {
   "n00_evidence": "campaign_runs/una_10x/tenx-20261005T2331Z/n00/native_implementation.json",
   "c_arm_matrix": "n00/legs L2A1/L1B1/L2A2/L1B2 (HEAD 3b55dd28db7d)",
   "wheel_sha256": wheel_sha, "sdist_sha256": sdist_sha,
 },
 "equal_C_comparison": {
   "law": "same cohort, same schedule (w4/t1/q64, cache off), numba vs "
          "native trace-producer under one shared trace key",
   "matrix_walls_s": matrix_walls,
   "verdict": "not a demonstrated lever: native legs contradict each other "
              "at n=2 under co-tenancy; no reliable full-region benefit "
              "(dossier 08 'reject default where no measured full-region "
              "benefit')",
 },
 "install_evidence": {
   "defect_found_and_fixed": {
     "defect": "top-level spawn shim src/una_batch_worker_main.py was NOT "
               "packaged: wheel packages only src/urban_network_analysis, "
               "so parallel RunBatch crashed in any non-editable install "
               "(ModuleNotFoundError at pool.start, runtime.py:379); "
               "INHERITED from parent e7c9eda (same pyproject packages "
               "line) and latent since commit 0acbb11",
     "fix": "pyproject [tool.hatch.build.targets.wheel.force-include] "
            "maps src/una_batch_worker_main.py -> una_batch_worker_main.py "
            "at the wheel root (module layout unchanged: the top-level "
            "stdlib-only placement is the design, dossier 05)",
     "defective_wheel_census": "n10/packaging_census.json (pre-fix state, "
               "preserved; fire-1 console preserved at n10/)",
   },
   "wheel_native_free": pkg["wheel"]["native_or_ext_or_sitecustomize_entries"] == [],
   "spawn_shim_at_wheel_root": pkg["wheel"]["spawn_shim_at_root"],
   "installed_context": ident,
   "installed_wheel_leg": {
     "file": "legs/wheel_installed_numba.json",
     "wall_s": leg["timing"]["public_call_wall_s"],
     "pre_leg_loadavg_1m": float(pre_la),
     "required_census": {"files": req_files, "bytes": req_bytes},
     "bitwise_equal_to_numba_matrix_arm": bitwise_equal,
     "interpreter": leg["pins"]["executable"],
   },
 },
 "build_cost_charged_s": float(build_s),
 "decision": {
   "gate_stack_default": "numba (unchanged): native trace producer NOT "
      "admitted to the 10x gate stack; installed wheel is numba-only by "
      "construction and the C-arm shows no reliable full-region win",
   "native_capability": "RETAINED as platform capability: bit-exact "
      "producer (40/40 native tests, native==numba bitwise at batch "
      "scale), env-gated opt-in, repo-scoped, never silently engaged",
   "fold_region": "stays numba (n00/probe_order.txt: fastmath+vectorized "
      "lane partials + numba-specific argsort tie permutation)",
 },
}
json.dump(q, open(f"{n10}/native_qualification.json", "w"), indent=1)
print(f"QUAL OK wall={leg['timing']['public_call_wall_s']}s "
      f"bitwise_equal={bitwise_equal} census={req_files}f/{req_bytes}B "
      f"build_s={build_s}")
PYEOF
say "S80 ok"
say "N10 FIRE-2 COMPLETE rc=0"
