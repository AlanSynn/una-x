#!/bin/zsh
# N10 qualification fire (una_10x post_p00, performance-owner scope).
# Stages (fail-closed, unique exit codes):
#   S10 fresh build venv + hatchling           (rc 10)
#   S20 sdist+wheel build -> /tmp/n10_build    (rc 20)
#   S30 wheel/sdist census + native exclusion  (rc 30)
#   S40 isolated install venv (copy of p01m)   (rc 40)
#   S50 installed-identity probes              (rc 50)
#   S60 loadavg gate + installed wheel leg     (rc 60/61)
#   S70 native fresh-build cost (copy, tree untouched) (rc 70)
#   S80 qualification JSON from tool outputs   (rc 80)
set -u
cd /Users/alansynn/orca/workspaces/una-x/main || exit 1

WS=/Users/alansynn/orca/workspaces/una-x
RUN=campaign_runs/una_10x/tenx-20261005T2331Z
N10=$RUN/n10
SCRATCH=/tmp/n10
DIST=$SCRATCH/dist
CONSOLE=$SCRATCH/console.log
mkdir -p "$N10/legs" "$SCRATCH"
: > "$CONSOLE"
exec > "$CONSOLE" 2>&1

say() { print -r -- "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

sha() { shasum -a 256 "$1" | awk '{print $1}'; }

# ---------- S10 build venv ----------
say "S10 begin"
rm -rf "$WS/venv-n10wheel" "$SCRATCH/venv_install" "$DIST"
"$WS/venv-p01m/bin/python" -m venv "$WS/venv-n10wheel" || exit 10
"$WS/venv-n10wheel/bin/python" -m pip install -q hatchling || { say "hatchling install FAILED (offline?)"; exit 11; }
"$WS/venv-n10wheel/bin/python" -m pip show hatchling | head -2 || exit 12
say "S10 ok"

# ---------- S20 build ----------
say "S20 begin (sdist+wheel from tree HEAD $(git rev-parse HEAD))"
# hatchling has no -o flag; it builds into <tree>/dist (repo-gitignored,
# .gitignore:12).  Artifacts are moved to $DIST and the in-tree dist/ is
# removed so the working tree carries nothing extra.
rm -rf dist
"$WS/venv-n10wheel/bin/python" -m hatchling build -t wheel -t sdist || exit 20
mkdir -p "$DIST"
mv dist/*.whl dist/*.tar.gz "$DIST"/ || exit 21
rmdir dist || rm -rf dist
ls -la "$DIST" || exit 21
WHEEL=$(ls "$DIST"/*.whl 2>/dev/null | head -1)
SDIST=$(ls "$DIST"/*.tar.gz 2>/dev/null | head -1)
[ -n "$WHEEL" ] && [ -n "$SDIST" ] || exit 22
say "S20 ok wheel=$(basename "$WHEEL") sdist=$(basename "$SDIST")"

# ---------- S30 census ----------
say "S30 begin"
"$WS/venv-p01m/bin/python" - "$WHEEL" "$SDIST" "$N10" <<'PYEOF' || exit 31
import json, sys, tarfile, zipfile
wheel, sdist, n10 = sys.argv[1], sys.argv[2], sys.argv[3]

wnames = zipfile.ZipFile(wheel).namelist()
snames = tarfile.open(sdist).getnames()

wheel_native = [n for n in wnames if "/native/" in n or n.endswith(".so")
                or "sitecustomize" in n or "una_n00_trace_producer" in n]
sdist_native = [n for n in snames if "/native/" in n]
pkg = sorted(n for n in wnames if n.endswith(".py"))

out = {
  "wheel": {"path": wheel, "sha256": None, "entries": len(wnames),
            "py_entries": len(pkg),
            "native_or_ext_or_sitecustomize_entries": wheel_native,
            "extension_modules": [n for n in wnames if n.endswith(".so")]},
  "sdist": {"path": sdist, "sha256": None, "entries": len(snames),
            "native_entries": sdist_native},
  "law_note": "wheel packages ONLY src/urban_network_analysis "
              "(pyproject tool.hatch.build.targets.wheel); native/ is "
              "repo-scoped. sdist native presence = sources ride along "
              "with NO build wiring (hatch hook only bakes commit date).",
}
import hashlib
for sec, path in (("wheel", wheel), ("sdist", sdist)):
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    out[sec]["sha256"] = h
json.dump(out, open(f"{n10}/packaging_census.json", "w"), indent=1)
assert not wheel_native, f"wheel carries native/ext/sitecustomize: {wheel_native}"
print("WHEEL clean:", len(wnames), "entries,", len(pkg), "py;",
      "SDIST native entries:", len(sdist_native))
PYEOF
WHEEL_SHA=$(sha "$WHEEL"); SDIST_SHA=$(sha "$SDIST")
say "S30 ok wheel_sha=$WHEEL_SHA sdist_sha=$SDIST_SHA"

# ---------- S40 isolated install ----------
say "S40 begin (copy p01m -> install venv, replace editable with wheel)"
cp -a "$WS/venv-p01m" "$SCRATCH/venv_install" || exit 40
"$SCRATCH/venv_install/bin/python" -m pip uninstall -q -y urban-network-analysis || exit 41
"$SCRATCH/venv_install/bin/python" -m pip install -q --no-deps "$WHEEL" || exit 42
"$SCRATCH/venv_install/bin/python" -m pip show urban-network-analysis | head -3 || exit 43
say "S40 ok"

# ---------- S50 identity probes ----------
say "S50 begin"
UNA_NATIVE_TRACE=1 PYTHONDONTWRITEBYTECODE=1 \
"$SCRATCH/venv_install/bin/python" - "$N10" "$WHEEL_SHA" "$SDIST_SHA" <<'PYEOF' || exit 51
import importlib, json, os, sys
n10, wheel_sha, sdist_sha = sys.argv[1], sys.argv[2], sys.argv[3]
import urban_network_analysis as u
pkg_file = u.__file__
assert "/venv_install/" in pkg_file, f"import did NOT resolve to install venv: {pkg_file}"
pkg_dir = os.path.dirname(pkg_file)
assert not os.path.exists(os.path.join(pkg_dir, "sitecustomize.py"))
try:
    import una_n00_trace_producer
    raise SystemExit(f"native extension IMPORTABLE in installed context: {una_n00_trace_producer.__file__}")
except ImportError as e:
    ext_err = str(e)
import sitecustomize as sc
sc_file = getattr(sc, "__file__", None)
native_dir_abs = "/Users/alansynn/orca/workspaces/una-x/main/native"
assert not (sc_file and sc_file.startswith(native_dir_abs)), \
    f"native sitecustomize on sys.path in installed context: {sc_file}"
# installed execution smoke: construct the public object (no timed call here)
from urban_network_analysis import UNA
p = UNA(verbosity=0)
out = {
  "probe_env": {"UNA_NATIVE_TRACE": os.environ.get("UNA_NATIVE_TRACE"),
                "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE")},
  "package_file": pkg_file,
  "package_dir": pkg_dir,
  "native_extension_importable": False,
  "native_extension_import_error": ext_err,
  "sitecustomize_file": sc_file,
  "native_sitecustomize_shadowing": bool(sc_file and sc_file.startswith(native_dir_abs)),
  "wheel_sha256": wheel_sha, "sdist_sha256": sdist_sha,
  "executable": sys.executable,
  "una_constructed": True,
}
json.dump(out, open(f"{n10}/install_identity.json", "w"), indent=1)
print("IDENTITY OK:", pkg_file)
PYEOF
say "S50 ok"

# ---------- S60 loadavg gate + installed wheel leg ----------
say "S60 begin (gate: 1-min loadavg < 8, 90-min bounded)"
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
say "S60 leg done"

# ---------- S70 native fresh-build cost ----------
say "S70 begin (build in /tmp copy; tree native/_build untouched)"
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
"$WS/venv-p01m/bin/python" - "$N10" "$SCRATCH" "$WHEEL_SHA" "$SDIST_SHA" "$BUILD_S" "$PRE_LA" <<'PYEOF' || exit 81
import json, os, re, sys
n10, scratch, wheel_sha, sdist_sha, build_s, pre_la = \
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]

leg = json.load(open(f"{n10}/legs/wheel_installed_numba.json"))
ident = json.load(open(f"{n10}/install_identity.json"))
pkg = json.load(open(f"{n10}/packaging_census.json"))

req = [f for f in leg["output_files"] if not f["rel"].split("/")[0].startswith(".")]
hid = [f for f in leg["output_files"] if f["rel"].split("/")[0].startswith(".")]
req_files = len(req); req_bytes = sum(f["bytes"] for f in req)
assert (req_files, req_bytes) == (64, 147829892), (req_files, req_bytes)
assert leg["zero_failed_jobs"], leg["rows"]

# bitwise census vs the N00 matrix numba reference arm (L2A1_numba)
ref = json.load(open("campaign_runs/una_10x/tenx-20261005T2331Z/n00/legs/L2A1_numba.json"))
ts = re.compile(r"accessibility_\d{4}-\d{2}-\d{2}_\d{4}")
def census(j):
    return {ts.sub("accessibility_TS", f["rel"]): f["sha256"]
            for f in j["output_files"]
            if not f["rel"].split("/")[0].startswith(".")}
bitwise_equal = census(leg) == census(ref)

wall = leg["timing"]["public_call_wall_s"]
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
   "law": "same cohort, same schedule (w4/t1/q64, cache off), numba vs native trace-producer under one shared trace key",
   "matrix_walls_s": matrix_walls,
   "verdict": "not a demonstrated lever: native legs contradict each other "
              "(12.5244 vs 16.2113) at n=2 under co-tenancy; numba controls "
              "15.1751/15.7252; no reliable full-region benefit (dossier 08 "
              "'reject default where no measured full-region benefit')",
 },
 "install_evidence": {
   "wheel_native_free": pkg["wheel"]["native_or_ext_or_sitecustomize_entries"] == [],
   "sdist_native_entries": len(pkg["sdist"]["native_entries"]),
   "installed_context": ident,
   "installed_wheel_leg": {
     "file": "legs/wheel_installed_numba.json",
     "wall_s": wall, "pre_leg_loadavg_1m": float(pre_la),
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
   "fold_region": "stays numba (probe_order.txt: fastmath+vectorized "
      "lane partials + numba-specific argsort tie permutation)",
 },
}
json.dump(q, open(f"{n10}/native_qualification.json", "w"), indent=1)
print(f"QUAL OK wall={wall}s bitwise_equal={bitwise_equal} "
      f"census={req_files}f/{req_bytes}B build_s={build_s}")
PYEOF
say "S80 ok"
say "N10 FIRE COMPLETE rc=0"
