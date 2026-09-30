"""F3R: build BOTH arm wheels from their PINNED commits into FRESH venvs
(f1r_build_candidate.py recipe; h04 released pre-fire authoring + builds
independent of the pending spec sha).

Arm pairing per spec arms_and_sources:
  b0   = c37cf22 (F3I implemented-at tip, line-MINUS-F3):
         AggregateFlow.py blob 455e92ad..., Engines/_large_flow_workspace.py
         ABSENT (ls-tree verified at spec authoring).
  cand = b387f90 (F3I admitted implementation, line-PLUS-F3):
         AggregateFlow.py blob 423efe33..., _large_flow_workspace.py blob
         6c374883...  The F3 delta is the ONLY source difference between the
         arms — both carry the A1+F1 lineage (nogil probe expected True on
         BOTH arms; the F1R b0 differentiation does not apply here).

Additions vs the F1R recipe:
  - two pinned-commit archives, two wheels (wheels_f3r/b0/, wheels_f3r/cand/),
    two fresh runtime venvs (f3r_b0_wheel / f3r_cand_wheel)
  - packaged-blob verification covers BOTH files on cand; b0 additionally
    asserts the module is ABSENT from the wheel namelist and from the
    installed tree
  - in-venv F3 probes: cand imports the private module and asserts the
    frozen F5 constants (268435456 / 67108864); b0 asserts find_spec None
  - cache roots are designated (not created here): the supervisor points
    NUMBA_CACHE_DIR at nbc_f3r_b0 / nbc_f3r_cand per arm at run time

Run: campaign-python f3r_build_arms.py   (heavy: pre-fire, pre-lease)
Writes: campaign_data/f3r_wheel_and_venv.json
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
WT = f"{CAMPAIGN}/wt-large-e2e"
AGG_REL = "src/urban_network_analysis/Engines/AggregateFlow.py"
LFWS_REL = "src/urban_network_analysis/Engines/_large_flow_workspace.py"
ARMS = {
    "b0": {
        "commit": "c37cf224afe678e0500749531a43c6be181dabef",
        "agg_blob": "455e92adb5f23748903435d3d6340a5d8a9c47d3",
        "lfws_blob": None,  # module must be ABSENT
    },
    "cand": {
        "commit": "b387f901caa1b529138452b084d016b9b1bead50",
        "agg_blob": "423efe332db06ca94d8832eb14e500810b6da13d",
        "lfws_blob": "6c3748838bc5f626a7b7e733e74d9a8e1c9e1aff",
    },
}
BASE_PY = "/opt/homebrew/opt/python@3.11/bin/python3.11"
CAMPAIGN_PY = f"{CAMPAIGN}/venvs/campaign/bin/python"
BUILD_DIR = f"{CAMPAIGN}/campaign_data/f3r_build"
WHEELS = f"{CAMPAIGN}/campaign_data/wheels_f3r"
BUILD_VENV = f"{CAMPAIGN}/campaign_data/venvs/f3r_build"
RUN_VENVS = {"b0": f"{CAMPAIGN}/campaign_data/venvs/f3r_b0_wheel",
             "cand": f"{CAMPAIGN}/campaign_data/venvs/f3r_cand_wheel"}
CACHE_ROOTS = {"b0": f"{CAMPAIGN}/campaign_data/nbc_f3r_b0",
               "cand": f"{CAMPAIGN}/campaign_data/nbc_f3r_cand"}
RECORD = f"{CAMPAIGN}/campaign_data/f3r_wheel_and_venv.json"
RULED_CAP_BYTES = 268435456
RULED_MARGIN_BYTES = 67108864
PINNED = {"python": "3.11.16", "numpy": "2.4.6", "numba": "0.67.0",
          "geopandas": "1.1.4", "pyogrio": "0.13.0", "shapely": "2.1.2"}
TIMEOUT_S = 1800

steps = []


def run(cmd, timeout=TIMEOUT_S, cwd=None):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, cwd=cwd)
    steps.append({"cmd": cmd if isinstance(cmd, str) else " ".join(cmd),
                  "cwd": cwd, "rc": p.returncode,
                  "wall_s": round(time.time() - t0, 1),
                  "stdout_tail": p.stdout[-3000:], "stderr_tail": p.stderr[-3000:]})
    return p


def fail(msg):
    rec = {"task": "F3R", "record": "arm_wheels_and_venvs", "status": "BLOCKED",
           "blocked_reason": msg, "steps": steps,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(RECORD, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[f3r-build] BLOCKED: {msg}")
    sys.exit(2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_of(path):
    p = subprocess.run(["git", "-C", WT, "hash-object", path],
                       capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        return None
    return p.stdout.strip()


def git_commit_blob(commit, rel):
    p = subprocess.run(["git", "-C", WT, "rev-parse", f"{commit}:{rel}"],
                       capture_output=True, text=True, timeout=60)
    return (p.stdout.strip(), p.returncode) if p.returncode == 0 else (None, p.returncode)


def wheel_file_blob(wheel_path, rel_suffix, extract_dir):
    """Extract ONE file matching rel_suffix from the wheel; return its blob."""
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)
    with zipfile.ZipFile(wheel_path) as z:
        names = [n for n in z.namelist() if n.endswith(rel_suffix)]
        if len(names) != 1:
            fail(f"wheel {wheel_path}: expected 1 {rel_suffix}, found {names}")
        z.extract(names[0], extract_dir)
    return git_blob_of(os.path.join(extract_dir, names[0]))


def main():
    # 0. pin checks on BOTH commits (independent of HEAD movement)
    for arm, a in ARMS.items():
        blob, rc = git_commit_blob(a["commit"], AGG_REL)
        if rc != 0 or blob != a["agg_blob"]:
            fail(f"{arm} AggregateFlow blob pin failed: {blob!r} != {a['agg_blob']}")
        if a["lfws_blob"] is None:
            _, rc = git_commit_blob(a["commit"], LFWS_REL)
            if rc == 0:
                fail(f"{arm}: _large_flow_workspace unexpectedly present at commit")
        else:
            blob, rc = git_commit_blob(a["commit"], LFWS_REL)
            if rc != 0 or blob != a["lfws_blob"]:
                fail(f"{arm} _large_flow_workspace blob pin failed: {blob!r}")
        p = subprocess.run(["git", "-C", WT, "merge-base", "--is-ancestor",
                            a["commit"], "HEAD"], capture_output=True, timeout=60)
        if p.returncode != 0:
            fail(f"{arm} commit is not an ancestor of wt-large-e2e HEAD")
    p = run(["git", "-C", WT, "rev-parse", "HEAD"])
    head_sha = p.stdout.strip()
    p = run(["git", "-C", WT, "status", "--porcelain", "--untracked-files=no"])
    dirty = [l for l in p.stdout.splitlines() if l.strip()]
    dirty_outside = [l for l in dirty if not l.startswith(" M campaigns/")
                     and not l.startswith("M campaigns/")]
    if p.returncode != 0 or dirty_outside:
        fail(f"wt-large-e2e tracked tree not clean outside campaigns/:\n{p.stdout}")

    # 1. build venv (hatchling), shared across both arms
    if os.path.exists(BUILD_VENV):
        shutil.rmtree(BUILD_VENV)
    p = run([BASE_PY, "-m", "venv", BUILD_VENV])
    if p.returncode != 0:
        fail(f"build venv creation failed: {p.stderr[-500:]}")
    p = run([f"{BUILD_VENV}/bin/pip", "install", "--quiet", "hatchling"])
    if p.returncode != 0:
        fail(f"hatchling install failed: {p.stderr[-800:]}")

    # 2. constraints from campaign venv (once)
    p = run([CAMPAIGN_PY, "-m", "pip", "freeze"])
    if p.returncode != 0:
        fail("pip freeze on campaign venv failed")
    freeze_lines = [l.strip() for l in p.stdout.splitlines() if l.strip()
                    and not l.startswith("-e ")
                    and "urban-network-analysis" not in l
                    and "urban_network_analysis" not in l]
    constraints = f"{BUILD_DIR}/constraints.txt"
    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(constraints, "w") as f:
        f.write("\n".join(freeze_lines) + "\n")

    arm_records = {}
    for arm, a in ARMS.items():
        # 3. archive the pinned commit into a clean build copy
        src_copy = f"{BUILD_DIR}/{arm}_src"
        if os.path.exists(src_copy):
            shutil.rmtree(src_copy)
        os.makedirs(src_copy)
        t0 = time.time()
        p = subprocess.run(
            f"git -C {WT} archive {a['commit']} | tar -x -C {src_copy}",
            shell=True, capture_output=True, timeout=300)
        steps.append({"cmd": f"git archive {a['commit'][:7]} | tar -x -C {arm}_src",
                      "wall_s": round(time.time() - t0, 1), "rc": p.returncode,
                      "stderr_tail": p.stderr.decode("utf-8", errors="replace")[-2000:]})
        if p.returncode != 0:
            fail(f"{arm}: archive extraction failed")
        archived_agg = git_blob_of(f"{src_copy}/{AGG_REL}")
        if archived_agg != a["agg_blob"]:
            fail(f"{arm}: archived AggregateFlow blob {archived_agg} != {a['agg_blob']}")

        # 4. build the wheel
        wdir = f"{WHEELS}/{arm}"
        if os.path.exists(wdir):
            shutil.rmtree(wdir)
        os.makedirs(wdir)
        p = run([f"{BUILD_VENV}/bin/pip", "wheel", ".", "--no-build-isolation",
                 "--no-deps", "-w", wdir], cwd=src_copy, timeout=900)
        if p.returncode != 0:
            fail(f"{arm}: wheel build failed: {p.stderr[-1500:]}")
        whls = [f for f in os.listdir(wdir) if f.endswith(".whl")]
        if len(whls) != 1:
            fail(f"{arm}: expected exactly one wheel, found {whls}")
        wheel_path = f"{wdir}/{whls[0]}"
        wheel_sha = sha256_file(wheel_path)
        pkg_agg = wheel_file_blob(wheel_path, "Engines/AggregateFlow.py",
                                  f"{BUILD_DIR}/{arm}_wheel_extract")
        if pkg_agg != a["agg_blob"]:
            fail(f"{arm}: wheel packaged AggregateFlow blob {pkg_agg} != {a['agg_blob']}")
        with zipfile.ZipFile(wheel_path) as z:
            lfws_in_wheel = [n for n in z.namelist()
                             if n.endswith("_large_flow_workspace.py")]
        if a["lfws_blob"] is None:
            if lfws_in_wheel:
                fail(f"{arm}: _large_flow_workspace unexpectedly IN wheel")
            pkg_lfws = None
        else:
            pkg_lfws = wheel_file_blob(wheel_path, "Engines/_large_flow_workspace.py",
                                       f"{BUILD_DIR}/{arm}_wheel_extract_lfws")
            if pkg_lfws != a["lfws_blob"]:
                fail(f"{arm}: wheel packaged _large_flow_workspace blob "
                     f"{pkg_lfws} != {a['lfws_blob']}")

        # 5. fresh runtime venv
        venv = RUN_VENVS[arm]
        if os.path.exists(venv):
            shutil.rmtree(venv)
        p = run([BASE_PY, "-m", "venv", venv])
        if p.returncode != 0:
            fail(f"{arm}: runtime venv creation failed: {p.stderr[-500:]}")
        p = run([f"{venv}/bin/pip", "install", "--quiet", "--upgrade", "pip"],
                timeout=600)
        p = run([f"{venv}/bin/pip", "install", "--quiet", "-c", constraints,
                 wheel_path, "pyarrow", "pyogrio", "threadpoolctl"],
                timeout=TIMEOUT_S)
        if p.returncode != 0:
            fail(f"{arm}: wheel install failed: {p.stderr[-1500:]}")

        # 6. installed-blob checks
        site = f"{venv}/lib/python3.11/site-packages/urban_network_analysis"
        inst_agg = git_blob_of(f"{site}/Engines/AggregateFlow.py")
        if inst_agg != a["agg_blob"]:
            fail(f"{arm}: installed AggregateFlow blob {inst_agg} != {a['agg_blob']}")
        if a["lfws_blob"] is None:
            if os.path.exists(f"{site}/Engines/_large_flow_workspace.py"):
                fail(f"{arm}: _large_flow_workspace unexpectedly installed")
            inst_lfws = None
        else:
            inst_lfws = git_blob_of(f"{site}/Engines/_large_flow_workspace.py")
            if inst_lfws != a["lfws_blob"]:
                fail(f"{arm}: installed _large_flow_workspace blob {inst_lfws} "
                     f"!= {a['lfws_blob']}")

        # 7. in-venv identity + F3 probes
        if arm == "cand":
            probe = (
                "import json, os, sys, inspect, numba, numpy, scipy, geopandas, "
                "shapely, pyogrio, pyarrow, threadpoolctl\n"
                "import urban_network_analysis as u\n"
                "from urban_network_analysis.Engines import AggregateFlow as AF\n"
                "from urban_network_analysis.Engines import _large_flow_workspace as lfws\n"
                "src = inspect.getsource(AF._accumulate_od_flow)\n"
                "af_src = inspect.getsource(AF)\n"
                "p = os.path.dirname(os.path.abspath(u.__file__))\n"
                "print(json.dumps({'una_path': p, 'site': sys.prefix,\n"
                " 'nogil': 'nogil=True' in src.split('def ')[0],\n"
                " 'af_imports_lfws': '_large_flow_workspace' in af_src,\n"
                " 'cap': int(lfws.DEFAULT_CAP_BYTES), 'margin': int(lfws.MARGIN_BYTES),\n"
                " 'versions': {'python': sys.version.split()[0], "
                "'numpy': numpy.__version__, 'numba': numba.__version__,\n"
                " 'geopandas': geopandas.__version__, 'pyogrio': pyogrio.__version__, "
                "'shapely': shapely.__version__}}))\n"
            )
            extra = {}
        else:
            probe = (
                "import importlib.util, json, os, sys, inspect, numba, numpy, "
                "scipy, geopandas, shapely, pyogrio, pyarrow, threadpoolctl\n"
                "import urban_network_analysis as u\n"
                "from urban_network_analysis.Engines import AggregateFlow as AF\n"
                "src = inspect.getsource(AF._accumulate_od_flow)\n"
                "af_src = inspect.getsource(AF)\n"
                "p = os.path.dirname(os.path.abspath(u.__file__))\n"
                "print(json.dumps({'una_path': p, 'site': sys.prefix,\n"
                " 'nogil': 'nogil=True' in src.split('def ')[0],\n"
                " 'af_imports_lfws': '_large_flow_workspace' in af_src,\n"
                " 'lfws_find_spec': importlib.util.find_spec("
                "'urban_network_analysis.Engines._large_flow_workspace') is None,\n"
                " 'versions': {'python': sys.version.split()[0], "
                "'numpy': numpy.__version__, 'numba': numba.__version__,\n"
                " 'geopandas': geopandas.__version__, 'pyogrio': pyogrio.__version__, "
                "'shapely': shapely.__version__}}))\n"
            )
            extra = {}
        try:
            p = subprocess.run([f"{venv}/bin/python", "-c", probe],
                               capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            fail(f"{arm}: identity probe timed out")
        if p.returncode != 0:
            fail(f"{arm}: identity probe failed: {p.stderr[-1500:]}")
        info = json.loads(p.stdout.strip().splitlines()[-1])
        identity_ok = os.path.realpath(info["una_path"]).startswith(
            os.path.realpath(venv) + os.sep)
        versions = info["versions"]
        pinned_matches = {k: (versions.get(k) == v) for k, v in PINNED.items()}
        if arm == "cand":
            if not (info["cap"] == RULED_CAP_BYTES
                    and info["margin"] == RULED_MARGIN_BYTES):
                fail(f"cand: installed module constants deviate from ruled F5 "
                     f"values: cap={info['cap']} margin={info['margin']}")
            f3_probe_ok = (info["af_imports_lfws"] is True)
        else:
            f3_probe_ok = (info["af_imports_lfws"] is False
                           and info["lfws_find_spec"] is True)
        if not info["nogil"]:
            fail(f"{arm}: nogil probe False — F1 lineage missing (both arms "
                 f"carry A1+F1; the F3 delta is the only arm difference)")
        arm_records[arm] = {
            "commit": a["commit"], "agg_blob": a["agg_blob"],
            "lfws_blob": a["lfws_blob"],
            "archived_agg_blob": archived_agg,
            "wheel": {"path": wheel_path, "sha256": wheel_sha,
                      "bytes": os.path.getsize(wheel_path)},
            "wheel_packaged_agg_blob": pkg_agg,
            "wheel_packaged_lfws_blob": pkg_lfws,
            "venv": venv, "cache_root_designated": CACHE_ROOTS[arm],
            "installed_agg_blob": inst_agg, "installed_lfws_blob": inst_lfws,
            "identity_ok": identity_ok, "versions": versions,
            "pinned_matches": pinned_matches,
            "probe": {"nogil": info["nogil"],
                      "af_imports_lfws": info["af_imports_lfws"],
                      "f3_probe_ok": f3_probe_ok,
                      **({"cap": info["cap"], "margin": info["margin"]}
                         if arm == "cand" else
                         {"lfws_absent": info["lfws_find_spec"]})},
        }
        print(f"[f3r-build] {arm}: wheel {wheel_sha[:12]} agg={pkg_agg[:12]} "
              f"venv_ok={identity_ok} f3_probe_ok={f3_probe_ok}")

    all_ok = all(
        r["identity_ok"] and r["probe"]["f3_probe_ok"] and r["probe"]["nogil"]
        and all(r["pinned_matches"].values())
        for r in arm_records.values())
    rec = {
        "task": "F3R", "record": "arm_wheels_and_venvs",
        "role": "screen-executor",
        "status": "ok" if all_ok else "DEGRADED",
        "arms": arm_records,
        "wt_head_at_build": head_sha,
        "constraints_file": constraints,
        "build_notes": [
            "wheels built from `git archive` of the PINNED arm commits "
            "(b0 c37cf22 line-minus-F3, cand b387f90 line-plus-F3); hatch "
            "bakes commit_date='unknown' (no .git in archive) — code is "
            "byte-identical to the committed trees; W00 rebuilds from the "
            "real tree with git",
            "the ONLY source delta between arms: Engines/AggregateFlow.py "
            "(455e92ad -> 423efe33) + Engines/_large_flow_workspace.py "
            "(absent -> 6c374883); nogil probe expected True on BOTH arms",
            "cache roots are DESIGNATED here; the supervisor creates them "
            "fresh per arm and points NUMBA_CACHE_DIR at them per run",
        ],
        "steps": steps,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(RECORD, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[f3r-build] status={rec['status']} record={RECORD}")
    if rec["status"] != "ok":
        sys.exit(2)


if __name__ == "__main__":
    main()
