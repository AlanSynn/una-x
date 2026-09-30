"""F1R: build the F1 candidate wheel from the PINNED candidate commit
(c1d7edc, AggregateFlow.py blob 455e92a) into a FRESH venv — installed
discipline for the F1 marginal screen (a1r_build_candidate.py recipe; spec
rev 4 cleared this build before the P4b driver verification).

Additions vs the A1R recipe (spec arms_and_sources):
  - archive of the PINNED COMMIT, not HEAD (HEAD has moved on with spec
    amendments; the candidate content is c1d7edc regardless)
  - packaged-blob verification: AggregateFlow.py extracted from the built
    candidate wheel must hash (git hash-object) to 455e92adb5f23748903435d3d
    6340a5d8a9c47d3; the b0 wheel (71fd5615...) must yield 0114ce8f318a3247
    b62e3d468e32d54e692737a2; the installed files in BOTH arm venvs are
    re-hashed the same way
  - source-level decorator probe: candidate venv's _accumulate_od_flow source
    carries nogil=True; b0 venv's does not
  - b0 arm identity: the existing H05 b0_wheel venv + wheel re-asserted
    (sha256 + blob); F1R cache roots are fresh per-arm paths regardless

Run: campaign-python f1r_build_candidate.py   (heavy: counts against F1R lease)
Writes: campaign_data/f1r_wheel_and_venv.json
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
CAND_COMMIT = "c1d7edc519f254c0e81475509d34b437e33abf09"
CAND_AGG_BLOB = "455e92adb5f23748903435d3d6340a5d8a9c47d3"
B0_AGG_BLOB = "0114ce8f318a3247b62e3d468e32d54e692737a2"
AGG_REL = "src/urban_network_analysis/Engines/AggregateFlow.py"
B0_WHEEL = f"{CAMPAIGN}/campaign_data/wheels/urban_network_analysis-2.6.0-py3-none-any.whl"
B0_WHEEL_SHA = "71fd5615b5683f3c798938a2762824da9ba3fada801aaa840343e7378d4fc979"
B0_VENV = f"{CAMPAIGN}/campaign_data/venvs/b0_wheel"
BASE_PY = "/opt/homebrew/opt/python@3.11/bin/python3.11"
CAMPAIGN_PY = f"{CAMPAIGN}/venvs/campaign/bin/python"
BUILD_DIR = f"{CAMPAIGN}/campaign_data/f1r_build"
WHEELS = f"{CAMPAIGN}/campaign_data/wheels_f1r"
BUILD_VENV = f"{CAMPAIGN}/campaign_data/venvs/f1r_build"
RUN_VENV = f"{CAMPAIGN}/campaign_data/venvs/f1r_cand_wheel"
RECORD = f"{CAMPAIGN}/campaign_data/f1r_wheel_and_venv.json"
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
    rec = {"task": "F1R", "record": "candidate_wheel_and_venv", "status": "BLOCKED",
           "blocked_reason": msg, "steps": steps,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(RECORD, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[f1r-build] BLOCKED: {msg}")
    sys.exit(2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_of(path):
    """Git blob hash of a working-tree file (equivalent to `git hash-object`)."""
    p = subprocess.run(["git", "-C", WT, "hash-object", path],
                       capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        return None
    return p.stdout.strip()


def wheel_agg_blob(wheel_path, extract_dir):
    """Extract Engines/AggregateFlow.py from a wheel and return its blob hash."""
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)
    with zipfile.ZipFile(wheel_path) as z:
        names = [n for n in z.namelist()
                 if n.endswith("Engines/AggregateFlow.py")]
        if len(names) != 1:
            fail(f"wheel {wheel_path}: expected 1 AggregateFlow.py, found {names}")
        z.extract(names[0], extract_dir)
    return git_blob_of(os.path.join(extract_dir, names[0]))


def main():
    # 1. pin checks on the candidate commit (independent of HEAD movement)
    p = run(["git", "-C", WT, "rev-parse", f"{CAND_COMMIT}:" + AGG_REL])
    if p.returncode != 0 or p.stdout.strip() != CAND_AGG_BLOB:
        fail(f"candidate commit blob pin failed: {p.stdout.strip()!r} "
             f"!= {CAND_AGG_BLOB}")
    p = run(["git", "-C", WT, "merge-base", "--is-ancestor",
             CAND_COMMIT, "HEAD"])
    if p.returncode != 0:
        fail("candidate commit is not an ancestor of wt-large-e2e HEAD")
    p = run(["git", "-C", WT, "rev-parse", "HEAD"])
    head_sha = p.stdout.strip()
    # tracked tree must be clean OUTSIDE campaigns/ (the archive is of the
    # pinned commit, immune to worktree state; the expected campaigns/ dirty
    # entry is the pending spec rev 4 awaiting coordinator commit)
    p = run(["git", "-C", WT, "status", "--porcelain", "--untracked-files=no"])
    dirty = [l for l in p.stdout.splitlines() if l.strip()]
    dirty_outside = [l for l in dirty if not l.startswith(" M campaigns/")
                     and not l.startswith("M campaigns/")]
    if p.returncode != 0 or dirty_outside:
        fail(f"wt-large-e2e tracked tree not clean outside campaigns/:\n"
             f"{p.stdout}")
    dirty_campaigns = [l for l in dirty if l not in dirty_outside]
    p = run(["git", "-C", WT, "status", "--porcelain"])
    untracked_note = [l for l in p.stdout.splitlines() if l.startswith("?? ")]

    # 2. b0 arm identity re-assert (wheel file + installed tree)
    b0_wheel_sha = sha256_file(B0_WHEEL)
    if b0_wheel_sha != B0_WHEEL_SHA:
        fail(f"b0 wheel sha {b0_wheel_sha} != pinned {B0_WHEEL_SHA}")
    b0_wheel_blob = wheel_agg_blob(B0_WHEEL, f"{BUILD_DIR}/b0_wheel_extract")
    if b0_wheel_blob != B0_AGG_BLOB:
        fail(f"b0 wheel packaged blob {b0_wheel_blob} != {B0_AGG_BLOB}")
    b0_installed = (f"{B0_VENV}/lib/python3.11/site-packages/"
                    "urban_network_analysis/Engines/AggregateFlow.py")
    if not os.path.isfile(b0_installed):
        fail(f"b0 venv installed AggregateFlow.py missing: {b0_installed}")
    b0_installed_blob = git_blob_of(b0_installed)
    if b0_installed_blob != B0_AGG_BLOB:
        fail(f"b0 venv installed blob {b0_installed_blob} != {B0_AGG_BLOB}")

    # 3. archive the pinned candidate commit into the build copy
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(f"{BUILD_DIR}/src_copy")
    t0 = time.time()
    p = subprocess.run(
        f"git -C {WT} archive {CAND_COMMIT} | tar -x -C {BUILD_DIR}/src_copy",
        shell=True, capture_output=True, timeout=300)
    steps.append({"cmd": f"git archive {CAND_COMMIT[:7]} | tar -x -C src_copy",
                  "wall_s": round(time.time() - t0, 1), "rc": p.returncode,
                  "stderr_tail": p.stderr.decode("utf-8", errors="replace")[-2000:]})
    if p.returncode != 0:
        fail("archive extraction failed")
    if not os.path.isfile(f"{BUILD_DIR}/src_copy/pyproject.toml"):
        fail("extracted build copy incomplete")
    archived_blob = git_blob_of(f"{BUILD_DIR}/src_copy/" + AGG_REL)
    if archived_blob != CAND_AGG_BLOB:
        fail(f"archived AggregateFlow.py blob {archived_blob} != {CAND_AGG_BLOB}")

    # 4. build venv + wheel
    if os.path.exists(BUILD_VENV):
        shutil.rmtree(BUILD_VENV)
    p = run([BASE_PY, "-m", "venv", BUILD_VENV])
    if p.returncode != 0:
        fail(f"build venv creation failed: {p.stderr[-500:]}")
    p = run([f"{BUILD_VENV}/bin/pip", "install", "--quiet", "hatchling"])
    if p.returncode != 0:
        fail(f"hatchling install failed: {p.stderr[-800:]}")
    os.makedirs(WHEELS, exist_ok=True)
    for f in os.listdir(WHEELS):
        os.remove(os.path.join(WHEELS, f))
    p = run([f"{BUILD_VENV}/bin/pip", "wheel", ".", "--no-build-isolation",
             "--no-deps", "-w", WHEELS], cwd=f"{BUILD_DIR}/src_copy", timeout=900)
    if p.returncode != 0:
        fail(f"wheel build failed: {p.stderr[-1500:]}")
    whls = [f for f in os.listdir(WHEELS) if f.endswith(".whl")]
    if len(whls) != 1:
        fail(f"expected exactly one wheel, found {whls}")
    wheel_path = f"{WHEELS}/{whls[0]}"
    wheel_sha = sha256_file(wheel_path)
    cand_wheel_blob = wheel_agg_blob(wheel_path, f"{BUILD_DIR}/cand_wheel_extract")
    if cand_wheel_blob != CAND_AGG_BLOB:
        fail(f"candidate wheel packaged blob {cand_wheel_blob} != {CAND_AGG_BLOB}")

    # 5. constraints from campaign venv
    p = run([CAMPAIGN_PY, "-m", "pip", "freeze"])
    if p.returncode != 0:
        fail("pip freeze on campaign venv failed")
    freeze_lines = [l.strip() for l in p.stdout.splitlines() if l.strip()
                    and not l.startswith("-e ")
                    and "urban-network-analysis" not in l
                    and "urban_network_analysis" not in l]
    constraints = f"{BUILD_DIR}/constraints.txt"
    with open(constraints, "w") as f:
        f.write("\n".join(freeze_lines) + "\n")

    # 6. fresh candidate runtime venv
    if os.path.exists(RUN_VENV):
        shutil.rmtree(RUN_VENV)
    p = run([BASE_PY, "-m", "venv", RUN_VENV])
    if p.returncode != 0:
        fail(f"runtime venv creation failed: {p.stderr[-500:]}")
    p = run([f"{RUN_VENV}/bin/pip", "install", "--quiet", "--upgrade", "pip"],
            timeout=600)
    p = run([f"{RUN_VENV}/bin/pip", "install", "--quiet", "-c", constraints,
             wheel_path, "pyarrow", "pyogrio", "threadpoolctl"], timeout=TIMEOUT_S)
    if p.returncode != 0:
        fail(f"candidate wheel install failed: {p.stderr[-1500:]}")

    # 7. identity probe (candidate venv) + decorator source check
    probe = (
        "import json, os, sys, inspect, numba, numpy, scipy, pandas, geopandas, "
        "shapely, sklearn, psutil, pyogrio, pyarrow, threadpoolctl\n"
        "import urban_network_analysis as u\n"
        "from urban_network_analysis.Engines import AggregateFlow as AF\n"
        "src = inspect.getsource(AF._accumulate_od_flow)\n"
        "p = os.path.dirname(os.path.abspath(u.__file__))\n"
        "print(json.dumps({'una_path': p, 'site': sys.prefix,\n"
        "    'has_accumulate_od_flow': hasattr(AF, '_accumulate_od_flow'),\n"
        "    'decorator_nogil': 'nogil=True' in src.split('def ')[0],\n"
        "    'versions': {'python': sys.version.split()[0], "
        "'numpy': numpy.__version__, 'numba': numba.__version__,\n"
        "    'geopandas': geopandas.__version__, 'pyogrio': pyogrio.__version__,\n"
        "    'pyarrow': pyarrow.__version__, 'shapely': shapely.__version__}}))\n"
    )
    try:
        p = subprocess.run([f"{RUN_VENV}/bin/python", "-c", probe],
                           capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        fail("identity probe timed out")
    if p.returncode != 0:
        fail(f"identity probe failed: {p.stderr[-1500:]}")
    info = json.loads(p.stdout.strip().splitlines()[-1])
    site_real = os.path.realpath(RUN_VENV)
    identity_ok = os.path.realpath(info["una_path"]).startswith(site_real + os.sep)
    versions = info["versions"]
    pinned_matches = {k: (versions.get(k) == v) for k, v in PINNED.items()}

    # 8. candidate venv installed blob + b0/cand decorator differentiation
    cand_installed = (f"{RUN_VENV}/lib/python3.11/site-packages/"
                      "urban_network_analysis/Engines/AggregateFlow.py")
    cand_installed_blob = git_blob_of(cand_installed)
    if cand_installed_blob != CAND_AGG_BLOB:
        fail(f"candidate venv installed blob {cand_installed_blob} != "
             f"{CAND_AGG_BLOB}")
    p = run([f"{B0_VENV}/bin/python", "-c",
             "import inspect\n"
             "from urban_network_analysis.Engines import AggregateFlow as AF\n"
             "src = inspect.getsource(AF._accumulate_od_flow)\n"
             "print('decorator_nogil=', 'nogil=True' in src.split('def ')[0])"])
    b0_nogil = "decorator_nogil= True" in p.stdout
    if b0_nogil:
        fail("b0 venv _accumulate_od_flow unexpectedly carries nogil=True")

    rec = {
        "task": "F1R", "record": "candidate_wheel_and_venv",
        "role": "screen-executor",
        "status": "ok" if (identity_ok and info["has_accumulate_od_flow"]
                           and info["decorator_nogil"]
                           and all(pinned_matches.values())) else "DEGRADED",
        "candidate_commit": CAND_COMMIT,
        "wt_head_at_build": head_sha,
        "blob_pins": {
            "aggregate_flow_rel": AGG_REL,
            "candidate_commit_blob": CAND_AGG_BLOB,
            "b0_blob": B0_AGG_BLOB,
            "candidate_commit_blob_verified": True,
            "archived_src_blob": archived_blob,
            "cand_wheel_packaged_blob": cand_wheel_blob,
            "cand_venv_installed_blob": cand_installed_blob,
            "b0_wheel_packaged_blob": b0_wheel_blob,
            "b0_venv_installed_blob": b0_installed_blob},
        "b0_arm": {"wheel_path": B0_WHEEL, "wheel_sha256": b0_wheel_sha,
                   "venv": B0_VENV,
                   "decorator_nogil_expected_false": not b0_nogil},
        "build_copy": f"{BUILD_DIR}/src_copy",
        "wheel": {"path": wheel_path, "sha256": wheel_sha,
                  "bytes": os.path.getsize(wheel_path)},
        "constraints_file": constraints,
        "runtime_venv": RUN_VENV,
        "identity_ok": identity_ok,
        "f1_probe": {"has_accumulate_od_flow": info["has_accumulate_od_flow"],
                     "decorator_nogil_cand": info["decorator_nogil"],
                     "decorator_nogil_b0": b0_nogil},
        "una_installed_path": info["una_path"],
        "versions": versions, "pinned": PINNED, "pinned_matches": pinned_matches,
        "build_notes": ["wheel built from `git archive` of the PINNED candidate "
                        "commit c1d7edc (HEAD has moved on with spec amendments; "
                        "candidate content is the pinned commit); hatch_build "
                        "bakes commit_date='unknown' (no .git in the archive) — "
                        "code is byte-identical to the committed tree; W00 "
                        "rebuilds from the real tree with git"],
        "untracked_files_at_build": untracked_note,
        "dirty_tracked_campaigns_files": {
            "files": dirty_campaigns,
            "expected_reason": "screen_spec.json rev 4 (P4b vehicle correction) "
                               "authored by reviewer, awaiting coordinator "
                               "commit; git archive of the pinned commit is "
                               "immune to worktree state and the AggregateFlow "
                               "blob pin was verified directly from the commit"},
        "steps": steps, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(RECORD, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[f1r-build] wheel: {wheel_path}\n      sha256 {wheel_sha}\n"
          f"      blobs: cand_pkg={cand_wheel_blob[:12]} "
          f"cand_installed={cand_installed_blob[:12]} "
          f"b0_pkg={b0_wheel_blob[:12]} b0_installed={b0_installed_blob[:12]}\n"
          f"      identity_ok={identity_ok} nogil_cand={info['decorator_nogil']} "
          f"nogil_b0={b0_nogil} pinned={pinned_matches}")
    if rec["status"] != "ok":
        sys.exit(2)


if __name__ == "__main__":
    main()
