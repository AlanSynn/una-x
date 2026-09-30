"""A1R: build the A1 candidate wheel from the committed wt-large-e2e HEAD
(7a27866) into a FRESH venv — installed discipline for the A1 marginal screen
(ahead of W00; does NOT discharge W00's real-wheel guard proof).

Same pipeline as H05's build: verify tree state, `git archive` into a temp copy
(the worktree itself is never written), hatchling build venv, constraints from
the campaign venv freeze, fresh runtime venv, identity probe importing every
pinned library.

Run: campaign-python a1r_build_candidate.py
Writes: campaign_data/a1r_wheel_and_venv.json
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
WT = f"{CAMPAIGN}/wt-large-e2e"
CAND_COMMIT = "7a27866670ae2b7f9ed6a4d61e01d95c18e17f29"
BASE_PY = "/opt/homebrew/opt/python@3.11/bin/python3.11"
CAMPAIGN_PY = f"{CAMPAIGN}/venvs/campaign/bin/python"
BUILD_DIR = f"{CAMPAIGN}/campaign_data/a1r_build"
WHEELS = f"{CAMPAIGN}/campaign_data/wheels_a1r"
BUILD_VENV = f"{CAMPAIGN}/campaign_data/venvs/a1r_build"
RUN_VENV = f"{CAMPAIGN}/campaign_data/venvs/a1r_cand_wheel"
RECORD = f"{CAMPAIGN}/campaign_data/a1r_wheel_and_venv.json"
PINNED = {"python": "3.11.16", "numpy": "2.4.6", "numba": "0.67.0",
          "geopandas": "1.1.4", "pyogrio": "0.13.0", "shapely": "2.1.2"}
TIMEOUT_S = 1800

steps = []


def run(cmd, timeout=TIMEOUT_S, cwd=None):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, cwd=cwd)
    steps.append({"cmd": cmd, "cwd": cwd, "rc": p.returncode,
                  "wall_s": round(time.time() - t0, 1),
                  "stdout_tail": p.stdout[-3000:], "stderr_tail": p.stderr[-3000:]})
    return p


def fail(msg):
    rec = {"task": "A1R", "record": "candidate_wheel_and_venv", "status": "BLOCKED",
           "blocked_reason": msg, "steps": steps,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(RECORD, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[a1r] BLOCKED: {msg}")
    sys.exit(2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1. candidate tree state (STOP condition: screening a moving tree is
    #    prohibited — HEAD must be the named commit and the tree clean)
    p = run(["git", "-C", WT, "rev-parse", "HEAD"])
    if p.returncode != 0 or p.stdout.strip() != CAND_COMMIT:
        fail(f"wt-large-e2e HEAD is not the candidate commit: {p.stdout.strip()!r}")
    # tracked tree must be pristine; untracked campaign-evidence scripts under
    # campaigns/ are tolerated (git archive ships only tracked files at HEAD)
    p = run(["git", "-C", WT, "status", "--porcelain", "--untracked-files=no"])
    if p.returncode != 0 or p.stdout.strip():
        fail(f"wt-large-e2e tracked tree not clean:\n{p.stdout}")
    p = run(["git", "-C", WT, "status", "--porcelain"])
    untracked_note = [l for l in p.stdout.splitlines() if l.startswith("?? ")]

    # 2. archive HEAD into the build copy
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(f"{BUILD_DIR}/src_copy")
    t0 = time.time()
    p = subprocess.run(f"git -C {WT} archive HEAD | tar -x -C {BUILD_DIR}/src_copy",
                       shell=True, capture_output=True, timeout=300)
    steps.append({"cmd": "git archive HEAD | tar -x -C src_copy",
                  "wall_s": round(time.time() - t0, 1), "rc": p.returncode,
                  "stderr_tail": p.stderr.decode("utf-8", errors="replace")[-2000:]})
    if p.returncode != 0:
        fail("archive extraction failed")
    if not os.path.isfile(f"{BUILD_DIR}/src_copy/pyproject.toml"):
        fail("extracted build copy incomplete")

    # 3. build venv + wheel
    if os.path.exists(BUILD_VENV):
        shutil.rmtree(BUILD_VENV)
    p = run([BASE_PY, "-m", "venv", BUILD_VENV])
    if p.returncode != 0:
        fail(f"build venv creation failed: {p.stderr[-500:]}")
    p = run([f"{BUILD_VENV}/bin/pip", "install", "--quiet", "hatchling"])
    if p.returncode != 0:
        fail(f"hatchling install failed: {p.stderr[-800:]}")
    os.makedirs(WHEELS, exist_ok=True)
    p = run([f"{BUILD_VENV}/bin/pip", "wheel", ".", "--no-build-isolation",
             "--no-deps", "-w", WHEELS], cwd=f"{BUILD_DIR}/src_copy", timeout=900)
    if p.returncode != 0:
        fail(f"wheel build failed: {p.stderr[-1500:]}")
    whls = [f for f in os.listdir(WHEELS) if f.endswith(".whl")]
    if len(whls) != 1:
        fail(f"expected exactly one wheel, found {whls}")
    wheel_path = f"{WHEELS}/{whls[0]}"
    wheel_sha = sha256_file(wheel_path)

    # 4. constraints from campaign venv
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

    # 5. fresh runtime venv
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

    # 6. identity probe
    probe = (
        "import json, os, sys, numba, numpy, scipy, pandas, geopandas, shapely, "
        "sklearn, psutil, pyogrio, pyarrow, threadpoolctl\n"
        "import urban_network_analysis as u\n"
        "from urban_network_analysis.Engines import AccessibilityWElevation as AWE\n"
        "from urban_network_analysis.Engines.Accessibility import od_compact_vector_node_view_scope\n"
        "from urban_network_analysis.Engines import _large_access_scratch as S\n"
        "p = os.path.dirname(os.path.abspath(u.__file__))\n"
        "print(json.dumps({'una_path': p, 'site': sys.prefix,\n"
        "    'a1_symbols': {'integrated_scope_access': hasattr(AWE, 'integrated_scope_access'),\n"
        "    'od_compact_vector_node_view_scope': True,\n"
        "    'a1_scope_admits': hasattr(S, '_a1_scope_admits'),\n"
        "    'a1_scope_search': hasattr(S, '_a1_scope_search')},\n"
        "    'versions': {'python': sys.version.split()[0], 'numpy': numpy.__version__,\n"
        "    'numba': numba.__version__, 'geopandas': geopandas.__version__,\n"
        "    'pyogrio': pyogrio.__version__, 'pyarrow': pyarrow.__version__,\n"
        "    'shapely': shapely.__version__}}))\n"
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
    a1_symbols_ok = all(info["a1_symbols"].values())

    rec = {
        "task": "A1R", "record": "candidate_wheel_and_venv", "role": "screen-executor",
        "status": "ok" if (identity_ok and a1_symbols_ok
                           and all(pinned_matches.values())) else "DEGRADED",
        "candidate_commit": CAND_COMMIT,
        "build_copy": f"{BUILD_DIR}/src_copy",
        "wheel": {"path": wheel_path, "sha256": wheel_sha,
                  "bytes": os.path.getsize(wheel_path)},
        "constraints_file": constraints,
        "runtime_venv": RUN_VENV,
        "identity_ok": identity_ok, "a1_symbols": info["a1_symbols"],
        "a1_symbols_ok": a1_symbols_ok,
        "una_installed_path": info["una_path"],
        "versions": versions, "pinned": PINNED, "pinned_matches": pinned_matches,
        "build_notes": ["wheel built from `git archive` of the verified candidate "
                        "HEAD; hatch_build bakes commit_date='unknown' (no .git in "
                        "the archive) — code is byte-identical to the tree; W00 "
                        "rebuilds from the real tree with git"],
        "untracked_files_at_build": untracked_note,
        "steps": steps, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(RECORD, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[a1r] wheel: {wheel_path}\n      sha256 {wheel_sha}\n"
          f"      identity_ok={identity_ok} a1_symbols_ok={a1_symbols_ok} "
          f"pinned={pinned_matches}")
    if rec["status"] != "ok":
        sys.exit(2)


if __name__ == "__main__":
    main()
