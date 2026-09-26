"""H05: build the B0 wheel from the read-only wt-b0 tree into campaign_data and
install it into a FRESH venv (installed-pilots prerequisite; pre-validates
wheel-building for W00 but does NOT discharge W00's real-wheel guard proof).

Steps (every subprocess bounded; every step recorded):
  1. Verify wt-b0 state: HEAD == 361928e4ba38f34622cafe065b0025244db61368,
     clean worktree, src tree == a5883dddcef3afb8debdb4438a6338da886add6f.
  2. `git archive` HEAD into campaign_data/h05_build/src_copy (wt-b0 itself is
     never written; hatch_build.py's _build_info.py lands in the copy only).
  3. Build venv (campaign_data/venvs/h05_build): install hatchling, then
     `pip wheel . --no-build-isolation --no-deps` -> campaign_data/wheels/.
  4. Constraints from the campaign venv `pip freeze` (pinned dep versions).
  5. Fresh runtime venv (campaign_data/venvs/b0_wheel): install the wheel with
     the pinned constraints.
  6. Identity + version assertion in the fresh venv: urban_network_analysis
     must resolve under that venv's site-packages; versions vs pinned table.

Run: campaign-python h05_build_wheel_and_venv.py   (writes campaign_data/h05_wheel_and_venv.json)
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
WT_B0 = f"{CAMPAIGN}/wt-b0"
B0_COMMIT = "361928e4ba38f34622cafe065b0025244db61368"
B0_SRC_TREE = "a5883dddcef3afb8debdb4438a6338da886add6f"
BASE_PY = "/opt/homebrew/opt/python@3.11/bin/python3.11"
CAMPAIGN_PY = f"{CAMPAIGN}/venvs/campaign/bin/python"
BUILD_DIR = f"{CAMPAIGN}/campaign_data/h05_build"
WHEELS = f"{CAMPAIGN}/campaign_data/wheels"
BUILD_VENV = f"{CAMPAIGN}/campaign_data/venvs/h05_build"
RUN_VENV = f"{CAMPAIGN}/campaign_data/venvs/b0_wheel"
RECORD = f"{CAMPAIGN}/campaign_data/h05_wheel_and_venv.json"
PINNED = {"python": "3.11.16", "numpy": "2.4.6", "numba": "0.67.0",
          "geopandas": "1.1.4", "pyogrio": "0.13.0", "shapely": "2.1.2"}
BUILD_NOTES = ["wheel built from `git archive` of the verified HEAD tree; hatch_build "
               "bakes commit_date='unknown' (no .git in the archive) — code is "
               "byte-identical to the tree; W00 rebuilds from the real tree with git"]
TIMEOUT_S = 1800

steps = []


def run(cmd, timeout=TIMEOUT_S, cwd=None, env=None):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, cwd=cwd, env=env)
    steps.append({"cmd": cmd, "cwd": cwd, "rc": p.returncode,
                  "wall_s": round(time.time() - t0, 1),
                  "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-4000:]})
    return p


def fail(msg):
    rec = {"task": "H05", "record": "wheel_and_venv", "status": "BLOCKED",
           "blocked_reason": msg, "steps": steps,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(RECORD, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[h05] BLOCKED: {msg}")
    sys.exit(2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1. wt-b0 state ---------------------------------------------------------
    p = run(["git", "-C", WT_B0, "rev-parse", "HEAD"])
    if p.returncode != 0 or p.stdout.strip() != B0_COMMIT:
        fail(f"wt-b0 HEAD mismatch: {p.stdout.strip()!r}")
    p = run(["git", "-C", WT_B0, "status", "--porcelain"])
    if p.returncode != 0 or p.stdout.strip():
        fail(f"wt-b0 worktree not clean:\n{p.stdout}")
    p = run(["git", "-C", WT_B0, "rev-parse", "HEAD:src"])
    if p.returncode != 0 or p.stdout.strip() != B0_SRC_TREE:
        fail(f"wt-b0 src tree mismatch: {p.stdout.strip()!r} != {B0_SRC_TREE}")

    # 2. git archive into build copy (binary stream — never text-captured) ----
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(f"{BUILD_DIR}/src_copy")
    t0 = time.time()
    p = subprocess.run(f"git -C {WT_B0} archive HEAD | tar -x -C {BUILD_DIR}/src_copy",
                       shell=True, capture_output=True, timeout=300)
    steps.append({"cmd": "git archive HEAD | tar -x -C src_copy",
                  "wall_s": round(time.time() - t0, 1), "rc": p.returncode,
                  "stderr_tail": p.stderr.decode("utf-8", errors="replace")[-2000:]})
    if p.returncode != 0:
        fail("archive extraction failed")
    # spot-verify the extracted copy: tracked tree identity is inherited from
    # the verified HEAD; check a sentinel file exists and pyproject matches HEAD
    if not os.path.isfile(f"{BUILD_DIR}/src_copy/pyproject.toml") or \
            not os.path.isdir(f"{BUILD_DIR}/src_copy/src/urban_network_analysis"):
        fail("extracted build copy incomplete")

    # 3. build venv + wheel ----------------------------------------------------
    if os.path.exists(BUILD_VENV):
        shutil.rmtree(BUILD_VENV)
    p = run([BASE_PY, "-m", "venv", BUILD_VENV])
    if p.returncode != 0:
        fail(f"build venv creation failed: {p.stderr[-500:]}")
    p = run([f"{BUILD_VENV}/bin/pip", "install", "--quiet", "hatchling"])
    if p.returncode != 0:
        fail(f"hatchling install failed (network?): {p.stderr[-800:]}")
    p = run([f"{BUILD_VENV}/bin/pip", "show", "hatchling"])
    hatch_v = next((l.split(": ")[1].strip() for l in p.stdout.splitlines()
                    if l.startswith("Version")), "unknown")
    os.makedirs(WHEELS, exist_ok=True)
    p = run([f"{BUILD_VENV}/bin/pip", "wheel", ".", "--no-build-isolation",
             "--no-deps", "-w", WHEELS], cwd=f"{BUILD_DIR}/src_copy", timeout=900)
    if p.returncode != 0:
        fail(f"wheel build failed: {p.stderr[-1500:]}")
    whls = [f for f in os.listdir(WHEELS) if f.endswith(".whl")]
    if len(whls) != 1:
        fail(f"expected exactly one wheel in {WHEELS}, found {whls}")
    wheel_path = f"{WHEELS}/{whls[0]}"
    wheel_sha = sha256_file(wheel_path)

    # 4. constraints from campaign venv ---------------------------------------
    p = run([CAMPAIGN_PY, "-m", "pip", "freeze"])
    if p.returncode != 0:
        fail("pip freeze on campaign venv failed")
    freeze_lines = [l.strip() for l in p.stdout.splitlines() if l.strip()
                    and not l.startswith("-e ") and "urban-network-analysis" not in l
                    and "urban_network_analysis" not in l]
    constraints = f"{BUILD_DIR}/constraints.txt"
    with open(constraints, "w") as f:
        f.write("\n".join(freeze_lines) + "\n")

    # 5. fresh runtime venv + wheel install ------------------------------------
    if os.path.exists(RUN_VENV):
        shutil.rmtree(RUN_VENV)
    p = run([BASE_PY, "-m", "venv", RUN_VENV])
    if p.returncode != 0:
        fail(f"runtime venv creation failed: {p.stderr[-500:]}")
    p = run([f"{RUN_VENV}/bin/pip", "install", "--quiet", "--upgrade", "pip"],
            timeout=600)
    # pyarrow + pyogrio are required RUNTIME deps for the frozen output formats
    # (output_feather=true, GeoJSON engine) but are only optional extras of
    # geopandas/pandas — install them explicitly, pinned by the same constraints.
    p = run([f"{RUN_VENV}/bin/pip", "install", "--quiet", "-c", constraints,
             wheel_path, "pyarrow", "pyogrio", "threadpoolctl"],
            timeout=TIMEOUT_S)
    if p.returncode != 0:
        fail(f"wheel install into fresh venv failed: {p.stderr[-1500:]}")
    p = run([f"{RUN_VENV}/bin/pip", "freeze"])
    run_venv_freeze = p.stdout.splitlines()

    # 6. identity + versions in the fresh venv ---------------------------------
    probe = (
        "import json, os, sys, numba, numpy, scipy, pandas, geopandas, shapely, "
        "sklearn, psutil, pyogrio, pyarrow, threadpoolctl\n"
        "import urban_network_analysis as u\n"
        "p = os.path.dirname(os.path.abspath(u.__file__))\n"
        "print(json.dumps({'una_path': p, 'site': sys.prefix,\n"
        "    'versions': {'python': sys.version.split()[0], 'numpy': numpy.__version__,\n"
        "    'numba': numba.__version__, 'scipy': scipy.__version__,\n"
        "    'pandas': pandas.__version__, 'geopandas': geopandas.__version__,\n"
        "    'shapely': shapely.__version__, 'sklearn': sklearn.__version__,\n"
        "    'psutil': psutil.__version__, 'pyogrio': pyogrio.__version__,\n"
        "    'pyarrow': pyarrow.__version__, 'threadpoolctl': threadpoolctl.__version__},\n"
        "    'about': getattr(u, 'about', lambda: None)() }))\n"
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

    rec = {
        "task": "H05", "record": "wheel_and_venv", "role": "performance-owner",
        "status": "ok" if (identity_ok and all(pinned_matches.values())) else "DEGRADED",
        "b0_commit": B0_COMMIT, "b0_src_tree": B0_SRC_TREE,
        "build_copy": f"{BUILD_DIR}/src_copy",
        "hatchling_version": hatch_v,
        "wheel": {"path": wheel_path, "sha256": wheel_sha,
                  "bytes": os.path.getsize(wheel_path)},
        "constraints_file": constraints,
        "runtime_venv": RUN_VENV, "runtime_venv_python": versions["python"],
        "identity_ok": identity_ok, "una_installed_path": info["una_path"],
        "versions": versions, "pinned": PINNED, "pinned_matches": pinned_matches,
        "campaign_venv_freeze_lines": freeze_lines,
        "runtime_venv_freeze": run_venv_freeze,
        "about": info.get("about"),
        "w00_note": "pre-validates wheel-building for W00; does NOT discharge W00's "
                    "real-wheel guard proof (H04-N2)",
        "build_notes": BUILD_NOTES,
        "steps": steps, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(RECORD, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[h05] wheel: {wheel_path}\n      sha256 {wheel_sha}\n"
          f"      identity_ok={identity_ok} pinned={pinned_matches}")
    if rec["status"] != "ok":
        sys.exit(2)


if __name__ == "__main__":
    main()
