"""A3R: build BOTH screen arms — b0 wheel from pinned commit
9b1340e and cand wheel from pinned commit f17184b — each into a FRESH venv
(arm-identity packaging precondition for the A3 marginal screen).

Clone of a1r_build_candidate.py; clone-rot statement:
  KEPT (A1R spine): steps journal with per-step rc/wall/tails; fail() writes a
    BLOCKED record; sha256_file; tracked-clean tree gate with untracked
    evidence tolerated (git archive ships only tracked files); `git archive`
    into a build copy (the worktree is never written); shared hatchling build
    venv; constraints pip-frozen from the campaign venv; fresh runtime venv
    per arm; identity probe importing the pinned libraries; hatch_build
    bakes commit_date='unknown' (no .git in the archive) — code is
    byte-identical to the archived tree.
  CHANGED (1) archives the PINNED COMMIT refs, never HEAD (the supervisor
    refuses a wheel record whose git_archive_ref != pinned commit — A1R's
    HEAD==commit guard is replaced by content-pinned archiving; HEAD and
    porcelain are recorded descriptively, not gated); (2) two arms in one
    invocation, one shared build venv, per-arm wheel dirs, per-arm records
    a3r_wheel_and_venv_<arm>.json carrying git_archive_ref; (3) identity
    probe asserts the A1 symbol family PRESENT in both arms and the A3
    symbol family PRESENT in cand / ABSENT in b0 (arm-identity pins);
    (4) installed Engines file shas are checked against shas derived from
    THIS invocation's archive (no duplicated pin constants — cross-check
    against a3r_bare_job.py ARM_MODULE_PINS is a review-time comparison);
    (5) probe runs with NUMBA_CACHE_DIR pinned to a throwaway root +
    PYTHONDONTWRITEBYTECODE=1, site-packages is censused for .nbc/.nbi
    afterward (expect 0).
  ADDED: (i) arm-identity re-derivation gate — `git diff --name-status
    9b1340e..f17184b` must classify to exactly 3 M engine paths + 13 A under
    tests/large_e2e/A3/ + 7 A under campaigns/una_large_e2e/evidence/A3I/
    (re-derives the discharged D6 commit-level diff inside the build record);
    (ii) item 12(a) full site-packages diff excluding __pycache__ — every
    differing path must be under urban_network_analysis*; (iii) item 12(d)
    wheel content compare — the differing .py members must be exactly the
    three engine modules; (iv) item 12(b) control-arm re-pin declaration
    (b0 wheel built fresh here, not inherited from A1R).

Run: campaign-python a3r_build_candidate.py
Writes: campaign_data/a3r_wheel_and_venv_b0.json,
        campaign_data/a3r_wheel_and_venv_cand.json
Writes nothing under the worktree; never touches the per-arm numba cache
roots (nbc_a3r_b0 / nbc_a3r_cand — owned by the screen supervisor).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
WT = f"{CAMPAIGN}/wt-large-e2e"
DATA = f"{CAMPAIGN}/campaign_data"
BASE_PY = "/opt/homebrew/opt/python@3.11/bin/python3.11"
CAMPAIGN_PY = f"{CAMPAIGN}/venvs/campaign/bin/python"
BUILD_DIR = f"{DATA}/a3r_build"
BUILD_VENV = f"{DATA}/venvs/a3r_build"
CONSTRAINTS = f"{BUILD_DIR}/constraints.txt"
B0_COMMIT = "9b1340eb008accdff45fbdf57c6009dd5a6b5146"
CAND_COMMIT = "f17184bf931ac5012bc3b85e91588502937906a7"
ARMS = {
    "b0": {"commit": B0_COMMIT,
           "wheels": f"{DATA}/wheels_a3r/b0",
           "build_copy": f"{BUILD_DIR}/b0_src",
           "run_venv": f"{DATA}/venvs/a3r_b0_wheel",
           "record": f"{DATA}/a3r_wheel_and_venv_b0.json"},
    "cand": {"commit": CAND_COMMIT,
             "wheels": f"{DATA}/wheels_a3r/cand",
             "build_copy": f"{BUILD_DIR}/cand_src",
             "run_venv": f"{DATA}/venvs/a3r_cand_wheel",
             "record": f"{DATA}/a3r_wheel_and_venv_cand.json"},
}
PINNED = {"python": "3.11.16", "numpy": "2.4.6", "numba": "0.67.0",
          "geopandas": "1.1.4", "pyogrio": "0.13.0", "shapely": "2.1.2"}
ENGINES = ("_large_access_scratch.py", "Accessibility.py",
           "AccessibilityWElevation.py")
ENGINE_PKG_PREFIX = "urban_network_analysis/Engines/"
A1_SYMBOLS = ("_a1_scope_admits", "_a1_scope_search")
A3_SYMBOLS = ("_a3_tail_admits", "_a3_scope_search_tailless")
# expected hasattr truth per arm: A1 family present everywhere (control
# surface), A3 family only in the candidate (arm-identity pins)
EXPECT_SYMBOLS = {
    "b0": {**{s: True for s in A1_SYMBOLS}, **{s: False for s in A3_SYMBOLS}},
    "cand": {s: True for s in A1_SYMBOLS + A3_SYMBOLS},
}
# arm-identity re-derivation gate: the full-tree diff between the two pinned
# commits classifies to exactly these (fresh re-derivation of the D6
# commit-level discharge, asserted inside the build record)
EXPECTED_M_PATHS = (
    "src/urban_network_analysis/Engines/Accessibility.py",
    "src/urban_network_analysis/Engines/AccessibilityWElevation.py",
    "src/urban_network_analysis/Engines/_large_access_scratch.py",
)
EXPECTED_A_PREFIXES = (("tests/large_e2e/A3/", 13),
                       ("campaigns/una_large_e2e/evidence/A3I/", 7))
TIMEOUT_S = 1800

steps = []
arm_records = {}


def run(cmd, timeout=TIMEOUT_S, cwd=None):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, cwd=cwd)
    steps.append({"cmd": cmd, "cwd": cwd, "rc": p.returncode,
                  "wall_s": round(time.time() - t0, 1),
                  "stdout_tail": p.stdout[-3000:], "stderr_tail": p.stderr[-3000:]})
    return p


def fail(msg, arm=None):
    rec = {"task": "A3R", "record": "arms_wheels_and_venvs",
           "status": "BLOCKED", "blocked_reason": msg,
           "blocked_arm": arm, "steps": steps,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if arm is not None:
        target = ARMS[arm]["record"]
    else:
        target = f"{DATA}/a3r_wheel_and_venv_BLOCKED.json"
    with open(target, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[a3r-build] BLOCKED: {msg}")
    sys.exit(2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def site_packages_map(root):
    """rel-path -> sha256 for every file under a venv's site-packages,
    excluding __pycache__ directories and .pyc files (item 12(a) frame)."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(".pyc"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            out[rel] = sha256_file(full)
    return out


def wheel_member_map(wheel_path):
    """member name -> sha256 of member bytes, for the item 12(d) compare."""
    out = {}
    with zipfile.ZipFile(wheel_path) as z:
        for name in z.namelist():
            out[name] = hashlib.sha256(z.read(name)).hexdigest()
    return out


def nbc_census(site):
    """count numba on-disk cache artifacts under a venv's site-packages
    (expect 0 — the probe compiled nothing; NUMBA_CACHE_DIR was pinned)."""
    n = 0
    hits = []
    for dirpath, dirnames, filenames in os.walk(site):
        for fn in filenames:
            if fn.endswith((".nbc", ".nbi")):
                n += 1
                hits.append(os.path.relpath(os.path.join(dirpath, fn), site))
    return n, hits[:20]


def diff_precondition():
    """re-derive the arm-identity commit diff; hard gate (ADDED item i)."""
    p = run(["git", "-C", WT, "diff", "--name-status",
             f"{B0_COMMIT}..{CAND_COMMIT}"])
    if p.returncode != 0:
        fail(f"git diff between pinned commits failed: {p.stderr[-500:]}")
    m_paths, a_paths, other = [], [], []
    for line in p.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            fail(f"unparseable name-status line: {line!r}")
        (m_paths if parts[0] == "M" else
         a_paths if parts[0] == "A" else other).append(parts[1])
    checks = {
        "m_paths_exact": sorted(m_paths) == sorted(EXPECTED_M_PATHS),
        "a_prefix_counts": {pref: sum(1 for q in a_paths
                                      if q.startswith(pref))
                            for pref, _ in EXPECTED_A_PREFIXES},
        "a_paths_all_known": all(any(q.startswith(pref)
                                     for pref, _ in EXPECTED_A_PREFIXES)
                                 for q in a_paths),
        "no_other_status": other == [],
        "total": len(m_paths) + len(a_paths) + len(other),
    }
    ok = (checks["m_paths_exact"] and checks["a_paths_all_known"]
          and checks["no_other_status"] and checks["total"] == 23
          and all(checks["a_prefix_counts"][pref] == n
                  for pref, n in EXPECTED_A_PREFIXES))
    return {"ok": ok, "checks": checks,
            "expected": {"m_paths": list(EXPECTED_M_PATHS),
                         "a_prefixes": {pref: n for pref, n
                                        in EXPECTED_A_PREFIXES},
                         "total": 23}}


def probe_env(probe_root):
    e = dict(os.environ)
    e["NUMBA_CACHE_DIR"] = probe_root
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    return e


def build_arm(arm):
    a = ARMS[arm]
    ref = a["commit"]

    # archive the PINNED ref, never HEAD (CHANGED 1)
    if os.path.exists(a["build_copy"]):
        shutil.rmtree(a["build_copy"])
    os.makedirs(a["build_copy"])
    t0 = time.time()
    p = subprocess.run(f"git -C {WT} archive {ref} | tar -x -C {a['build_copy']}",
                       shell=True, capture_output=True, timeout=300)
    steps.append({"cmd": f"git archive {ref[:12]} | tar -x -C {a['build_copy']}",
                  "wall_s": round(time.time() - t0, 1), "rc": p.returncode,
                  "stderr_tail": p.stderr.decode("utf-8",
                                                 errors="replace")[-2000:]})
    if p.returncode != 0:
        fail(f"{arm}: archive extraction failed", arm)
    if not os.path.isfile(f"{a['build_copy']}/pyproject.toml"):
        fail(f"{arm}: extracted build copy incomplete", arm)

    # shas of the archived engine modules — the pins THIS invocation derives
    # (CHANGED 4: no duplicated pin constants; compare against
    # a3r_bare_job.py ARM_MODULE_PINS at review time)
    archived_shas = {}
    for fname in ENGINES:
        fpath = (f"{a['build_copy']}/src/urban_network_analysis/"
                 f"Engines/{fname}")
        if not os.path.isfile(fpath):
            fail(f"{arm}: archived engine module missing: {fname}", arm)
        archived_shas[fname] = sha256_file(fpath)

    # wheel
    if os.path.exists(a["wheels"]):
        shutil.rmtree(a["wheels"])
    os.makedirs(a["wheels"])
    p = run([f"{BUILD_VENV}/bin/pip", "wheel", ".", "--no-build-isolation",
             "--no-deps", "-w", a["wheels"]], cwd=a["build_copy"], timeout=900)
    if p.returncode != 0:
        fail(f"{arm}: wheel build failed: {p.stderr[-1500:]}", arm)
    whls = [f for f in os.listdir(a["wheels"]) if f.endswith(".whl")]
    if len(whls) != 1:
        fail(f"{arm}: expected exactly one wheel, found {whls}", arm)
    wheel_path = f"{a['wheels']}/{whls[0]}"
    wheel_sha = sha256_file(wheel_path)

    # fresh runtime venv
    if os.path.exists(a["run_venv"]):
        shutil.rmtree(a["run_venv"])
    p = run([BASE_PY, "-m", "venv", a["run_venv"]])
    if p.returncode != 0:
        fail(f"{arm}: runtime venv creation failed: {p.stderr[-500:]}", arm)
    # [A3R-B2 fold, h04] constrained pip upgrade — an unconstrained upgrade
    # is a spurious-DEGRADED vector (mid-build PyPI pip release would diverge
    # the two arms' site-packages under the item 12(a) diff)
    p = run([f"{a['run_venv']}/bin/pip", "install", "--quiet",
             "-c", CONSTRAINTS, "--upgrade", "pip"], timeout=600)
    p = run([f"{a['run_venv']}/bin/pip", "install", "--quiet", "-c",
             CONSTRAINTS, wheel_path, "pyarrow", "pyogrio", "threadpoolctl"],
            timeout=TIMEOUT_S)
    if p.returncode != 0:
        fail(f"{arm}: wheel install failed: {p.stderr[-1500:]}", arm)

    # identity probe: symbols + versions + installed location, with the
    # numba cache channel pinned to a throwaway root (CHANGED 5)
    probe_root = tempfile.mkdtemp(prefix="a3r_build_probe_nbc_")
    expected_symbols = EXPECT_SYMBOLS[arm]
    probe = (
        "import json, os, sys, numba, numpy, scipy, pandas, geopandas, "
        "shapely, sklearn, psutil, pyogrio, pyarrow, threadpoolctl\n"
        "import urban_network_analysis as u\n"
        "from urban_network_analysis.Engines import _large_access_scratch as S\n"
        "from urban_network_analysis.Engines import AccessibilityWElevation\n"
        "p = os.path.dirname(os.path.abspath(u.__file__))\n"
        "sym = " + repr(sorted(expected_symbols)) + "\n"
        "print(json.dumps({'una_path': p, 'site': sys.prefix,\n"
        "    'symbols': {s: hasattr(S, s) for s in sym},\n"
        "    'versions': {'python': sys.version.split()[0],\n"
        "    'numpy': numpy.__version__, 'numba': numba.__version__,\n"
        "    'geopandas': geopandas.__version__, 'pyogrio': pyogrio.__version__,\n"
        "    'pyarrow': pyarrow.__version__, 'shapely': shapely.__version__}}))\n"
    )
    try:
        p = subprocess.run([f"{a['run_venv']}/bin/python", "-c", probe],
                           capture_output=True, text=True, timeout=300,
                           env=probe_env(probe_root))
    except subprocess.TimeoutExpired:
        fail(f"{arm}: identity probe timed out", arm)
    if p.returncode != 0:
        fail(f"{arm}: identity probe failed: {p.stderr[-1500:]}", arm)
    info = json.loads(p.stdout.strip().splitlines()[-1])
    shutil.rmtree(probe_root, ignore_errors=True)

    site = f"{a['run_venv']}/lib/python3.11/site-packages"
    nbc_n, nbc_hits = nbc_census(site)
    una_real = os.path.realpath(info["una_path"])
    identity_ok = una_real.startswith(os.path.realpath(a["run_venv"])
                                      + os.sep)
    versions = info["versions"]
    pinned_matches = {k: (versions.get(k) == v) for k, v in PINNED.items()}
    symbols = info["symbols"]
    symbols_ok = symbols == expected_symbols

    # installed Engines file bytes == this invocation's archive (CHANGED 4)
    installed_engines = f"{una_real}/Engines"
    module_pin_checks = {}
    for fname in ENGINES:
        fpath = os.path.join(installed_engines, fname)
        got = sha256_file(fpath) if os.path.isfile(fpath) else None
        module_pin_checks[fname] = {"archived": archived_shas[fname],
                                    "installed": got,
                                    "match": got == archived_shas[fname]}
    module_pins_ok = all(c["match"] for c in module_pin_checks.values())

    rec = {
        "task": "A3R", "record": f"wheel_and_venv_{arm}",
        "role": "screen-executor", "arm": arm,
        "status": "PENDING_CROSS_ARM",
        "git_archive_ref": ref,
        "build_copy": a["build_copy"],
        "wheel": {"path": wheel_path, "sha256": wheel_sha,
                  "bytes": os.path.getsize(wheel_path)},
        "constraints_file": CONSTRAINTS,
        "runtime_venv": a["run_venv"],
        "python": f"{a['run_venv']}/bin/python",
        "site_packages": site,
        "identity_ok": identity_ok,
        "una_installed_path": info["una_path"],
        "symbols": symbols, "expected_symbols": expected_symbols,
        "symbols_ok": symbols_ok,
        "versions": versions, "pinned": PINNED,
        "pinned_matches": pinned_matches,
        "module_pins": {"engines_dir": installed_engines,
                        "derived_from": f"git archive {ref[:12]} this run",
                        "checks": module_pin_checks,
                        "all_match": module_pins_ok},
        "probe_cache_root": "tempfile.mkdtemp(a3r_build_probe_nbc_) deleted "
                            "after probe",
        "site_packages_nbc_nbi_count": nbc_n,
        "site_packages_nbc_nbi_hits": nbc_hits,
        "build_notes": [
            "wheel built from `git archive` of the PINNED commit (never "
            "HEAD); hatch_build bakes commit_date='unknown' (no .git in the "
            "archive) — code is byte-identical to the archived tree",
            "control-arm re-pin (item 12(b)): the b0 wheel is built fresh "
            "from 9b1340e in this invocation, NOT inherited from A1R",
        ],
        "steps": steps[steps_at_arm_start[arm]:],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    arm_records[arm] = rec
    print(f"[a3r-build] {arm}: wheel {wheel_path}\n"
          f"            sha256 {wheel_sha}\n"
          f"            identity_ok={identity_ok} symbols_ok={symbols_ok} "
          f"module_pins_ok={module_pins_ok} pinned={pinned_matches} "
          f"nbc_nbi_count={nbc_n}")


steps_at_arm_start = {}


def cross_arm():
    """item 12(a) site-packages diff + item 12(d) wheel content compare."""
    # 12(a): full site-packages diff excluding __pycache__/.pyc
    maps = {arm: site_packages_map(ARMS[arm]["run_venv"])
            for arm in ("b0", "cand")}
    all_paths = sorted(set(maps["b0"]) | set(maps["cand"]))
    differing = [q for q in all_paths
                 if maps["b0"].get(q) != maps["cand"].get(q)]
    off = [q for q in differing
           if not (q.startswith("urban_network_analysis")
                   or q.startswith("urban_network_analysis-"))]
    site_diff = {
        "frame": "rel-path sha256 under each venv's site-packages; "
                 "__pycache__ dirs and .pyc files excluded (item 12(a))",
        "compared_paths": len(all_paths),
        "differing_count": len(differing),
        "differing_outside_urban_network_analysis": off[:50],
        "differing_inside": [q for q in differing
                             if q not in set(off)][:80],
        "ok": off == [],
    }
    # 12(d): wheel member content compare
    # [A3R-B5 fold, h04] attempt-1 crash fix: ARMS has no "wheel" key —
    # the path lives on the per-arm record built in build_arm
    wmaps = {arm: wheel_member_map(arm_records[arm]["wheel"]["path"])
             for arm in ("b0", "cand")}
    wnames = sorted(set(wmaps["b0"]) | set(wmaps["cand"]))
    wdiff = [q for q in wnames
             if wmaps["b0"].get(q) != wmaps["cand"].get(q)]
    wdiff_py = [q for q in wdiff if q.endswith(".py")]
    expected_py = sorted(ENGINE_PKG_PREFIX + e for e in ENGINES)
    wheel_diff = {
        "frame": "per-member sha256 of wheel contents (item 12(d))",
        "members_total": len(wnames),
        "differing_count": len(wdiff),
        "differing_py": wdiff_py,
        "differing_non_py": [q for q in wdiff if not q.endswith(".py")][:50],
        "ok": sorted(wdiff_py) == expected_py,
    }
    block = {
        "site_packages_diff": site_diff,
        "wheel_content_diff": wheel_diff,
        "ok": site_diff["ok"] and wheel_diff["ok"],
    }
    for arm in ("b0", "cand"):
        arm_records[arm]["cross_arm_identity"] = block
    return block


def finalize(block_ok):
    for arm in ("b0", "cand"):
        rec = arm_records[arm]
        rec["status"] = (
            "ok" if (block_ok and rec["identity_ok"] and rec["symbols_ok"]
                     and all(rec["pinned_matches"].values())
                     and rec["module_pins"]["all_match"]
                     and rec["site_packages_nbc_nbi_count"] == 0)
            else "DEGRADED")
        with open(ARMS[arm]["record"], "w") as f:
            json.dump(rec, f, indent=1)
        print(f"[a3r-build] {arm}: record {ARMS[arm]['record']} "
              f"status={rec['status']}")
    return 0 if all(r["status"] == "ok"
                    for r in arm_records.values()) else 2


def main():
    # 1. repo state: both pinned commits must exist; tracked tree clean;
    #    HEAD recorded descriptively (NOT gated — content is pinned by ref)
    for arm in ("b0", "cand"):
        ref = ARMS[arm]["commit"]
        p = run(["git", "-C", WT, "cat-file", "-e", ref + "^{commit}"])
        if p.returncode != 0:
            fail(f"pinned commit not present in {WT}: {ref}", arm)
    p = run(["git", "-C", WT, "rev-parse", "HEAD"])
    head_at_build = p.stdout.strip() if p.returncode == 0 else None
    p = run(["git", "-C", WT, "status", "--porcelain", "--untracked-files=no"])
    if p.returncode != 0 or p.stdout.strip():
        fail(f"wt-large-e2e tracked tree not clean:\n{p.stdout}")
    p = run(["git", "-C", WT, "status", "--porcelain"])
    untracked_note = [l for l in p.stdout.splitlines() if l.startswith("?? ")]

    # 2. arm-identity re-derivation gate (ADDED i) — before any packaging
    diff_block = diff_precondition()
    if not diff_block["ok"]:
        fail(f"arm-identity diff precondition FAILED: {diff_block['checks']}")

    # 3. constraints from the campaign venv (arm-independent: the freeze
    #    describes the shared environment, not either candidate)
    if os.path.exists(BUILD_VENV):
        shutil.rmtree(BUILD_VENV)
    p = run([BASE_PY, "-m", "venv", BUILD_VENV])
    if p.returncode != 0:
        fail(f"build venv creation failed: {p.stderr[-500:]}")
    p = run([f"{BUILD_VENV}/bin/pip", "install", "--quiet", "hatchling"])
    if p.returncode != 0:
        fail(f"hatchling install failed: {p.stderr[-800:]}")
    p = run([CAMPAIGN_PY, "-m", "pip", "freeze"])
    if p.returncode != 0:
        fail("pip freeze on campaign venv failed")
    freeze_lines = [l.strip() for l in p.stdout.splitlines() if l.strip()
                    and not l.startswith("-e ")
                    and "urban-network-analysis" not in l
                    and "urban_network_analysis" not in l]
    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(CONSTRAINTS, "w") as f:
        f.write("\n".join(freeze_lines) + "\n")

    # 4. per-arm packaging (b0 first: control arm before candidate)
    for arm in ("b0", "cand"):
        steps_at_arm_start[arm] = len(steps)
        build_arm(arm)

    # 5. cross-arm identity (items 12(a) and 12(d))
    block = cross_arm()
    print(f"[a3r-build] cross-arm: site_diff_ok="
          f"{block['site_packages_diff']['ok']} "
          f"({block['site_packages_diff']['differing_count']} differing) "
          f"wheel_diff_ok={block['wheel_content_diff']['ok']} "
          f"({block['wheel_content_diff']['differing_count']} differing)")

    # 6. finalize both records
    for arm in ("b0", "cand"):
        arm_records[arm]["head_at_build"] = head_at_build
        arm_records[arm]["untracked_files_at_build"] = untracked_note
        arm_records[arm]["arm_identity_diff_rederivation"] = diff_block
    rc = finalize(block["ok"])
    if rc != 0:
        # [A3R-B1 fold, h04] the supervisor's run-stage gate
        # (a3r_run_pairs.py :1014-1026) checks only record/venv path
        # existence and git_archive_ref equality — it does NOT read the
        # ok-flags; enforcement of those is this build's rc=2 plus the
        # reviewer's step-3 audit
        print("[a3r-build] DEGRADED — records retained for review; the "
              "supervisor's run-stage gate checks only record/venv path "
              "existence and git_archive_ref equality, so enforcement of "
              "the ok-flags is this build's rc=2 exit plus the reviewer's "
              "step-3 audit, not a supervisor refusal")
    sys.exit(rc)


if __name__ == "__main__":
    main()
