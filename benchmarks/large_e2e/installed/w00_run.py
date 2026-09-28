"""W00 supervisor: build both wheels in clean matching envs, install them
into fresh venvs, freeze arm identities, run the installed engagement
probe per arm, one bounded installed-mode smoke run per arm through the
campaign harness (run.py), the H04-N2 tamper probe, and evidence
aggregation into evidence/W00/.

Authored by h05 (packaging/performance owner) cloned from the I00/H05
supervisor frames; NEW instrument — per standing ruling it requires h04
review BEFORE its first fire and runs under two-key authorization.

Build route (declared): `git archive` of each verified tree into
campaign_data build copies, then `pip wheel` — the H05-prevalidated
route.  Neither source worktree is ever written (hatch_build.py's
_build_info.py lands in the copy only, so both wheels bake
commit_date='unknown'; source identity is carried by the pinned
commit/src-tree shas instead).  NO install-time precompile/warm step is
included (that is a Q00 decision under the ruled W00_precompile_reading;
adding one would be an amendment).

Stages (run one at a time; h05 is the sole authorized firer):
  --stage init                      chain this lease log to the I00 log
  --stage build --arm b0|selected|all
  --stage venvs  --arm b0|selected|all
  --stage identities --arm b0|selected|all
  --stage engagement --arm b0|selected|all
  --stage smoke --arm b0|selected|all      (run.py installed mode, O2)
  --stage tamper                    (H04-N2 real-wheel guard probe)
  --stage records                   aggregate evidence/W00/*.json
  --stage status                    read-only accounting
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

CAMPAIGN = "/Users/alansynn/orca/workspaces/una-x"
REPO = f"{CAMPAIGN}/wt-large-e2e"
DATA = f"{CAMPAIGN}/campaign_data"
WT_B0 = f"{CAMPAIGN}/wt-b0"
B0_COMMIT = "361928e4ba38f34622cafe065b0025244db61368"
B0_SRC_TREE = "a5883dddcef3afb8debdb4438a6338da886add6f"
# W00-G1 (h04): selected arm build pin — verified by ancestry + pin
# src-tree integrity in _verify_tree, NOT by HEAD equality; the archive
# window archives this commit directly
SEL_HEAD = "6b53a274e8a8d3e7360ab5e1b38f741107ce3358"
SEL_SRC_TREE = "3c59bb0d3f9bc1fba7547be47ec63add88dd3031"
BASE_PY = "/opt/homebrew/opt/python@3.11/bin/python3.11"
CAMPAIGN_PY = f"{CAMPAIGN}/venvs/campaign/bin/python"

BUILDROOT = f"{DATA}/w00_build"
WHEELS = {"b0": f"{DATA}/wheels/w00_b0", "selected": f"{DATA}/wheels/w00_selected"}
VENV = {"b0": f"{DATA}/venvs/w00_b0", "selected": f"{DATA}/venvs/w00_selected"}
SITES = {a: f"{VENV[a]}/lib/python3.11/site-packages" for a in VENV}
IDENT = {"b0": f"{DATA}/w00_identities/b0.json",
         "selected": f"{DATA}/w00_identities/selected.json",
         "tamper": f"{DATA}/w00_identities/tamper.json"}
RUNROOT = f"{DATA}/w00_run"
NBC_ENG = {a: f"{DATA}/nbc_w00_engage_{a}" for a in ("b0", "selected")}
NBC_IDENT = {a: f"{DATA}/nbc_w00_ident_{a}" for a in ("b0", "selected")}
NBC_PROBE = {a: f"{DATA}/nbc_w00_probe_{a}" for a in ("b0", "selected")}
NBC_SMOKE = {a: f"{DATA}/nbc_w00_smoke_{a}" for a in ("b0", "selected")}
TAMPER_VENV = f"{DATA}/w00_tamper_venv"
NBC_TAMPER = f"{DATA}/nbc_w00_tamper"
# C4a (W00 resume-fire adjudication): tamper_identity and tamper_run must
# not share one numba cache root — run.py's fresh-root guard refuses a
# --cache-root a prior window already populated (path_collision).  The
# identity window keeps NBC_TAMPER; the run window gets its own root,
# left NONEXISTENT so run.py's "new (or empty)" admission accepts it.
NBC_TAMPER_RUN = f"{DATA}/nbc_w00_tamper_run"
ENGAGEMENT_OUT = {a: f"{DATA}/w00_engagement_{a}.json" for a in ("b0", "selected")}
EVIDENCE = f"{REPO}/campaigns/una_large_e2e/evidence/W00"

LEASE_LOG = f"{DATA}/w00_lease_log.json"
PARENT_LOG = f"{DATA}/i00_lease_log.json"
PARENT_SHA256 = "6617dbb7ff6d57c86006a9123546f4a60f189d9627712ac2be65cc483da8c4ea"
EXPECTED_PARENT_CHARGED_S = 1661.6
PARENT_CHARGED_TOL_S = 0.05
HEAVY_WALL_BUDGET_S = 14400.0
PHYS_BYTES = 17179869184
PRESSURE_STOP = max(1 << 30, int(0.10 * PHYS_BYTES))

PINNED = {"python": "3.11.16", "numpy": "2.4.6", "numba": "0.67.0",
          "geopandas": "1.1.4", "pyogrio": "0.13.0", "shapely": "2.1.2"}
# harness-format (schema_version 1) workload manifest authored for W00 from
# the frozen O2 acquisition manifest; the acquisition manifest itself is NOT
# a run.py workload manifest (lacks analysis/settings/input_files)
W00_MANIFEST = f"{REPO}/tests/large_e2e/installed/w00_O2_smoke.manifest.json"
W00_MANIFEST_SHA256 = (
    "84350bee33d414826371f38b13a37331d971502b79bbe463c0c7f98d828990c0")
RUN_PY = f"{REPO}/benchmarks/large_e2e/run.py"
PROBE = f"{REPO}/tests/large_e2e/installed/engagement_probe.py"
PROBE_SHA256 = (
    "4d5cb80629f61b91b3203a48dd494f211acc182f12c75b2ffaabaf42ab11ddcc")
# one observed job, one worker, fresh per-arm cache root: the smallest
# honest installed-mode exercise of the real harness (H04-N2)
SMOKE_ARGS = ["--manifest", W00_MANIFEST, "--mode", "single",
              "--jobs", "1", "--workers", "1", "--numba-threads", "2",
              "--flow-stripes", "default", "--queue-depth", "1",
              "--writer-limit", "1", "--cpu-budget", "4",
              "--memory-budget-mib", "2048", "--timeout-s", "3600"]
TIMEOUTS = {"build": 1800, "venvs": 1800, "identities": 600,
            "engagement": 3600, "smoke": 3600, "tamper": 1800}
ARM_KIND = {"b0": "b0", "selected": "selected"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lease_log():
    if os.path.exists(LEASE_LOG):
        with open(LEASE_LOG) as f:
            return json.load(f)
    return None


def save_lease_log(log):
    with open(LEASE_LOG, "w") as f:
        json.dump(log, f, indent=1)


# --- class-8 (h04 disposition): chain-integrity window schema -----------
# Every window entry lands at START classed "crashed_child_start" with an
# explicit null end, and is completed on exit (real class, end, wall, rc).
# A mid-stage crash can therefore never leave the lease log silent, and
# the fire chain is run fail-fast (&&) so no rc is ever laundered.

def _complete_refused(entry, window_class):
    """A refusal is a decided window: complete it (rc 2), never leave a
    null end that would read as a crash."""
    entry["class"] = window_class
    entry["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry["wall_s"] = 0.0
    entry["rc"] = 2


def _open_window(what, lease_id):
    log = load_lease_log()
    if log is None:
        return None
    log["windows"].append({
        "what": what, "class": "crashed_child_start", "lease_id": lease_id,
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "end_utc": None, "wall_s": None, "rc": None})
    save_lease_log(log)
    return len(log["windows"]) - 1


def _close_window(index, cls, rc, extra=None):
    if index is None:
        return
    log = load_lease_log()
    entry = log["windows"][index]
    entry["class"] = cls
    entry["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry["wall_s"] = 0.0   # envelopes are wall-0: child windows carry wall
    entry["rc"] = rc
    if extra:
        entry.update(extra)
    log["updated_utc"] = entry["end_utc"]
    save_lease_log(log)


def _run_stage(what, fn, lease_id, *args):
    """Stage envelope: open at start, complete with the stage rc.  An
    exception propagates and leaves the persisted crashed_child_start +
    null-end entry — the failure is visible in the log, and the fail-fast
    chain stops on the nonzero exit."""
    idx = _open_window(what, lease_id)
    rc = fn(*args)
    _close_window(idx, "stage", rc)
    return rc


def child_env(extra):
    """Clean-room child env: no PYTHONPATH, no NUMBA_DISABLE_JIT (both are
    forbidden in installed mode), plus the declared additions."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "NUMBA_DISABLE_JIT", "NUMBA_CACHE_DIR")}
    env.update(extra)
    return env


def stage_init(lease_id):
    if load_lease_log() is not None:
        print("[w00-run] REFUSED: lease log already exists (no double init)")
        return 2
    if not os.path.exists(PARENT_LOG):
        print("[w00-run] REFUSED: parent I00 lease log missing")
        return 2
    got = sha256_file(PARENT_LOG)
    if got != PARENT_SHA256:
        print(f"[w00-run] REFUSED: parent log sha {got} != pinned")
        return 2
    with open(PARENT_LOG) as f:
        parent = json.load(f)
    opening = round(float(parent["cumulative_charged_s"]), 1)
    if abs(opening - EXPECTED_PARENT_CHARGED_S) > PARENT_CHARGED_TOL_S:
        print(f"[w00-run] REFUSED: parent cumulative_charged_s {opening} != "
              f"{EXPECTED_PARENT_CHARGED_S}")
        return 2
    manifest_sha = sha256_file(W00_MANIFEST)
    if manifest_sha != W00_MANIFEST_SHA256:
        print(f"[w00-run] REFUSED: W00 manifest sha {manifest_sha} != pinned")
        return 2
    probe_sha = sha256_file(PROBE)
    if probe_sha != PROBE_SHA256:
        print(f"[w00-run] REFUSED: engagement probe sha {probe_sha} != pinned")
        return 2
    # N2: engagement/smoke/tamper children launch with cwd=RUNROOT; create
    # it at init so the first spawn cannot die on a missing directory
    os.makedirs(RUNROOT, exist_ok=True)
    save_lease_log({
        "lease_id": lease_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_log": PARENT_LOG, "parent_log_sha256": got,
        "w00_manifest": W00_MANIFEST,
        "w00_manifest_sha256": W00_MANIFEST_SHA256,
        "probe": PROBE, "probe_sha256": PROBE_SHA256,
        "opening_balance_s": opening,
        "opening_balance_source": ("cumulative_charged_s of "
                                   f"{PARENT_LOG} read fresh at init"),
        "budget_s": HEAVY_WALL_BUDGET_S,
        "pressure_stop_bytes": PRESSURE_STOP,
        "windows": [], "cumulative_charged_s": opening,
        "remaining_s": round(HEAVY_WALL_BUDGET_S - opening, 1),
    })
    print(f"[w00-run] init: chained to {os.path.basename(PARENT_LOG)} "
          f"(opening {opening}s of {HEAVY_WALL_BUDGET_S}s)")
    return 0


def guard_and_launch(cmd, env, lease_id, what, timeout_s,
                     cwd=None, window_class="heavy"):
    import psutil
    log = load_lease_log()
    if log is None:
        print("[w00-run] REFUSED: lease not initialized (run --stage init)")
        return 2
    avail = psutil.virtual_memory().available
    ceiling = min(10650 * (1 << 20), int(0.80 * avail))
    # class-8: lands at START crashed_child_start + null end; completed
    # on exit with the real window class
    entry = {
        "what": what, "class": "crashed_child_start",
        "window_class": window_class, "lease_id": lease_id,
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "end_utc": None, "wall_s": None, "rc": None,
        "available_at_admission_bytes": avail, "ceiling_bytes": ceiling,
        "pressure_stop_bytes": PRESSURE_STOP,
        "loadavg_at_admission": os.getloadavg(),
        "cmd": cmd, "cwd": cwd,
        "env": {k: v for k, v in env.items()
                if k.startswith(("NUMBA", "PYTHONDONT", "UNA_"))},
    }
    if avail < PRESSURE_STOP:
        entry["refused"] = "available < pressure stop before launch (RESOURCES.md)"
        _complete_refused(entry, window_class)
        log["windows"].append(entry)
        save_lease_log(log)
        print(f"[w00-run] REFUSED: available {avail} < pressure stop")
        return 2
    if log["cumulative_charged_s"] >= HEAVY_WALL_BUDGET_S:
        entry["refused"] = "heavy wall budget exhausted"
        _complete_refused(entry, window_class)
        log["windows"].append(entry)
        save_lease_log(log)
        print("[w00-run] REFUSED: heavy wall exhausted")
        return 2
    log["windows"].append(entry)
    save_lease_log(log)

    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, env=env, timeout=timeout_s, cwd=cwd,
                           capture_output=True, text=True)
        rc, tail = p.returncode, (p.stdout[-3000:] + p.stderr[-3000:])
    except subprocess.TimeoutExpired:
        rc, tail = -9, f"TIMEOUT after {timeout_s}s"
    wall = time.monotonic() - t0

    log = load_lease_log()
    entry = log["windows"][-1]
    entry.update({"class": window_class,
                  "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "wall_s": round(wall, 1), "rc": rc, "tail": tail[-2000:],
                  "loadavg_at_completion": os.getloadavg()})
    log["cumulative_charged_s"] = round(
        log["opening_balance_s"]
        + sum(w.get("wall_s", 0.0) or 0.0 for w in log["windows"]), 1)
    log["remaining_s"] = round(HEAVY_WALL_BUDGET_S - log["cumulative_charged_s"], 1)
    log["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_lease_log(log)
    print(f"[w00-run] rc={rc} wall={wall:.1f}s "
          f"charged={log['cumulative_charged_s']}s "
          f"remaining={log['remaining_s']}s")
    return rc


def _verify_tree(repo, head, src_tree, label):
    # W00-G1 (h04 ruling): HEAD equality is non-convergent for the selected
    # arm under commit-before-init (the amendment commit moves HEAD past any
    # constant the committed bytes can name), and a HEAD-containment gate
    # would leave pyproject/hatch_build unpinned across descendants.  The
    # build is archive-by-pin (stage_build archives `head` itself), so the
    # selected arm verifies pin ancestry + pin integrity; b0 keeps the
    # HEAD-equality checks.
    if label == "selected":
        p = subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor",
                            head, "HEAD"], capture_output=True, text=True)
        if p.returncode != 0:
            print(f"[w00-run] REFUSED: {label} pin {head} is not an ancestor "
                  f"of HEAD (git rc={p.returncode})")
            return False
        p = subprocess.run(["git", "-C", repo, "rev-parse", f"{head}:src"],
                           capture_output=True, text=True)
        if p.returncode != 0 or p.stdout.strip() != src_tree:
            print(f"[w00-run] REFUSED: {label} pin src tree "
                  f"{p.stdout.strip()!r} != {src_tree}")
            return False
    else:
        p = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        if p.returncode != 0 or p.stdout.strip() != head:
            print(f"[w00-run] REFUSED: {label} HEAD {p.stdout.strip()!r} "
                  f"!= {head}")
            return False
    p = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                       capture_output=True, text=True)
    if p.returncode != 0 or p.stdout.strip():
        print(f"[w00-run] REFUSED: {label} worktree not clean:\n{p.stdout}")
        return False
    if label != "selected":
        p = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD:src"],
                           capture_output=True, text=True)
        if p.returncode != 0 or p.stdout.strip() != src_tree:
            print(f"[w00-run] REFUSED: {label} src tree {p.stdout.strip()!r} "
                  f"!= {src_tree}")
            return False
    return True


TREES = {"b0": (WT_B0, B0_COMMIT, B0_SRC_TREE),
         "selected": (REPO, SEL_HEAD, SEL_SRC_TREE)}


def stage_build(arm, lease_id):
    rc_total = 0
    for a in ([arm] if arm != "all" else ["b0", "selected"]):
        repo, head, src_tree = TREES[a]
        if not _verify_tree(repo, head, src_tree, a):
            return 2
        broot = f"{BUILDROOT}/{a}"
        if os.path.exists(broot):
            shutil.rmtree(broot)
        os.makedirs(f"{broot}/src_copy")
        # archive+extract are write-capable: logged windows, no shell pipe
        tarball = f"{broot}/src.tar"
        rc = guard_and_launch(
            ["git", "-C", repo, "archive", f"--output={tarball}", head],
            child_env({}), lease_id, f"archive:{a}", timeout_s=300,
            window_class="build")
        if rc != 0:
            print(f"[w00-run] archive failed: {a}")
            return 2
        rc = guard_and_launch(
            ["tar", "-x", "-C", f"{broot}/src_copy", "-f", tarball],
            child_env({}), lease_id, f"extract:{a}", timeout_s=300,
            window_class="build")
        if rc != 0:
            print(f"[w00-run] archive extraction failed: {a}")
            return 2
        if not _verify_tree_extracted(f"{broot}/src_copy"):
            return 2
        bvenv = f"{broot}/venv"
        if os.path.exists(bvenv):
            shutil.rmtree(bvenv)
        env = child_env({"PYTHONDONTWRITEBYTECODE": "1"})
        rc = guard_and_launch(
            [BASE_PY, "-m", "venv", bvenv], env, lease_id, f"build:{a}",
            timeout_s=TIMEOUTS["build"], window_class="build")
        if rc != 0:
            rc_total = rc_total or rc
            continue
        rc = guard_and_launch(
            [f"{bvenv}/bin/pip", "install", "--quiet", "hatchling"], env,
            lease_id, f"build_venv:{a}", timeout_s=600, window_class="build")
        if rc != 0:
            rc_total = rc_total or rc
            continue
        os.makedirs(WHEELS[a], exist_ok=True)
        rc = guard_and_launch(
            [f"{bvenv}/bin/pip", "wheel", ".", "--no-build-isolation",
             "--no-deps", "-w", WHEELS[a]],
            env, lease_id, f"build_wheel:{a}", timeout_s=TIMEOUTS["build"],
            cwd=f"{broot}/src_copy", window_class="build")
        if rc != 0:
            rc_total = rc_total or rc
            continue
        whls = [f for f in os.listdir(WHEELS[a]) if f.endswith(".whl")]
        if len(whls) != 1:
            print(f"[w00-run] REFUSED: {a} expected one wheel, found {whls}")
            return 2
        print(f"[w00-run] built {a}: {whls[0]} "
              f"sha {sha256_file(f'{WHEELS[a]}/{whls[0]}')[:16]}")
    return rc_total


def _verify_tree_extracted(copy):
    ok = os.path.isfile(f"{copy}/pyproject.toml") and \
        os.path.isdir(f"{copy}/src/urban_network_analysis") and \
        os.path.isfile(f"{copy}/hatch_build.py")
    if not ok:
        print(f"[w00-run] REFUSED: extracted build copy incomplete: {copy}")
    return ok


def stage_venvs(arm, lease_id):
    rc_total = 0
    for a in ([arm] if arm != "all" else ["b0", "selected"]):
        whls = [f for f in os.listdir(WHEELS[a]) if f.endswith(".whl")]
        if len(whls) != 1:
            print(f"[w00-run] REFUSED: {a} wheel missing (run --stage build)")
            return 2
        wheel_path = f"{WHEELS[a]}/{whls[0]}"
        broot = f"{BUILDROOT}/{a}"
        p = subprocess.run([CAMPAIGN_PY, "-m", "pip", "freeze"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            print("[w00-run] REFUSED: pip freeze on campaign venv failed")
            return 2
        freeze_lines = [l.strip() for l in p.stdout.splitlines()
                        if l.strip() and not l.startswith("-e ")
                        and "urban" not in l.lower()]
        constraints = f"{broot}/constraints.txt"
        with open(constraints, "w") as f:
            f.write("\n".join(freeze_lines) + "\n")
        if os.path.exists(VENV[a]):
            shutil.rmtree(VENV[a])
        env = child_env({})
        rc = guard_and_launch(
            [BASE_PY, "-m", "venv", VENV[a]], env, lease_id, f"venv:{a}",
            timeout_s=600, window_class="build")
        if rc != 0:
            rc_total = rc_total or rc
            continue
        rc = guard_and_launch(
            [f"{VENV[a]}/bin/pip", "install", "--quiet", "--upgrade", "pip"],
            env, lease_id, f"venv_pip:{a}", timeout_s=600,
            window_class="build")
        if rc != 0:
            rc_total = rc_total or rc
            continue
        rc = guard_and_launch(
            [f"{VENV[a]}/bin/pip", "install", "--quiet", "-c", constraints,
             wheel_path, "pyarrow", "pyogrio", "threadpoolctl"],
            env, lease_id, f"venv_install:{a}", timeout_s=TIMEOUTS["venvs"],
            window_class="build")
        if rc != 0:
            rc_total = rc_total or rc
            continue
        # the probe IMPORTS the engine package: route it through
        # guard_and_launch with a pinned cache dir + no bytecode writes,
        # never a bare subprocess under ambient env
        nbc = NBC_PROBE[a]
        if os.path.exists(nbc):
            shutil.rmtree(nbc)
        os.makedirs(nbc)
        probe_out = f"{broot}/identity_probe.json"
        probe = (
            "import json, sys, numba, numpy, geopandas, pyogrio, shapely, "
            "urban_network_analysis as u\n"
            "import os\n"
            "p = os.path.dirname(os.path.abspath(u.__file__))\n"
            "info = {'una_path': p, 'site': sys.prefix,\n"
            " 'versions': {'python': sys.version.split()[0],\n"
            " 'numpy': numpy.__version__, 'numba': numba.__version__,\n"
            " 'geopandas': geopandas.__version__, "
            "'pyogrio': pyogrio.__version__, "
            "'shapely': shapely.__version__}}\n"
            f"open({probe_out!r}, 'w').write(json.dumps(info))\n")
        rc = guard_and_launch(
            [f"{VENV[a]}/bin/python", "-c", probe],
            child_env({"NUMBA_CACHE_DIR": nbc,
                       "PYTHONDONTWRITEBYTECODE": "1"}),
            lease_id, f"identity_probe:{a}", timeout_s=300,
            window_class="probe")
        if rc != 0:
            print(f"[w00-run] REFUSED: identity probe failed for {a}")
            return 2
        with open(probe_out) as f:
            info = json.load(f)
        versions = info["versions"]
        site_real = os.path.realpath(SITES[a])
        if not os.path.realpath(info["una_path"]).startswith(site_real + os.sep):
            print(f"[w00-run] REFUSED: {a} package not under site-packages: "
                  f"{info['una_path']}")
            return 2
        bad = {k: versions.get(k) for k, v in PINNED.items() if versions.get(k) != v}
        if bad:
            print(f"[w00-run] REFUSED: {a} pinned version mismatch: {bad}")
            return 2
        p = subprocess.run([f"{VENV[a]}/bin/pip", "freeze"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            print(f"[w00-run] REFUSED: pip freeze on {VENV[a]} failed")
            return 2
        with open(f"{BUILDROOT}/{a}/venv_freeze.json", "w") as f:
            json.dump(p.stdout.splitlines(), f, indent=1)
        print(f"[w00-run] venv {a}: pinned={ {k: versions[k] for k in PINNED} }")
    return rc_total


def _arm_wheel_sha(a):
    """Exactly-one-wheel check; returns (path, sha256) or (None, None)."""
    whls = [f for f in os.listdir(WHEELS[a]) if f.endswith(".whl")]
    if len(whls) != 1:
        return None, None
    path = f"{WHEELS[a]}/{whls[0]}"
    return path, sha256_file(path)


def stage_identities(arm, lease_id):
    rc_total = 0
    for a in ([arm] if arm != "all" else ["b0", "selected"]):
        os.makedirs(os.path.dirname(IDENT[a]), exist_ok=True)
        # identity make IMPORTS the package (identity.py :370): the numba
        # cache dir MUST be pinned to a declared fresh root and bytecode
        # writes disabled, or caches land inside the arm venv
        nbc = NBC_IDENT[a]
        if os.path.exists(nbc):
            shutil.rmtree(nbc)
        os.makedirs(nbc)
        wheel_path, wheel_sha = _arm_wheel_sha(a)
        if wheel_sha is None:
            print(f"[w00-run] REFUSED: arm {a} does not have exactly one wheel")
            return 2
        rc = guard_and_launch(
            [f"{VENV[a]}/bin/python", "-m", "harness.identity", "--make",
             "--arm", f"w00_{a}", "--kind", "installed_wheel",
             "--out", IDENT[a], "--site-packages", SITES[a],
             "--notes", (f"W00 installed identity for the {a} arm; "
                         f"wheel_sha256={wheel_sha}")],
            child_env({"NUMBA_CACHE_DIR": nbc,
                       "PYTHONDONTWRITEBYTECODE": "1"}),
            lease_id, f"identity:{a}", timeout_s=TIMEOUTS["identities"],
            cwd=f"{REPO}/benchmarks/large_e2e", window_class="identity")
        if rc != 0:
            rc_total = rc_total or rc
            continue
        digest = sha256_file(IDENT[a])
        with open(IDENT[a]) as f:
            idata = json.load(f)
        print(f"[w00-run] identity {a}: tree {idata['package_tree_sha256'][:16]} "
              f"({len(idata['module_hashes'])} files) file-sha {digest[:16]} "
              f"wheel {wheel_sha[:16]}")
    return rc_total


def stage_engagement(arm, lease_id):
    rc_total = 0
    for a in ([arm] if arm != "all" else ["b0", "selected"]):
        nbc = NBC_ENG[a]
        if os.path.exists(nbc):
            shutil.rmtree(nbc)
        os.makedirs(nbc)
        env = child_env({"NUMBA_CACHE_DIR": nbc,
                         "PYTHONDONTWRITEBYTECODE": "1",
                         "NUMBA_NUM_THREADS": "2"})
        rc = guard_and_launch(
            [f"{VENV[a]}/bin/python", PROBE, "--arm", f"w00_{a}",
             "--kind", ARM_KIND[a], "--identity", IDENT[a],
             "--out", ENGAGEMENT_OUT[a]],
            env, lease_id, f"engage:{a}", timeout_s=TIMEOUTS["engagement"],
            cwd=RUNROOT, window_class="engagement")
        if rc != 0:
            rc_total = rc_total or rc
    return rc_total


def stage_smoke(arm, lease_id):
    rc_total = 0
    for a in ([arm] if arm != "all" else ["b0", "selected"]):
        nbc = NBC_SMOKE[a]
        if os.path.exists(nbc):
            shutil.rmtree(nbc)
        os.makedirs(nbc)
        out = f"{RUNROOT}/smoke_{a}"
        env = child_env({"NUMBA_CACHE_DIR": nbc,
                         "PYTHONDONTWRITEBYTECODE": "1"})
        rc = guard_and_launch(
            [f"{VENV[a]}/bin/python", RUN_PY, *SMOKE_ARGS,
             "--arm", f"w00_{a}", "--identity", IDENT[a],
             "--cache-root", nbc, "--out", out],
            env, lease_id, f"smoke:{a}", timeout_s=TIMEOUTS["smoke"],
            cwd=RUNROOT, window_class="smoke")
        if rc != 0:
            rc_total = rc_total or rc
    return rc_total


def stage_tamper(lease_id):
    """H04-N2 real-wheel tamper probe: a disposable copy of the selected
    arm venv, a fresh identity over the COPY's site-packages, then one
    mutated installed module file — run.py installed mode must REFUSE
    (rc != 0).  rc == 0 is a BLOCKING guard failure, recorded as such."""
    if os.path.exists(TAMPER_VENV):
        shutil.rmtree(TAMPER_VENV)
    if os.path.exists(NBC_TAMPER):
        shutil.rmtree(NBC_TAMPER)
    os.makedirs(NBC_TAMPER)
    shutil.copytree(VENV["selected"], TAMPER_VENV, symlinks=True)
    tamper_site = f"{TAMPER_VENV}/lib/python3.11/site-packages"
    _, sel_wheel_sha = _arm_wheel_sha("selected")
    rc = guard_and_launch(
        [f"{TAMPER_VENV}/bin/python", "-m", "harness.identity", "--make",
         "--arm", "w00_tamper", "--kind", "installed_wheel",
         "--out", IDENT["tamper"], "--site-packages", tamper_site,
         "--notes", ("tamper probe: identity over the copied venv, pre-tamper; "
                     f"wheel_sha256={sel_wheel_sha}")],
        child_env({"NUMBA_CACHE_DIR": NBC_TAMPER,
                   "PYTHONDONTWRITEBYTECODE": "1"}),
        lease_id, "tamper_identity", timeout_s=600,
        cwd=f"{REPO}/benchmarks/large_e2e", window_class="tamper")
    if rc != 0:
        return rc
    target = f"{tamper_site}/urban_network_analysis/Settings.py"
    with open(target, "a") as f:
        f.write("\n# W00 TAMPER PROBE — appended byte, guard must refuse\n")
    # C4a: dispose any stale run-window cache root from a prior fire; it
    # is left nonexistent for run.py's fresh-root admission (see the
    # NBC_TAMPER_RUN constant note)
    if os.path.exists(NBC_TAMPER_RUN):
        shutil.rmtree(NBC_TAMPER_RUN)
    out = f"{RUNROOT}/tamper_smoke"
    rc = guard_and_launch(
        [f"{TAMPER_VENV}/bin/python", RUN_PY, *SMOKE_ARGS,
         "--arm", "w00_tamper", "--identity", IDENT["tamper"],
         "--cache-root", NBC_TAMPER_RUN, "--out", out],
        child_env({"NUMBA_CACHE_DIR": NBC_TAMPER_RUN,
                   "PYTHONDONTWRITEBYTECODE": "1"}),
        lease_id, "tamper_run", timeout_s=TIMEOUTS["tamper"],
        cwd=RUNROOT, window_class="tamper")
    outcome = {"expected": ("refusal (rc != 0) sourced from the worker "
                            "startup whole-tree hash comparison"),
               "observed_rc": rc,
               "tampered_file": target, "verdict": None}
    session = f"{out}/session.json"
    if os.path.exists(session):
        with open(session) as f:
            outcome["session"] = json.load(f)
    # C4b (W00 resume-fire adjudication): rc != 0 alone is not "refused
    # as designed" — the verdict must verify the refusal's SOURCE.  The
    # designed refusal is the worker startup whole-tree hash comparison:
    # PoolFailure reason 'worker_startup_failed' carrying the
    # 'package tree does not match identity module hashes' message.
    failure = (outcome.get("session") or {}).get("failure") or {}
    detail_str = json.dumps(failure.get("detail") or {}, default=str)
    designed = (rc != 0
                and failure.get("reason") == "worker_startup_failed"
                and "does not match identity module hashes" in detail_str)
    outcome["designed_signature"] = {
        "rc_nonzero": rc != 0,
        "failure_reason": failure.get("reason"),
        "tree_hash_message": ("does not match identity module hashes"
                              in detail_str),
    }
    if rc == 0:
        outcome["verdict"] = ("GUARD_FAILURE — installed mode accepted a "
                              "tampered wheel")
        print(f"[w00-run] TAMPER PROBE GUARD FAILURE: {outcome}")
    elif designed:
        outcome["verdict"] = ("ok — installed mode refused the tampered "
                              "wheel at the designed tree-hash comparison")
        print(f"[w00-run] tamper probe: refused as designed "
              f"(rc={rc}, worker_startup_failed tree-hash mismatch)")
    else:
        outcome["verdict"] = (
            "MISROUTED_REFUSAL reason="
            f"{failure.get('reason') or 'no-session-failure'}")
        print(f"[w00-run] TAMPER PROBE MISROUTED REFUSAL: {outcome}")
    with open(f"{DATA}/w00_tamper_outcome.json", "w") as f:
        json.dump(outcome, f, indent=1, default=str)
    # dispose the tampered copy either way (evidence retained in the JSON)
    if os.path.exists(TAMPER_VENV):
        shutil.rmtree(TAMPER_VENV)
    # stage rc: 0 only on the designed refusal; any other outcome is a
    # stage failure (rc 1) the fire chain must stop on (h04 C4b ruling)
    return 0 if designed else 1


def _same(x, y):
    return x == y


def stage_records(lease_id):
    """Aggregate evidence/W00/{wheels,installed_tests,path_engagement}.json.
    Self-enveloped (class-8): the window opens at START in the
    crashed_child_start schema and is completed on every decided exit; a
    crash leaves the explicit null end persisted.  Missing engagement
    receipts are an explicit failed-records verdict BEFORE any write —
    never a raw FileNotFoundError after side effects."""
    idx = _open_window("records:evidence_W00", lease_id)
    missing = [a for a in ("b0", "selected")
               if not os.path.exists(ENGAGEMENT_OUT[a])]
    if missing:
        verdict = (f"failed_records: engagement receipts missing for "
                   f"{missing}; no aggregation performed")
        _close_window(idx, "records", 1, {"verdict": verdict})
        print(f"[w00-run] {verdict}")
        return 1
    os.makedirs(EVIDENCE, exist_ok=True)
    wheels = {"task": "W00", "record": "wheels",
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "build_route": {
                  "method": "git archive of the verified tree -> pip wheel in a fresh build venv",
                  "source_worktrees_written": False,
                  "commit_date_note": ("hatch_build.py bakes commit_date='unknown' "
                                       "on the archive route; source identity is "
                                       "carried by the pinned commit/src-tree shas"),
                  "install_time_precompile": ("none — a precompile/warm step is a "
                                              "Q00 decision under the ruled "
                                              "W00_precompile_reading"),
                  "diagnostic_frame_cells": ("none — every W00 evidence cell runs "
                                             "installed (identity kind "
                                             "installed_wheel); no "
                                             "diagnostic_profile_source cell "
                                             "exists in W00"),
                  "rebuild_comparability": ("cross-time comparability rides the "
                                            "commit+policy pins (source_commit / "
                                            "src_tree_sha256 / constraints_sha256 "
                                            "per arm), NOT wheel byte equality "
                                            "(h04 W00 ruling); the git-archive "
                                            "route keeps per-route wheel bytes "
                                            "reproducible"),
              },
              "arms": {}}
    ident_data = {}
    for a in ("b0", "selected"):
        repo, head, src_tree = TREES[a]
        wheel_path, wheel_sha = _arm_wheel_sha(a)
        with open(IDENT[a]) as f:
            idata = json.load(f)
        ident_data[a] = idata
        constraints = f"{BUILDROOT}/{a}/constraints.txt"
        with open(f"{BUILDROOT}/{a}/venv_freeze.json") as f:
            vfreeze = json.load(f)
        wheels["arms"][a] = {
            "source_commit": head, "src_tree_sha256": src_tree,
            "wheel": {"path": wheel_path,
                      "filename": os.path.basename(wheel_path) if wheel_path else None,
                      "sha256": wheel_sha,
                      "bytes": os.path.getsize(wheel_path) if wheel_path else None},
            "identity": {"path": IDENT[a], "sha256": sha256_file(IDENT[a]),
                         "package_tree_sha256": idata["package_tree_sha256"],
                         "module_count": len(idata["module_hashes"]),
                         "package_root": idata["package_root"]},
            "constraints_sha256": sha256_file(constraints),
            "runtime_venv": VENV[a],
            "runtime_venv_freeze": vfreeze,
        }
    wheels_path = f"{EVIDENCE}/wheels.json"
    with open(wheels_path, "w") as f:
        json.dump(wheels, f, indent=1)

    installed = {"task": "W00", "record": "installed_tests",
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "cross_arm_byte_verdicts": {}, "arms": {}}
    probes = {}
    for a in ("b0", "selected"):
        with open(ENGAGEMENT_OUT[a]) as f:
            probes[a] = json.load(f)
        installed["arms"][a] = {
            "probe": ENGAGEMENT_OUT[a],
            "probe_sha256": sha256_file(ENGAGEMENT_OUT[a]),
            "status": probes[a].get("status"),
            "records": probes[a].get("records"),
        }
    bh = {a: probes[a]["byte_hashes"] for a in ("b0", "selected")}
    for surface in ("a_side", "f_side"):
        for key in bh["selected"][surface]:
            installed["cross_arm_byte_verdicts"][f"{surface}.{key}"] = \
                _same(bh["selected"][surface][key], bh["b0"][surface][key])
    for sched in ("production_cap", "forced_chunks"):
        installed["cross_arm_byte_verdicts"][f"f3_side.{sched}"] = \
            _same(bh["selected"]["f3_side"][sched], bh["b0"]["f3_side"][sched])
    sel_rec = {r["label"]: r for r in probes["selected"]["records"]}
    installed["selected_receipts"] = {
        "guard_matrix_cells": len(sel_rec["guard_matrix"]["cells"]),
        "f2_long_fast_calls": sel_rec["f_side"]["legs"]["long_admitting"]
        ["f2_stats"]["fast_calls"],
        "f2_refusing_fallback_calls": sel_rec["f_side"]["legs"]["oracle_refusing"]
        ["f2_stats"]["fallback_calls"],
        "no_fit_recorded": bool(sel_rec["f3_side"]["no_fit"]),
    }
    installed["all_equal"] = all(installed["cross_arm_byte_verdicts"].values())
    installed_path = f"{EVIDENCE}/installed_tests.json"
    with open(installed_path, "w") as f:
        json.dump(installed, f, indent=1)

    tamper = {}
    tamper_path = f"{DATA}/w00_tamper_outcome.json"
    if os.path.exists(tamper_path):
        with open(tamper_path) as f:
            tamper = json.load(f)
    path_eng = {"task": "W00", "record": "path_engagement",
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "identity_guards": {a: next(
                    (r for r in probes[a]["records"]
                     if r["label"] == "identity_guards"), None)
                    for a in ("b0", "selected")},
                "smoke_runs": {}, "tamper_probe": tamper}
    for a in ("b0", "selected"):
        session_file = f"{RUNROOT}/smoke_{a}/session.json"
        entry = {"out": f"{RUNROOT}/smoke_{a}", "session_file": session_file}
        if os.path.exists(session_file):
            with open(session_file) as f:
                sess = json.load(f)
            entry["status"] = sess.get("status")
            entry["measurement_class"] = sess.get("measurement_class")
            entry["installed_qualified"] = sess.get("installed_qualified")
            entry["qualification_valid"] = sess.get("qualification_valid")
            sid = sess.get("identity") or {}
            entry["session_identity"] = {
                "identity_kind": sid.get("identity_kind"),
                "identity_sha256": sid.get("identity_sha256"),
                "package_tree_sha256": sid.get("package_tree_sha256"),
                "notes": sid.get("notes"),
            }
        path_eng["smoke_runs"][a] = entry
    # wheel sha + source-commit identity carried into the session record
    # (harness-native: session.identity = identity.describe(), which includes
    # the notes field where the identities stage recorded wheel_sha256)
    bindings = {}
    for a in ("b0", "selected"):
        sid = (path_eng["smoke_runs"][a].get("session_identity") or {})
        idata = ident_data[a]
        whl = wheels["arms"][a]["wheel"]
        bindings[a] = {
            "wheel_sha256": whl["sha256"],
            "source_commit": wheels["arms"][a]["source_commit"],
            "src_tree_sha256": wheels["arms"][a]["src_tree_sha256"],
            "identity_sha256": idata["identity_sha256"],
            "session_identity_sha256": sid.get("identity_sha256"),
            "session_package_tree_sha256": sid.get("package_tree_sha256"),
            "identity_sha256_match": (sid.get("identity_sha256")
                                      == idata["identity_sha256"]),
            "package_tree_sha256_match": (sid.get("package_tree_sha256")
                                          == idata["package_tree_sha256"]),
            "session_notes_carries_wheel_sha": (
                whl["sha256"] in (sid.get("notes") or "")),
        }
    path_eng["session_record_identity_binding"] = bindings
    path_engagement_path = f"{EVIDENCE}/path_engagement.json"
    with open(path_engagement_path, "w") as f:
        json.dump(path_eng, f, indent=1)

    ok = (installed["all_equal"]
          and tamper.get("verdict", "").startswith("ok")
          and all(path_eng["smoke_runs"][a].get("installed_qualified")
                  for a in ("b0", "selected"))
          and all(b["identity_sha256_match"] and b["package_tree_sha256_match"]
                  and b["session_notes_carries_wheel_sha"]
                  for b in bindings.values()))
    _close_window(idx, "records", 0 if ok else 1,
                  {"wrote": [wheels_path, installed_path,
                             path_engagement_path],
                   "verdict": f"all_equal={installed['all_equal']} ok={ok}"})
    print(f"[w00-run] records: wheels/installed_tests/path_engagement "
          f"written; all_equal={installed['all_equal']} ok={ok}")
    return 0 if ok else 1


def stage_status():
    log = load_lease_log()
    if log is None:
        print("[w00-run] no lease log (not initialized)")
        return 0
    print(json.dumps({k: log[k] for k in
                      ("lease_id", "parent_log_sha256", "opening_balance_s",
                       "cumulative_charged_s", "remaining_s", "budget_s")},
                     indent=1))
    print(f"[w00-run] {len(log['windows'])} windows")
    for w in log["windows"]:
        print(f"  {w.get('what')}: rc={w.get('rc')} wall={w.get('wall_s')}s"
              f"{' REFUSED: ' + w['refused'] if w.get('refused') else ''}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["init", "build", "venvs", "identities",
                             "engagement", "smoke", "tamper", "records",
                             "status"])
    ap.add_argument("--arm", choices=["b0", "selected", "all"], default=None)
    ap.add_argument("--lease-id", default="W00")
    args = ap.parse_args()

    if args.stage != "status":
        if not sys.prefix.startswith(f"{CAMPAIGN}/venvs/campaign"):
            print("[w00-run] REFUSED: supervisor must run under the "
                  "campaign venv interpreter")
            return 2
    if args.stage in ("build", "venvs", "identities", "engagement", "smoke"):
        if not args.arm:
            print("[w00-run] REFUSED: --arm required for this stage")
            return 2

    if args.stage == "init":
        # init runs before the lease log exists — no envelope possible;
        # the log it writes IS the init record
        return stage_init(args.lease_id)
    if args.stage == "records":
        # self-enveloped: records completes its own window with the
        # explicit verdict (class-8)
        return stage_records(args.lease_id)
    fns = {"build": (stage_build, (args.arm, args.lease_id)),
           "venvs": (stage_venvs, (args.arm, args.lease_id)),
           "identities": (stage_identities, (args.arm, args.lease_id)),
           "engagement": (stage_engagement, (args.arm, args.lease_id)),
           "smoke": (stage_smoke, (args.arm, args.lease_id)),
           "tamper": (stage_tamper, (args.lease_id,)),
           "status": (stage_status, ())}
    fn, fargs = fns[args.stage]
    what = f"stage:{args.stage}" + (f":{args.arm}" if args.arm else "")
    return _run_stage(what, fn, args.lease_id, *fargs)


if __name__ == "__main__":
    sys.exit(main())
