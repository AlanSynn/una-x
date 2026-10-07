"""P05S staged screen driver (una_10x post_p00, designs/07 staged
selection + designs/06 backends).

Runs the matched public-call leg matrix over the staged arms, using the
per-leg instrument run_screen_leg.py under EACH ARM'S OWN VENV:

  cmp395cdc5    395cdc5  (numerical/performance comparator, own tree)
  parent        e7c9eda  (P00 measured parent — predecessor arm)
  p02           6fb01d0  (P02I complete: prepared reuse + single-flight)
  p03a          4b680b6  (P02 + P03A: compiled outer trace replay)
  full          86e8b0a  (+ P03B metric reuse, P04 not admitted; HEAD)

Leg matrix (all: cache mode OFF — P00 frozen policy; threads 1;
queue_depth 64; w4 unless named; fresh NUMBA_CACHE_DIR + fresh outdir
per leg; timer brackets ONLY the public RunBatch call):
  round 0  shakedown, one per arm (labelled; consumed, never claimed)
  rounds 1-3  measured windows, round-robin across arms to decorrelate
              host drift (AB/BA-style alternation)
  config   full-stack joint schedules: w2t1, w4t2, w2t2 (w4t1 = the
           stage legs; at most four feasible configs, logical stripes
           fixed per designs/07)

Driver-side asserts (fail-closed):
  * per leg: exit 0, zero failed rows, 64 output files / 147829892 B,
    tree_head == the arm's pinned sha;
  * per arm: output census sha256 BY REL PATH identical across its
    windows (bitwise determinism of the measured artifact);
  * index: screen_index.json with walls, peak RSS, medians, ratios vs
    the parent arm, determinism verdicts, budget appendix.

This driver ORCHESTRATES; every timed figure is read back from the
per-leg JSON files produced by the instrument (never transcribed).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WS = Path("/Users/alansynn/orca/workspaces/una-x")
MAIN = WS / "main"
RUNROOT = MAIN / "campaign_runs/una_10x/tenx-20261005T2331Z"
HERE = RUNROOT / "p05s"
RUNNER = HERE / "run_screen_leg.py"
LEGS = HERE / "legs"
SCRATCH = Path("/tmp/p05s")

PAIRING = MAIN / "campaign_data/inputs/upstream_Boston/Boston/" \
               "local_audit_manifest.csv"
DATA_FOLDER = MAIN / "campaign_data/inputs/upstream_Boston/Boston"

# Expectations pinned from the P01F-registered control cohort census.
EXPECT_FILES = 64
EXPECT_BYTES = 147829892

ARMS = {
    "cmp395cdc5": {"sha": "395cdc5",
                   "tree": WS / "baseline-395cdc5",
                   "venv": WS / "venv-baseline"},
    "parent": {"sha": "e7c9eda",
               "tree": WS / "parent-e7c9eda",
               "venv": WS / "venv-parent-e7c9eda"},
    "p02": {"sha": "6fb01d0",
            "tree": WS / "p02-6fb01d0",
            "venv": WS / "venv-p02-6fb01d0"},
    "p03a": {"sha": "4b680b6",
             "tree": WS / "p03a-4b680b6",
             "venv": WS / "venv-p03a-4b680b6"},
    "full": {"sha": "86e8b0a",
             "tree": MAIN,
             "venv": WS / "venv-p01m"},
}

ROUND_ARMS = ["cmp395cdc5", "parent", "p02", "p03a", "full"]

# (tag, arm, workers, threads, queue_depth)
CONFIG_LEGS = [
    ("cfg_full_w2t1_q64", "full", 2, 1, 64),
    ("cfg_full_w4t2_q64", "full", 4, 2, 64),
    ("cfg_full_w2t2_q64", "full", 2, 2, 64),
]


def _run_leg(tag, arm, workers, threads, qdepth, kind):
    a = ARMS[arm]
    numba_dir = SCRATCH / "numba" / tag
    outdir = SCRATCH / "out" / tag
    for d in (numba_dir, outdir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    leg_json = LEGS / f"{tag}.json"
    if leg_json.exists():
        leg_json.unlink()
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(numba_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("UNA_BATCH_PHASES", None)   # diagnostics off in timed legs
    cmd = [str(a["venv"] / "bin" / "python"), str(RUNNER),
           "--tree", str(a["tree"]),
           "--pairing", str(PAIRING),
           "--data-folder", str(DATA_FOLDER),
           "--outdir", str(outdir),
           "--tag", tag,
           "--out", str(leg_json),
           "--parallel", "--workers", str(workers),
           "--threads", str(threads),
           "--queue-depth", str(qdepth)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    host_wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"LEG FAIL {tag} rc={proc.returncode}\n"
              f"stdout:\n{proc.stdout[-3000:]}\n"
              f"stderr:\n{proc.stderr[-3000:]}")
        sys.exit(20 + ROUND_ARMS.index(arm))
    leg = json.loads(leg_json.read_text())
    # Census law (C30 residue: the cohort store root is the one allowed
    # hidden residue — compared with hidden entries excluded): the
    # REQUIRED census excludes dot-prefixed top-level entries; the
    # hidden residue is recorded per leg, never asserted away.
    required = [f for f in leg["output_files"]
                if not f["rel"].split("/")[0].startswith(".")]
    hidden = [f for f in leg["output_files"]
              if f["rel"].split("/")[0].startswith(".")]
    leg["required_census"] = {"files": len(required),
                              "bytes": sum(f["bytes"] for f in required)}
    leg["hidden_residue"] = {"files": len(hidden),
                             "bytes": sum(f["bytes"] for f in hidden),
                             "law": "cohort store root — the one allowed "
                                    "hidden residue (C30); excluded from "
                                    "the required census"}
    # instrument-level asserts (belt: the runner asserts too)
    assert leg["tag"] == tag, tag
    assert leg["pins"]["tree_head"].startswith(a["sha"]), \
        (tag, leg["pins"]["tree_head"], a["sha"])
    assert leg["zero_failed_jobs"], tag
    assert leg["required_census"]["files"] == EXPECT_FILES, \
        (tag, leg["required_census"]["files"])
    assert leg["required_census"]["bytes"] == EXPECT_BYTES, \
        (tag, leg["required_census"]["bytes"])
    assert leg["config"]["cache_mode"] == "off", tag
    assert leg["config"]["workers"] == workers, tag
    assert leg["config"]["threads_per_worker"] == threads, tag
    assert leg["config"]["queue_depth"] == qdepth, tag
    print(f"LEG OK {tag:24s} arm={arm:11s} w{workers} t{threads} "
          f"wall={leg['timing']['public_call_wall_s']:8.4f}s "
          f"(host {host_wall:6.1f}s) "
          f"rss_tree={leg['sampler']['peak_tree_rss_bytes']/2**20:7.1f}MiB "
          f"residue={leg['hidden_residue']['files']}f/"
          f"{leg['hidden_residue']['bytes']}B")
    sys.stdout.flush()
    return leg


def _census(leg):
    # Artifact contract: fixed 64 file names under a run-TIMESTAMPED
    # output folder (accessibility_YYYY-MM-DD_HHMM).  Normalize that
    # one component before comparing; the hidden cohort-store residue
    # is excluded (C30 residue law).
    import re
    ts = re.compile(r"accessibility_\d{4}-\d{2}-\d{2}_\d{4}")
    return {ts.sub("accessibility_TS", f["rel"]): f["sha256"]
            for f in leg["output_files"]
            if not f["rel"].split("/")[0].startswith(".")}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-only", action="store_true",
                    help="rebuild determinism + screen_index.json from "
                         "the existing legs/*.json without running legs")
    args = ap.parse_args()

    assert PAIRING.exists(), PAIRING
    LEGS.mkdir(parents=True, exist_ok=True)
    legs = {}

    if args.index_only:
        wanted = ([f"sd_{a}" for a in ROUND_ARMS]
                  + [f"{a}_w4t1_r{r}" for r in (1, 2, 3)
                     for a in ROUND_ARMS]
                  + [t for t, *_ in CONFIG_LEGS])
        for t in wanted:
            p = LEGS / f"{t}.json"
            assert p.exists(), f"--index-only: missing {p}"
            legs[t] = json.loads(p.read_text())
    else:
        if SCRATCH.exists():
            shutil.rmtree(SCRATCH)   # own scratch only
        SCRATCH.mkdir(parents=True)

        # round 0: shakedown per arm (labelled, consumed, never claimed)
        for arm in ROUND_ARMS:
            legs[f"sd_{arm}"] = _run_leg(f"sd_{arm}", arm, 4, 1, 64,
                                         "shakedown")

        # rounds 1-3: measured windows, round-robin
        for r in (1, 2, 3):
            for arm in ROUND_ARMS:
                tag = f"{arm}_w4t1_r{r}"
                legs[tag] = _run_leg(tag, arm, 4, 1, 64, "measured")

        # config screen at the full stack (w4t1 comes from stage legs)
        for tag, arm, w, t, q in CONFIG_LEGS:
            legs[tag] = _run_leg(tag, arm, w, t, q, "measured")

    # ---- per-arm bitwise determinism across windows ----------------
    determinism = {}
    for arm in ROUND_ARMS:
        tags = [f"{arm}_w4t1_r{r}" for r in (1, 2, 3)]
        censuses = [_census(legs[t]) for t in tags]
        same = all(c == censuses[0] for c in censuses[1:])
        shakedown_same = _census(legs[f"sd_{arm}"]) == censuses[0]
        determinism[arm] = {
            "windows": tags,
            "bitwise_identical_across_windows": same,
            "shakedown_matches_windows": shakedown_same,
        }
        assert same, f"{arm}: outputs differ across windows"
        assert shakedown_same, f"{arm}: shakedown differs from windows"

    # ---- medians and ratios (computed here, from the leg JSONs) ----
    def med(arm):
        v = sorted(legs[f"{arm}_w4t1_r{r}"]["timing"]["public_call_wall_s"]
                   for r in (1, 2, 3))
        return v[1]

    parent_med = med("parent")
    stage_table = {}
    for arm in ROUND_ARMS:
        m = med(arm)
        walls = [legs[f"{arm}_w4t1_r{r}"]["timing"]["public_call_wall_s"]
                 for r in (1, 2, 3)]
        stage_table[arm] = {
            "sha_prefix": ARMS[arm]["sha"],
            "walls_s": walls,
            "median_s": m,
            "median_ratio_vs_parent": parent_med / m,
            "peak_tree_rss_mib_median":
                sorted(legs[f"{arm}_w4t1_r{r}"]["sampler"]
                       ["peak_tree_rss_bytes"] / 2**20 for r in (1, 2, 3))[1],
        }

    cfg_table = {}
    for tag, arm, w, t, q in CONFIG_LEGS:
        cfg_table[tag] = {
            "workers": w, "threads": t, "queue_depth": q,
            "wall_s": legs[tag]["timing"]["public_call_wall_s"],
            "peak_tree_rss_mib":
                legs[tag]["sampler"]["peak_tree_rss_bytes"] / 2**20,
            "ratio_vs_parent_median": parent_med /
                legs[tag]["timing"]["public_call_wall_s"],
        }

    index = {
        "schema": "una-10x-p05s-screen-index-1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design_law": "designs/07 staged selection: P02 alone -> "
                      "+P03A -> +accepted P03B -> admitted P04 (none "
                      "admitted); compare each delta to its immediate "
                      "predecessor at the SAME resource schedule; cache "
                      "mode off is the P00 frozen fixed policy; timer "
                      "brackets ONLY the public RunBatch call",
        "arms": {k: {"sha_prefix": v["sha"], "tree": str(v["tree"]),
                     "venv": str(v["venv"])} for k, v in ARMS.items()},
        "cohort": {"manifest": str(PAIRING),
                   "manifest_sha256": legs["sd_parent"]["pins"]
                   ["manifest_sha256"],
                   "expected_files": EXPECT_FILES,
                   "expected_bytes": EXPECT_BYTES},
        "stage_table": stage_table,
        "config_table": cfg_table,
        "determinism": determinism,
        "legs": {t: {"wall_s": l["timing"]["public_call_wall_s"],
                     "kind": ("shakedown" if t.startswith("sd_")
                              else "measured"),
                     "file": f"legs/{t}.json"}
                 for t, l in legs.items()},
        "budget_appendix": {
            "law": "every leg below is a timed public call; append to "
                   "budget_reconciliation.json (ledger only grows)",
            "leg_walls_s": {t: l["timing"]["public_call_wall_s"]
                            for t, l in legs.items()},
            "sum_leg_walls_s": sum(l["timing"]["public_call_wall_s"]
                                   for l in legs.values()),
        },
        "notes": [
            "screen evidence for candidate selection ONLY — NOT the "
            "final 10x qualification (7 paired AB/BA windows, seed "
            "20261003, 20000 bootstrap remain the original protocol)",
            "the frozen primary 92.1141s (PACE) is untouched; local "
            "ratios are same-host matched-arm ratios",
        ],
    }
    out = HERE / "screen_index.json"
    out.write_text(json.dumps(index, indent=1) + "\n")
    print("SCREEN COMPLETE ->", out)
    for arm, row in stage_table.items():
        print(f"  {arm:11s} median {row['median_s']:8.4f}s  "
              f"ratio_vs_parent {row['median_ratio_vs_parent']:6.3f}x")
    for tag, row in cfg_table.items():
        print(f"  {tag:22s} {row['wall_s']:8.4f}s  "
              f"ratio_vs_parent {row['ratio_vs_parent_median']:6.3f}x")


if __name__ == "__main__":
    main()
