#!/bin/bash
# T07 (L3): stage model + worker/thread sweep under the exclusive machine lease.
#
#   Part 1  stage model : profile_stages.py x5 per tree, alternating trees per
#                         rep to cancel drift (base, cand, base, cand, ...).
#   Part 2  W/H sweep   : configs {1x1, 1x8, 2x4, 4x2, 8x1} x {cand, base},
#                         batch mode, W3_medium_proxy, 12 jobs, queue 8,
#                         sum(workers*threads) <= C=8. W8H1 last (highest
#                         memory pressure). Candidate first per config.
#
# Raw per-run artifacts (session.json + raw.jsonl, stage_profile.json) are the
# evidence; this log only records order, lease state, and failures.
set -u
PY=/Users/alansynn/orca/workspaces/una-x/venvs/campaign/bin/python
BENCH=/Users/alansynn/orca/workspaces/una-x/wt-integration/benchmarks/una_cpu
BASE_SRC=/Users/alansynn/orca/workspaces/una-x/main/src
CAND_SRC=/Users/alansynn/orca/workspaces/una-x/wt-integration/src
WLDIR=/tmp/workloads
E=/Users/alansynn/orca/workspaces/una-x/wt-integration/optimization_una_cpu/evidence/selection/initial
NBC_BASE=/Users/alansynn/orca/workspaces/una-x/venvs/nbc_base
NBC_CAND=/Users/alansynn/orca/workspaces/una-x/venvs/nbc_cand

mkdir -p "$E/stage_runs" "$E/runs"

lease() { echo "[lease $(date -u +%H:%M:%S)] loadavg=$(uptime | sed -E 's/.*load.*average[^0-9]*//') mem_avail=$(memory_pressure -Q 2>/dev/null || echo n/a)"; }

stage_rep() { # tree src nbc rep
  local tree=$1 src=$2 nbc=$3 rep=$4
  lease
  env -u PYTHONPATH PYTHONPATH="$src" NUMBA_CACHE_DIR="$nbc" NUMBA_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PY" "$BENCH/profile_stages.py" --src-root "$src" \
    --workload W3_medium_proxy --workload-dir "$WLDIR" --threads 1 \
    --out "$E/stage_runs/${tree}_rep${rep}" >>"$E/stage_runs/driver.log" 2>&1 \
    || echo "STAGE FAIL $tree rep$rep (see $E/stage_runs/driver.log)"
}

batch_run() { # tag src W H
  local tag=$1 src=$2 W=$3 H=$4
  lease
  env -u PYTHONPATH "$PY" "$BENCH/runner.py" --mode batch --arm source \
    --src "$src" --workload W3_medium_proxy --workload-dir "$WLDIR" \
    --workers "$W" --threads "$H" --jobs 12 --queue 8 \
    --out "$E/runs/${tag}_${W}x${H}" >>"$E/runs/${tag}_${W}x${H}.log" 2>&1 \
    || echo "BATCH FAIL $tag ${W}x${H} (see $E/runs/${tag}_${W}x${H}.log)"
}

echo "=== part 1: stage model (alternating trees, 5 reps each) ==="
for rep in 1 2 3 4 5; do
  stage_rep base "$BASE_SRC" "$NBC_BASE" "$rep"
  stage_rep cand "$CAND_SRC" "$NBC_CAND" "$rep"
done

echo "=== part 2: W/H sweep (candidate then baseline per config) ==="
for cfg in "1 1" "1 8" "2 4" "4 2" "8 1"; do
  set -- $cfg
  batch_run cand "$CAND_SRC" "$1" "$2"
  batch_run base "$BASE_SRC" "$1" "$2"
done

echo "=== done ==="
ls "$E/runs" "$E/stage_runs"
