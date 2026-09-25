#!/bin/bash
# T07 part 3: stage-model reps (fixture now exists) + base_2x4 rerun
# (warm-up race fixed by per-worker warm-up dirs in runner.py).
set -u
PY=/Users/alansynn/orca/workspaces/una-x/venvs/campaign/bin/python
BENCH=/Users/alansynn/orca/workspaces/una-x/wt-integration/benchmarks/una_cpu
BASE_SRC=/Users/alansynn/orca/workspaces/una-x/main/src
CAND_SRC=/Users/alansynn/orca/workspaces/una-x/wt-integration/src
WLDIR=/tmp/workloads
E=/Users/alansynn/orca/workspaces/una-x/wt-integration/optimization_una_cpu/evidence/selection/initial
NBC_BASE=/Users/alansynn/orca/workspaces/una-x/venvs/nbc_base
NBC_CAND=/Users/alansynn/orca/workspaces/una-x/venvs/nbc_cand

lease() { echo "[lease $(date -u +%H:%M:%S)] loadavg=$(uptime | sed -E 's/.*load.*average[^0-9]*//')"; }

for rep in 1 2 3 4 5; do
  for pair in "base $BASE_SRC $NBC_BASE" "cand $CAND_SRC $NBC_CAND"; do
    set -- $pair
    lease
    env -u PYTHONPATH PYTHONPATH="$2" NUMBA_CACHE_DIR="$3" NUMBA_NUM_THREADS=1 \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$PY" "$BENCH/profile_stages.py" --src-root "$2" \
      --workload W3_medium_proxy --workload-dir "$WLDIR" --threads 1 \
      --out "$E/stage_runs/$1_rep${rep}" >>"$E/stage_runs/driver.log" 2>&1 \
      || echo "STAGE FAIL $1 rep$rep"
  done
done

lease
rm -rf "$E/runs/base_2x4"
env -u PYTHONPATH "$PY" "$BENCH/runner.py" --mode batch --arm source \
  --src "$BASE_SRC" --workload W3_medium_proxy --workload-dir "$WLDIR" \
  --workers 2 --threads 4 --jobs 12 --queue 8 \
  --out "$E/runs/base_2x4" >>"$E/runs/base_2x4.log" 2>&1 \
  || echo "BATCH FAIL base 2x4"
echo "=== part 3 done ==="
ls "$E/stage_runs" | grep -c stage_profile.json || true
python3 -c "
import json
j=json.load(open('$E/runs/base_2x4/session.json'))
print('base_2x4:', j['jobs_done'], j['jobs_failed'], round(j['throughput_jobs_per_s'],3))"
