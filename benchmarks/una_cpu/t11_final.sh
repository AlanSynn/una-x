#!/bin/bash
# T11 final confirmation: alternating-order batch runs at the SELECTED
# configuration (8 workers x 1 thread, queue 8, 12 jobs, W3 accessibility),
# frozen sources, exclusive machine lease. Order: cand, base, cand, base, cand.
set -u
PY=/Users/alansynn/orca/workspaces/una-x/venvs/campaign/bin/python
BENCH=/Users/alansynn/orca/workspaces/una-x/wt-integration/benchmarks/una_cpu
BASE_SRC=/Users/alansynn/orca/workspaces/una-x/main/src
CAND_SRC=/Users/alansynn/orca/workspaces/una-x/wt-integration/src
WLDIR=/tmp/workloads
E=/Users/alansynn/orca/workspaces/una-x/wt-integration/optimization_una_cpu/evidence
NBC_BASE=/Users/alansynn/orca/workspaces/una-x/venvs/nbc_base
NBC_CAND=/Users/alansynn/orca/workspaces/una-x/venvs/nbc_cand

lease() { echo "[lease $(date -u +%H:%M:%S)] loadavg=$(uptime | sed -E 's/.*load.*average[^0-9]*//')"; }

run() { # tag src W H
  lease
  env -u PYTHONPATH "$PY" "$BENCH/runner.py" --mode batch --arm source \
    --src "$2" --workload W3_medium_proxy --workload-dir "$WLDIR" \
    --workers 8 --threads 1 --jobs 12 --queue 8 \
    --out "$E/trials/final/runs/$1" >>"$E/trials/final/runs/$1.log" 2>&1 \
    || echo "BATCH FAIL $1"
}

mkdir -p "$E/trials/final/runs"
run run1_cand "$CAND_SRC"
run run2_base "$BASE_SRC"
run run3_cand "$CAND_SRC"
run run4_base "$BASE_SRC"
run run5_cand "$CAND_SRC"
echo "=== final confirmation done ==="
for d in "$E"/trials/final/runs/*/; do
  python3 -c "
import json
j=json.load(open('$d/session.json'))
print('$(basename $d)', j['jobs_done'], j['jobs_failed'], round(j['throughput_jobs_per_s'],3), round(j['batch_wall_ns']/1e9,3))"
done
