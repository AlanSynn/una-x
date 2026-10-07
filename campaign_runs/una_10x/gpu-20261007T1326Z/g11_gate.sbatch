#!/bin/bash
#SBATCH --job-name=g11-gate
#SBATCH --account=bicm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/u/dsynn/projects/una-x/runs/gpu-20261007T1326Z/job_%j.out

set -u
RUN=/u/dsynn/projects/una-x/runs/gpu-20261007T1326Z
REPO=/u/dsynn/projects/una-x/repo
PY=/u/dsynn/projects/una-x/env/bin/python
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR=/u/dsynn/projects/una-x/cache/numba_g11

cd "$REPO"

echo "== STAGE 1: identity =="
hostname
date -u
echo "-- nvidia-smi -L (inside the job) --"
nvidia-smi -L
echo "-- nvidia-smi query --"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
echo "-- modules --"
module list 2>&1
echo "-- nvcc under default toolkit --"
which nvcc && nvcc --version | tail -2

echo "== STAGE 2: host battery (rc 5 expected) =="
"$PY" -m gpu.kernels.validate_device_bits --emit-host-battery "$RUN/host_battery.json"
echo "HOST_BATTERY_RC=$?"

echo "== STAGE 3: device gate PRIMARY (toolkit 12.9 -> PTX 8.8) =="
module load cudatoolkit/26.5_12.9-unsupported
echo "nvcc now: $(which nvcc)"
"$PY" -m gpu.kernels.validate_device_bits \
    --gate-store "$RUN/gate_store.json" \
    --evidence "$RUN/gate_evidence.json"
GATE_RC=$?
echo "GATE_RC_PRIMARY=$GATE_RC"

if [ "$GATE_RC" -eq 3 ]; then
    echo "== STAGE 4: device gate FALLBACK (toolkit 13.2 -> PTX 9.2) =="
    module swap cudatoolkit cudatoolkit/26.5_13.2
    echo "nvcc now: $(which nvcc)"
    "$PY" -m gpu.kernels.validate_device_bits \
        --gate-store "$RUN/gate_store.json" \
        --evidence "$RUN/gate_evidence.json"
    GATE_RC=$?
    echo "GATE_RC_FALLBACK=$GATE_RC"
fi
echo "GATE_RC_FINAL=$GATE_RC"

echo "== STAGE 5: fresh gate-state read from disk =="
"$PY" -c "
import json
from gpu.backends.gpu import gate_state
print(json.dumps(gate_state('una_legacy.fold.default', '$RUN/gate_store.json'), indent=2))
"

echo "== STAGE 6: tests/tenx/gpu on the GPU node =="
"$PY" -m pytest tests/tenx/gpu -q
echo "SUITE_RC=$?"

echo "== DONE =="
date -u
