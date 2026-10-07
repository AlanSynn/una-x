#!/bin/bash
#SBATCH --job-name=g11-xover
#SBATCH --account=bicm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/u/dsynn/projects/una-x/runs/gpu-20261007T1326Z/job_%j.out

# G11 fold-cell crossover (PREREGISTRATION A1 walls + A2 probe-first shape).
# Both arms in ONE job on the same A100 node: matched host class, matched
# core allocation (4), same I/O root.  CPU arm = installed engine fold under
# 4 spawn workers x 1 numba thread, queue depth 64, cache off.  GPU arm =
# execute_fold_region via DeviceRuntime, PTX reused from the rc=0 gate
# verdict, durable gate store authority per dispatch.
set -u
RUN=/u/dsynn/projects/una-x/runs/gpu-20261007T1326Z
PY=/u/dsynn/projects/una-x/env/bin/python
export PYTHONDONTWRITEBYTECODE=1
# fresh numba cache root per job (A1: cold rep 1, warm reps 2-7 within job)
export NUMBA_CACHE_DIR=/u/dsynn/projects/una-x/cache/numba_g11_xover_${SLURM_JOB_ID}
export NUMBA_NUM_THREADS=1

cd "$RUN"

echo "== identity =="
hostname
date -u
echo "-- nvidia-smi -L (inside the job) --"
nvidia-smi -L
echo "-- numba / numpy --"
"$PY" -c "import numba, numpy; print('numba', numba.__version__, '| numpy', numpy.__version__)"
echo "-- engine install --"
"$PY" -c "import urban_network_analysis, importlib.metadata as md; print(md.distribution('urban-network-analysis').metadata['Name'], md.distribution('urban-network-analysis').version)"

echo "== crossover driver =="
"$PY" crossover_driver.py
echo "XOVER_RC=$?"
date -u
