#!/bin/bash
# RACE_LR unit tests (2026-11-06; written while the allocation was exhausted, not run yet): Ewald vs Madelung,
# alpha independence, batched Ewald, Qeq optimality, model forces / rotation / total-charge dependence.  1 GPU, minutes.
# usage: sbatch -J racelr -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 00:30:00 runs/diag/race_lr_check.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
python -m pytest -q tests/test_race_lr.py 2>&1 | tail -n 15
python -m pytest -q tests/test_oam_stress.py tests/test_periodic.py 2>&1 | tail -n 3      # data_3bpa / trainer edits
