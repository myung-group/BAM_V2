#!/bin/bash
# usage: sbatch -J oam_nan -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 01:00:00 runs/diag/oam_nan_debug.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
python runs/diag/oam_nan_debug.py input/input_3bpa_omat24_oamfix_salex_v2_s42_cue.json 2>&1 | grep -v "cuda_timer\|Warning\|warn("
