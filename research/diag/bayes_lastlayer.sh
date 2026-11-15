#!/bin/bash
# Last-layer Bayesian linear regression of RACEv2 on 3BPA (2026-11-06): posterior over the stratum readout weights
# (evidence-maximised hyperparameters, training set), calibration on valid / 300 / 600 / 1200 K / dihedral vs the
# 3-seed deep ensemble of the final recipe.  One GPU.  MAX_TEST=200 for a quick check.
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
R=runs/abl3_full_s4
python -m race_v2.eval.bayes_lastlayer configs/base/race_v2_base_recipe.json \
  --ckpt ${R}2_cue/ckpt_abl3_full_s42_cue.gauge.pkl \
  --ensemble ${R}2_cue/ckpt_abl3_full_s42_cue.gauge.pkl ${R}3_cue/ckpt_abl3_full_s43_cue.gauge.pkl ${R}4_cue/ckpt_abl3_full_s44_cue.gauge.pkl \
  --dihedral /scratch/t510a13/data/datasets_ext/BOTNet-datasets/dataset_3BPA/test_dih.xyz \
  --out runs/bayes/abl3_full_s42${MAX_TEST:+_quick} ${MAX_TEST:+--max-test $MAX_TEST} ${BACKEND:+--backend $BACKEND} 2>&1 \
  | grep -a -v -E "Delay kernel|it/s\]|^\s*$"
