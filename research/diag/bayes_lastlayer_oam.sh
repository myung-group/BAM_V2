#!/bin/bash
# Last-layer Bayesian regression on crystals (2026-11-09): RACEv2 sz100k L5x128 amsgrad on the OMat24 100k subset,
# periodic design rows with the stress block (the subset carries stress labels).  Posterior from a random training
# subset (MAX_TRAIN, default 2000), calibration on valid / test (MAX_TEST, default 500).  One GPU.
# Cost: P = n_layers * scalars + 1 VJPs per structure (JAC_CHUNK at a time); MAX_TRAIN=200 MAX_TEST=50 for a quick check.
# usage: sbatch -J blr_oam -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 04:00:00 runs/diag/bayes_lastlayer_oam.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
CFG=input/input_3bpa_omat24_sz100k_L5x128amsgrad_omat24_v2_s42_cue.json
CK=$(python -c "import json; print(json.load(open('$CFG'))['ckpt_name'])")
[ -f "${CK%.pkl}.gauge.pkl" ] && CK=${CK%.pkl}.gauge.pkl
echo "checkpoint $CK"
python -m bam_omat24.bayes_lastlayer $CFG --ckpt $CK --max-train ${MAX_TRAIN:-2000} --max-test ${MAX_TEST:-500} \
  --jac-chunk ${JAC_CHUNK:-32} --n-bucket 64 --e-bucket 4096 --stress auto \
  --out runs/bayes/oam100k_L5x128_s42_n${MAX_TRAIN:-2000} 2>&1 | grep -a -v -E "Delay kernel|it/s\]|^\s*$"
