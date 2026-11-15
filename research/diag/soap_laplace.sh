#!/bin/bash
# SOAP-basis Laplace with the GGN curvature and the linearised predictive on the 3BPA final recipe
# (2026-11-09; local runs in docs/local_test_report.md section 19).  GGN from $GGN training structures (default 500),
# kappa = n_train, lam grid {1, 10} plus the evidence optimum; MAX_VALID/MAX_TEST cap the sets.
# usage: sbatch -J soaplap -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 02:00:00 research/diag/soap_laplace.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
python -m race_v2.eval.soap_laplace configs/base/race_v2_base_recipe.json \
  --ckpt runs/abl3_full_s42_cue/ckpt_abl3_full_s42_cue.gauge.pkl --out runs/bayes/soap_laplace_abl3_full_s42 \
  --samples 16 --predictive glm --jvp-chunk 4 --ggn ${GGN:-500} --ggn-probes 2 --kappa 1 --kappa-times-ntrain \
  --lam 1 10 --lam-evidence ${MAX_VALID:+--max-valid $MAX_VALID} ${MAX_TEST:+--max-test $MAX_TEST} \
  2>&1 | grep -a -v -E "Delay kernel|it/s\]"
