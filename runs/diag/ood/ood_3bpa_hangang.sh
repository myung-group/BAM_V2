#!/bin/bash
# 3BPA out-of-distribution uncertainty check (the BAM V1 paper's Table 4 setting: train 300 K, test 300 / 600 / 1200 K)
# on the final-recipe checkpoint: last-layer BLR (+ 3-seed ensemble), SOAP-basis GGN Laplace with the linearised and the
# MC predictive, then runs/diag/ood/ood_summary.py (calibration per temperature, OOD-detection AUROC, figure).
# Local reduced-model version: docs/local_test_report.md section 20.  One GPU, ~1-2 h; MAX_TEST=400 for a quick pass.
# usage: sbatch -J ood3bpa -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 04:00:00 runs/diag/ood/ood_3bpa_hangang.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
F='grep -a -v -E "Delay kernel|it/s\]|^\s*$"'
CFG=input/input_3bpa_omat24_abl3_full_s42_cue.json
R=runs/abl3_full_s4
CK=${R}2_cue/ckpt_abl3_full_s42_cue.gauge.pkl
O=runs/bayes/ood_3bpa_abl3_full_s42${MAX_TEST:+_quick}; mkdir -p $O
MT=${MAX_TEST:+--max-test $MAX_TEST}
echo "== last-layer BLR + ensemble"
python -m bam_omat24.bayes_lastlayer $CFG --ckpt $CK \
  --ensemble $CK ${R}3_cue/ckpt_abl3_full_s43_cue.gauge.pkl ${R}4_cue/ckpt_abl3_full_s44_cue.gauge.pkl \
  --out $O/blr --jac-chunk 16 $MT 2>&1 | eval $F
echo "== SOAP-basis GGN Laplace, linearised predictive"
python -m bam_omat24.soap_laplace $CFG --ckpt $CK --out $O/lap_glm --ggn ${GGN:-450} --ggn-probes 2 --samples 16 \
  --predictive glm --jvp-chunk 4 --kappa 1 --kappa-times-ntrain --lam 1 10 --lam-evidence --max-valid 50 $MT 2>&1 | eval $F
echo "== SOAP-basis GGN Laplace, MC predictive"
python -m bam_omat24.soap_laplace $CFG --ckpt $CK --out $O/lap_mc --ggn ${GGN:-450} --ggn-probes 2 --samples 16 \
  --kappa 1 10 --kappa-times-ntrain --lam 10 30 100 --max-valid 50 $MT 2>&1 | eval $F
echo "== summary"
python runs/diag/ood/ood_summary.py --blr $O/blr --lap "GGN Laplace, linearised=$O/lap_glm" "GGN Laplace, MC=$O/lap_mc" --out $O/summary
echo OOD3BPA_DONE
