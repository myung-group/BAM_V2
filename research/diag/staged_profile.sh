#!/bin/bash
# GPU time of the staged halo functions vs the monolithic function (2026-11-06), one GPU
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export BAM_INDEXED_LINEAR=ragged
C=research/input/input_3bpa_omat24_lipsF_fu_s42_cue.json
K=runs/lipsF_fu_s42_cue/ckpt_lipsF_fu_s42_cue.gauge.pkl
for REP in 2 3; do
python research/diag/staged_profile.py $C $K /scratch/t510a13/data/datasets_ext/lips/fu_test.xyz cue $REP 2>&1 | grep -a -E "halo graph|ms$|Error|error|Traceback"
done
