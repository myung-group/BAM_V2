#!/bin/bash
# LAMMPS plan step 1 (2026-11-06): export the LiPS RACEv2 model (float64) as StableHLO for PJRT, e3nn and cue
# backends, and check the exported function against the normal model call on LiPS test frames.
# usage: sbatch -J rv2_export -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 01:00:00 runs/diag/export_lammps.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
C=input/input_3bpa_omat24_lipsF_fu_s42_cue.json
K=runs/lipsF_fu_s42_cue/ckpt_lipsF_fu_s42_cue.gauge.pkl
for B in e3nn cue; do
  python -m bam_omat24.export_lammps $C --ckpt $K --out runs/lammps_export/lips_$B --backend $B \
    --buckets 128:4096 256:8192 --check /scratch/t510a13/data/datasets_ext/lips/fu_test.xyz --n-check 20 2>&1 \
    | grep -E "exported|custom_call|written|check on|Error|error|Traceback" 
done
