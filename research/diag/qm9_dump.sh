#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -J qm9_dump
#SBATCH -o runs/slurm/%x_%j.out
# per-molecule U0 errors of the QM9 checkpoint (gauge-refit EMA) on valid / test, plus the water runs' per-frame errors
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
N=qm9_u0_hill_soap_s42
python -m race_v2.eval.dump_frame_errors research/input/input_3bpa_omat24_${N}_cue.json --ckpt runs/${N}_cue/ckpt_${N}_cue.gauge.pkl \
  --xyz /scratch/t510a13/data/datasets_ext/qm9/valid.xyz --xyz /scratch/t510a13/data/datasets_ext/qm9/test.xyz \
  --out research/diag/qm9/u0_s42 --batch-size 64 2>&1 | grep -v -E "warn|^W1|^I1"
for W in water2_hill_soap_s42 water2_v9a_soap_s42; do
python -m race_v2.eval.dump_frame_errors research/input/input_3bpa_omat24_${W}_cue.json --ckpt runs/${W}_cue/ckpt_${W}_cue.gauge.pkl \
  --xyz /scratch/t510a13/data/datasets_ext/water_xyz/train_only.xyz --xyz /scratch/t510a13/data/datasets_ext/water_xyz/test.xyz \
  --out research/diag/qm9/${W} --batch-size 2 --index-key none 2>&1 | grep -v -E "warn|^W1|^I1"
done
