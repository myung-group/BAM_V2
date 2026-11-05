#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -J r32_dump
#SBATCH -o runs/slurm/%x_%j.out
# per-atom dihedral / 1200K errors of the ZBL runs -> collision-bias and repulsion-slope analysis
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
B=/scratch/t589a02/datasets_ext/BOTNet-datasets/dataset_3BPA
for N in r32_zbl r32_zbl_nb r32_zbl03 r32_bm; do
  python -m bam_omat24.dump_frame_errors input/input_3bpa_omat24_${N}_cue.json --ckpt runs/${N}_cue/ckpt_${N}_cue.gauge.pkl --per-atom --index-key none --batch-size 32 \
    --xyz $B/test_dih.xyz --xyz dataset_3BPA/test_1200K.xyz --out runs/diag/wd/$N 2>&1 | grep -E "MAE|Error|Traceback"
done
