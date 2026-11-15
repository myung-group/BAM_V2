#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:4
#SBATCH -t 02:00:00
#SBATCH -J wd_dump
#SBATCH -o runs/slurm/%x_%j.out
# per-atom errors for the water / 3BPA dihedral analysis (docs/RACE_water_dihedral_analysis_2026-11-02.md)
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
B=/scratch/t510a13/data/datasets_ext/BOTNet-datasets/dataset_3BPA
W=/scratch/t510a13/data/datasets_ext/water_xyz
d3() { N=$1; G=$2; C=runs/${N}_cue/ckpt_${N}_cue.gauge.pkl; [ -f $C ] || C=runs/${N}_cue/ckpt_${N}_cue.pkl
  CUDA_VISIBLE_DEVICES=$G python -m race_v2.eval.dump_frame_errors research/input/input_3bpa_omat24_${N}_cue.json --ckpt $C --per-atom --index-key none --batch-size 32 \
   --xyz $B/test_dih.xyz --xyz research/dataset_3BPA/train_300K.xyz --xyz research/dataset_3BPA/test_300K.xyz --xyz research/dataset_3BPA/test_1200K.xyz --out research/diag/wd/$N 2>&1 | grep -E "MAE|Error|Traceback"; }
dw() { N=$1; G=$2; CUDA_VISIBLE_DEVICES=$G python -m race_v2.eval.dump_frame_errors research/input/input_3bpa_omat24_${N}_cue.json --ckpt runs/${N}_cue/ckpt_${N}_cue.last.pkl --per-atom --index-key none --batch-size 2 \
   --xyz $W/train_only.xyz --xyz $W/valid.xyz --xyz $W/test.xyz --out research/diag/wd/$N 2>&1 | grep -E "MAE|Error|Traceback"; }
( for N in r28_soap_s42 r28_soap_s43 r28_soap_s44; do d3 $N 0; done ) &
( for N in r28_soap_lr3e3_s42 r28_soap_lr3e3_s43 r28_soap_lr3e3_s44; do d3 $N 1; done ) &
( for N in r31_no_shells r31_base v9a_rp_s42 ; do d3 $N 2; done ) &
( for N in w3_base w3_loss2st w3_noprior; do dw $N 3; done ) &
wait
