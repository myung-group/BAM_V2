#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:3
#SBATCH -t 00:30:00
#SBATCH -J water_noout
#SBATCH -o runs/slurm/%x_%j.out
# water test set without the worst outlier frame (min distance 0.70 A), gauge checkpoints
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
RUNS=(water2_hill_soap_s42 water2_hill_adam_s42 water2_v9a_soap_s42)
for i in 0 1 2; do N=${RUNS[$i]}
  (CUDA_VISIBLE_DEVICES=$i python -m bam_omat24.eval_sets input/input_3bpa_omat24_${N}_cue.json --ckpt runs/${N}_cue/ckpt_${N}_cue.gauge.pkl \
    --tag noout --batch-size 2 --xyz /scratch/t589a02/datasets_ext/water_xyz/test.xyz --xyz /scratch/t589a02/datasets_ext/water_xyz/test_noout.xyz 2>&1 | grep -E "^set|test|Error" > runs/${N}_cue/noout.txt) &
done
wait
for N in ${RUNS[@]}; do echo $N; cat runs/${N}_cue/noout.txt; done
