#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:3
#SBATCH -t 05:30:00
#SBATCH -J r30lr
#SBATCH -o runs/slurm/%x_%j.out
# SOAP learning-rate schedule test (3BPA A+hill, 1500 ep, seed 42), then the dihedral scan of each run
cd /scratch/t510a13/race_v2/work
RUNS=(r30_soap_lr1e2_pat60_s42 r30_soap_lr3e3_pat60_s42 r30_soap_lr3e3_cos_s42)
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i bash runs/run_cue.sh ${RUNS[$i]} & done
wait
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
for i in 0 1 2; do
  N=${RUNS[$i]}; C=runs/${N}_cue/ckpt_${N}_cue.gauge.pkl; [ -f $C ] || C=runs/${N}_cue/ckpt_${N}_cue.pkl
  (CUDA_VISIBLE_DEVICES=$i python -m bam_omat24.eval_sets input/input_3bpa_omat24_${N}_cue.json --ckpt $C \
     --group-key dihedrals --group-index 1 --tag dihbeta_gauge \
     --xyz /scratch/t510a13/data/datasets_ext/BOTNet-datasets/dataset_3BPA/test_dih.xyz 2>&1 | grep -E "^set|test_dih|Error" > runs/${N}_cue/dih.txt) &
done
wait
for N in ${RUNS[@]}; do echo "$N: $(tail -n1 runs/${N}_cue/stdout.txt)"; grep "dihedrals=all" runs/${N}_cue/dih.txt; done
