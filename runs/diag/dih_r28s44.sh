#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:2
#SBATCH -t 01:30:00
#SBATCH -J dih_r28s44
#SBATCH -o runs/slurm/%x_%j.out
# dihedral (beta slice) scan of the r28 seed-44 SOAP runs (same call as runs/diag/r30_job.sh)
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
RUNS=(r28_soap_s44 r28_soap_lr3e3_s44)
# wait until both training runs have finished (the queue runner may start this job early)
for N in ${RUNS[@]}; do until grep -q "^exit" runs/${N}_cue/stdout.txt 2>/dev/null; do sleep 60; done; done
for i in 0 1; do
  N=${RUNS[$i]}; C=runs/${N}_cue/ckpt_${N}_cue.gauge.pkl; [ -f $C ] || C=runs/${N}_cue/ckpt_${N}_cue.pkl
  (CUDA_VISIBLE_DEVICES=$i python -m bam_omat24.eval_sets input/input_3bpa_omat24_${N}_cue.json --ckpt $C \
     --group-key dihedrals --group-index 1 --tag dihbeta_gauge \
     --xyz /scratch/t589a02/datasets_ext/BOTNet-datasets/dataset_3BPA/test_dih.xyz 2>&1 | grep -E "^set|test_dih|Error" > runs/${N}_cue/dih.txt) &
done
wait
for N in ${RUNS[@]}; do echo "$N"; grep "dihedrals=all" runs/${N}_cue/dih.txt; done
