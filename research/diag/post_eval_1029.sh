#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:4
#SBATCH -t 03:00:00
#SBATCH -J posteval
#SBATCH -o runs/slurm/%x_%j.out
# 2026-10-29: (a) dihedral scan of the SOAP A+hill checkpoints raw / gauge-refit; (b) rMD17 official split 01:
# gauge refit on the training split, test_split01 (1000) and rest_split01 (all 99,000 non-training frames)
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
D=/scratch/t510a13/data/datasets_ext/rmd17_official
MOLS=(aspirin azobenzene benzene ethanol malonaldehyde naphthalene paracetamol salicylic toluene uracil)
ev() { python -m race_v2.eval.eval_sets research/input/input_3bpa_omat24_$1_cue.json --ckpt $2 $3 --tag $4 --xyz $5 2>&1 | grep -E "^set|xyz|written|Error|error"; }
# (a) dihedral on GPU 0
(
  export CUDA_VISIBLE_DEVICES=0
  JAX_PLATFORMS=cpu python research/diag/make_gauge_ckpt.py 27 r26_hill_soap_lr1e2_s42 r26_hill_soap_lr3e3_s42 2>&1 | grep -v Warn
  for N in r26_hill_soap_lr1e2_s42 r26_hill_soap_lr3e3_s42; do
    echo "=== dih $N raw";   ev $N runs/${N}_cue/ckpt_${N}_cue.pkl "--group-key dihedrals --group-index 1" dihbeta /scratch/t510a13/data/datasets_ext/BOTNet-datasets/dataset_3BPA/test_dih.xyz
    echo "=== dih $N gauge"; ev $N runs/${N}_cue/ckpt_${N}_cue.gauge.pkl "--group-key dihedrals --group-index 1" dihbeta_gauge /scratch/t510a13/data/datasets_ext/BOTNet-datasets/dataset_3BPA/test_dih.xyz
  done
) > research/diag/post_eval_dih.txt 2>&1 &
# (b) rMD17, molecules round-robin over GPUs 1-3
for g in 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES=$g
    for k in "${!MOLS[@]}"; do
      [ $((k % 3 + 1)) -eq $g ] || continue
      m=${MOLS[$k]}; N=rmd17o_${m}_v9a_s42
      python research/diag/gauge_refit.py $N 2>&1 | grep -E "^==|test_|valid"
      JAX_PLATFORMS=cpu python research/diag/make_gauge_ckpt.py auto $N 2>&1 | grep -v Warn
      echo "=== rest $N raw";   ev $N runs/${N}_cue/ckpt_${N}_cue.pkl "" rest99k $D/$m/rest_split01.xyz
      echo "=== rest $N gauge"; ev $N runs/${N}_cue/ckpt_${N}_cue.gauge.pkl "" rest99k_gauge $D/$m/rest_split01.xyz
    done
  ) > research/diag/post_eval_rmd17_gpu$g.txt 2>&1 &
done
wait
echo done
