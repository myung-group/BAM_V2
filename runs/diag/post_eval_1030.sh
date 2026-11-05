#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:4
#SBATCH -t 03:00:00
#SBATCH -J posteval2
#SBATCH -o runs/slurm/%x_%j.out
# 2026-10-30: (a) dihedral scan of the 1500-ep SOAP 3BPA runs (gauge checkpoints); (b) rMD17 SOAP runs on
# rest_split01 (all non-training frames), gauge checkpoints
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
D=/scratch/t589a02/datasets_ext/rmd17_official
DIH=/scratch/t589a02/datasets_ext/BOTNet-datasets/dataset_3BPA/test_dih.xyz
ev() { python -m bam_omat24.eval_sets input/input_3bpa_omat24_$1_cue.json --ckpt $2 $3 --tag $4 --xyz $5 2>&1 | grep -E "^set|xyz|written|Error|error"; }
(
  export CUDA_VISIBLE_DEVICES=0
  for N in r28_soap_s42 r28_soap_s43 r28_soap_lr3e3_s42 r28_soap_lr3e3_s43; do
    echo "=== dih $N gauge"; ev $N runs/${N}_cue/ckpt_${N}_cue.gauge.pkl "--group-key dihedrals --group-index 1" dihbeta_gauge $DIH
  done
) > runs/diag/post_eval2_dih.txt 2>&1 &
MOLS=(aspirin azobenzene benzene ethanol malonaldehyde naphthalene paracetamol salicylic toluene uracil)
for g in 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES=$g
    for k in "${!MOLS[@]}"; do
      [ $((k % 3 + 1)) -eq $g ] || continue
      m=${MOLS[$k]}; N=rmd17o_${m}_soap_s42
      echo "=== rest $N gauge"; ev $N runs/${N}_cue/ckpt_${N}_cue.gauge.pkl "" rest99k_gauge $D/$m/rest_split01.xyz
    done
  ) > runs/diag/post_eval2_rmd17_gpu$g.txt 2>&1 &
done
wait
echo done
