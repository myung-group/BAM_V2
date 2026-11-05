#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:4
#SBATCH -t 10:00:00
#SBATCH -J lipslc_soap
#SBATCH -o runs/slurm/%x_%j.out
# LiPS learning curve (NequIP Table VI / CAMP): train 10/100/1000/2500, valid lc_valid (1000), test fu_test (5001)
cd /scratch/t589a02/BAM_V2
RUNS=(lips_lc10_hill_soap_s42 lips_lc100_hill_soap_s42 lips_lc1000_hill_soap_s42 lips_lc2500_hill_soap_s42)
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i bash runs/run_cue.sh ${RUNS[$i]} & done
wait
for N in ${RUNS[@]}; do echo "$N: $(tail -n1 runs/${N}_cue/stdout.txt)"; done
