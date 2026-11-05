#!/bin/bash
# usage: sbatch -J <job> -p normal --comment="field=sciai;appl=jax" --gres=gpu:<n> -t <time> runs/diag/train_runs.sh <run1> [<run2> ...]
#   trains each run on its own GPU (runs/run_cue.sh, input/input_3bpa_omat24_<run>_cue.json); no 3BPA dihedral scan
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t589a02/BAM_V2
RUNS=("$@")
for i in "${!RUNS[@]}"; do CUDA_VISIBLE_DEVICES=$i bash runs/run_cue.sh ${RUNS[$i]} & done
wait
for N in "${RUNS[@]}"; do echo "$N: $(tail -n1 runs/${N}_cue/stdout.txt)"; done
