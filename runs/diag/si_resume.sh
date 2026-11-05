#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:2
#SBATCH -t 05:00:00
#SBATCH -J si_resume
#SBATCH -o runs/slurm/%x_%j.out
# resume the Si GAP-18 full runs cut by the 10 h limit of si_fair (restart: true, .last.pkl)
cd /scratch/t589a02/BAM_V2
while squeue -h -j 1181879 2>/dev/null | grep -q .; do sleep 60; done
RUNS=()
for N in si_full_hill_soap_s42 si_full_hill_adam_s42; do grep -q "^exit 0" runs/${N}_cue/stdout.txt || RUNS+=($N); done
for i in "${!RUNS[@]}"; do cp runs/${RUNS[$i]}_cue/stdout.txt runs/${RUNS[$i]}_cue/stdout_part1.txt; CUDA_VISIBLE_DEVICES=$i bash runs/run_cue.sh ${RUNS[$i]} & done
wait
for N in "${RUNS[@]}"; do echo "$N: $(tail -n1 runs/${N}_cue/stdout.txt)"; done
