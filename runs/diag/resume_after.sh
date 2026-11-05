#!/bin/bash
# usage: sbatch -J <job> --gres=gpu:<n> -t <time> runs/diag/resume_after.sh <jobid> <run1> [<run2> ...]
#   waits until <jobid> has left the queue, then restarts (restart: true, .last.pkl) every run whose
#   stdout has no "exit 0"; one run per GPU, or a single run on all GPUs of the job
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH -o runs/slurm/%x_%j.out
cd /scratch/t589a02/BAM_V2
JID=$1; shift
while squeue -h -j $JID 2>/dev/null | grep -q .; do sleep 60; done
RUNS=()
for N in "$@"; do grep -q "^exit 0" runs/${N}_cue/stdout.txt 2>/dev/null || RUNS+=($N); done
echo "resuming: ${RUNS[@]}"
if [ ${#RUNS[@]} -eq 1 ] && [ $# -eq 1 ]; then
  cp runs/${RUNS[0]}_cue/stdout.txt runs/${RUNS[0]}_cue/stdout_prev_$SLURM_JOB_ID.txt; bash runs/run_cue.sh ${RUNS[0]}
else
  for i in "${!RUNS[@]}"; do cp runs/${RUNS[$i]}_cue/stdout.txt runs/${RUNS[$i]}_cue/stdout_prev_$SLURM_JOB_ID.txt
    CUDA_VISIBLE_DEVICES=$i bash runs/run_cue.sh ${RUNS[$i]} & done
  wait
fi
for N in "${RUNS[@]}"; do echo "$N: $(tail -n1 runs/${N}_cue/stdout.txt)"; done
