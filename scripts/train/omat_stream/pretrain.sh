#!/bin/bash
# OMat24 pretraining with the streaming loader on N whole GPU nodes (2026-11-11; paths of account t589a02 since 2026-11-16).
# One process per node drives its 4 GPUs (jax.distributed, BAM_DISTRIBUTED=1); restart: true in the config resumes
# from <ckpt>.last.pkl (data cursor included), so the same command continues an interrupted run.
# usage (from the work folder, default /scratch/t589a02/BAM_V2_more/work; the config is made by scripts/train/omat_stream/make_config.py):
#   $ sbatch -J <name> -p normal --comment="field=sciai;appl=jax" --nodes=4 --ntasks-per-node=1 --exclusive --gres=gpu:4 -t <time> --export=ALL,XLA_EXTRA=--xla_gpu_shard_autotuning=false scripts/train/omat_stream/pretrain.sh <config.json>
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd ${RACE_WORK:-/scratch/t589a02/BAM_V2_more/work}
CFG=${1:?config json}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/home01/t589a02/.conda/envs/bam_cue}
# XLA_EXTRA: more XLA flags.  --xla_gpu_shard_autotuning=false: every process autotunes its whole program itself
# instead of sharing the work with the other processes and joining the results during compilation (job 1262448 sat
# in the first compilation of the training step on both nodes for 2 h; that join makes a compilation wait for
# the other processes -- suspected, not shown, to be where it hangs).
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false ${XLA_EXTRA:-}" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 NCCL_PROTO=^LL128
export BAM_STACK_DUMP_SECS=${BAM_STACK_DUMP_SECS:-300} BAM_PROGRESS_STEPS=${BAM_PROGRESS_STEPS:-1000} PYTHONUNBUFFERED=1
echo "nodes $SLURM_JOB_NODELIST  tasks $SLURM_NTASKS  config $CFG  code $(git -C $(readlink -f race_v2)/.. rev-parse --short HEAD)  $(date '+%F %T')"
TPN=${TPN:-1}                      # processes per node: 1 (4 GPUs each) or 4 (one per GPU; sbatch --ntasks-per-node=4)
if [ "${SLURM_NNODES:-1}" -gt 1 ] || [ $TPN -gt 1 ]; then
  export BAM_DISTRIBUTED=1 MASTER_ADDR=$(hostname) MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
  # start-up watch: a run that has not finished its first training step after START_LIMIT seconds is stopped and
  # started again in the same allocation, at most TRIES times in all (restart: true continues from the last
  # checkpoint, or from the beginning when there is none)
  OUT=runs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out; START_LIMIT=${START_LIMIT:-1500}; TRIES=${TRIES:-3}
  for TRY in $(seq 1 $TRIES); do
    srun --nodes=$SLURM_NNODES --ntasks=$((SLURM_NNODES * TPN)) --ntasks-per-node=$TPN --cpus-per-task=$((288 / TPN)) \
      python -m race_v2.training.train $CFG &
    SPID=$!; T0=$(date +%s); STARTED=0
    while kill -0 $SPID 2>/dev/null; do
      if grep -q 'first training step done' $OUT 2>/dev/null; then STARTED=1; break; fi
      [ $(( $(date +%s) - T0 )) -gt $START_LIMIT ] && break
      sleep 30
    done
    if [ $STARTED = 1 ] || ! kill -0 $SPID 2>/dev/null; then wait $SPID; RC=$?; break; fi
    echo "attempt $TRY of $TRIES: no training step after $START_LIMIT s, stopping this step  $(date '+%F %T')"
    scancel ${SLURM_JOB_ID}.$((TRY - 1)); wait $SPID; RC=$?; sleep 30
  done
else
  python -m race_v2.training.train $CFG; RC=$?
fi
echo "exit $RC  $(date '+%F %T')"
