#!/bin/bash
# OMat24 pretraining with the streaming loader on N whole GPU nodes (2026-11-11, account t510a13).
# One process per node drives its 4 GPUs (jax.distributed, BAM_DISTRIBUTED=1); restart: true in the config resumes
# from <ckpt>.last.pkl (data cursor included), so the same command continues an interrupted run.
# usage (from /scratch/t510a13/race_v2/work; the config is made by runs/diag/omat_stream/make_config.py):
#   $ sbatch -J <name> -p exclusive06 --comment="field=sciai;appl=jax" --nodes=4 --ntasks-per-node=1 --exclusive --gres=gpu:4 -t <time> runs/diag/omat_stream/pretrain.sh <config.json>
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
CFG=${1:?config json}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 NCCL_PROTO=^LL128
echo "nodes $SLURM_JOB_NODELIST  tasks $SLURM_NTASKS  config $CFG  code $(git -C $(readlink -f bam_omat24)/.. rev-parse --short HEAD)  $(date '+%F %T')"
TPN=${TPN:-1}                      # processes per node: 1 (4 GPUs each) or 4 (one per GPU; sbatch --ntasks-per-node=4)
if [ "${SLURM_NNODES:-1}" -gt 1 ] || [ $TPN -gt 1 ]; then
  export BAM_DISTRIBUTED=1 MASTER_ADDR=$(hostname) MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
  srun --nodes=$SLURM_NNODES --ntasks=$((SLURM_NNODES * TPN)) --ntasks-per-node=$TPN --cpus-per-task=$((288 / TPN)) \
    python -m bam_omat24.train_3bpa $CFG
else
  python -m bam_omat24.train_3bpa $CFG
fi
echo "exit $?  $(date '+%F %T')"
