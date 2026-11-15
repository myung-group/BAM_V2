#!/bin/bash
# Multi-host debug of the streaming trainer: 2 nodes x 4 GPUs, 2 x 50 steps, full per-task logs and periodic stack
# dumps (BAM_STACK_DUMP_SECS), so a hang in the first step shows where every process waits (2026-11-11).
# usage (from /scratch/t510a13/race_v2/work):
#   (TPN=4 with --ntasks-per-node=4 and --export=ALL,TPN=4 tests one process per GPU)
#   $ sbatch -J omatmh -p exclusive06 --comment="field=sciai;appl=jax" --nodes=2 --ntasks-per-node=1 --exclusive --gres=gpu:4 -t 00:12:00 research/diag/omat_stream/mh_debug.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 NCCL_PROTO=^LL128 NCCL_DEBUG=${NCCL_DEBUG:-WARN} PYTHONUNBUFFERED=1
export BAM_DISTRIBUTED=1 MASTER_ADDR=$(hostname) MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000)) BAM_STACK_DUMP_SECS=150
D=/scratch/t510a13/data; SZ=$D/oam_sizes
BASE=configs/omat_pretrain/omat24_100k_L5x64_screening.json
TPN=${TPN:-1}                      # processes per node: 1 (4 GPUs each) or 4 (one per GPU; sbatch --ntasks-per-node=4)
N=omat_mh_debug_tpn$TPN; rm -rf runs/${N}_cue
python -I scripts/train/omat_stream/make_config.py $BASE runs/omat_stream/$N.json --name $N \
  --train aselmdb:$D/oam_lmdb/omat24_val --train-sizes $SZ/omat24_val_c5 \
  --valid aselmdb:$D/oam_lmdb/omat24_val --valid-sizes $SZ/omat24_val_c5 \
  --steps-per-epoch 100 --epochs 3 --workers $((48 / TPN)) --valid-max 500 \
  --set lr_warmup_epochs=0 --set ckpt_last_every=1000 || exit 1
echo "nodes $SLURM_JOB_NODELIST  coordinator $MASTER_ADDR:$MASTER_PORT  $(date +%T)"
srun --nodes=$SLURM_NNODES --ntasks=$((SLURM_NNODES * TPN)) --ntasks-per-node=$TPN --cpus-per-task=$((288 / TPN)) --output=runs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_task%t.out \
  python -X faulthandler -m race_v2.training.train runs/omat_stream/$N.json
echo "srun exit $?  $(date +%T)"
grep -E 'data parallel|steps per data|^[0-9]+/[0-9]+/[0-9]+ |valid \|' runs/${N}_cue/loss_${N}_cue.out | cut -c1-230 | tail -n 8
