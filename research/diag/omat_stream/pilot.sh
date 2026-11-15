#!/bin/bash
# Multi-node layout pilot for OMat24 streaming training, 4 nodes x 4 GPUs (2026-11-11): the same 3 x 400 steps
#   (a) one process per node (4 GPUs each),  (b) one process per GPU;
# the validation files serve as training data here (their size table exists).  Epochs 2-3 give the throughput.
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J omatpilot -p exclusive06 --comment="field=sciai;appl=jax" --nodes=4 --ntasks-per-node=4 --exclusive --gres=gpu:4 -t 00:40:00 research/diag/omat_stream/pilot.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 NCCL_PROTO=^LL128
export BAM_DISTRIBUTED=1 MASTER_ADDR=$(hostname)
D=/scratch/t510a13/data; SZ=$D/oam_sizes
BASE=configs/omat_pretrain/omat24_100k_L5x64_screening.json
echo "nodes $SLURM_JOB_NODELIST  code $(git -C $(readlink -f race_v2)/.. rev-parse --short HEAD)  $(date +%T)"
for L in pernode:1:288:96 pergpu:4:72:24; do
  IFS=: read NAME TPN CPT W <<< "$L"
  N=omat_pilot_$NAME
  rm -rf runs/${N}_cue
  python -I scripts/train/omat_stream/make_config.py $BASE runs/omat_stream/$N.json --name $N \
    --train aselmdb:$D/oam_lmdb/omat24_val --train-sizes $SZ/omat24_val_c5 \
    --valid aselmdb:$D/oam_lmdb/omat24_val --valid-sizes $SZ/omat24_val_c5 \
    --steps-per-epoch 400 --epochs 3 --workers $W --valid-max 2000 \
    --set lr_warmup_epochs=0 --set ckpt_last_every=1000 || exit 1
  export MASTER_PORT=$((20000 + (SLURM_JOB_ID + TPN) % 20000))
  echo "== layout $NAME: $((4 * TPN)) processes, $W loader workers each  $(date +%T)"
  /usr/bin/time -f "WALL %e s ($NAME)" srun --nodes=4 --ntasks=$((4 * TPN)) --ntasks-per-node=$TPN --cpus-per-task=$CPT \
    python -m race_v2.training.train runs/omat_stream/$N.json 2>&1 | grep -v -E '^\s*\^+\s*$' | tail -n 30
  echo "-- log $NAME"
  grep -E 'data parallel|steps per data|^[0-9]+/[0-9]+/[0-9]+ |valid \|' runs/${N}_cue/loss_${N}_cue.out | cut -c1-230 | tail -n 8
done
echo "done $(date +%T)"
