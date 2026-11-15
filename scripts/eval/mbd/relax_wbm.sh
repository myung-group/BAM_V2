#!/bin/bash
# WBM relaxation with a RACEv2 checkpoint on one node: one process per GPU, each relaxing its CRC32 shards
# (race_v2/inference/relax_wbm.py; Matbench Discovery protocol FIRE / 500 steps / fmax 0.05).  Rerun-safe.
# usage (from /scratch/t510a13/race_v2/work):
#   test, 1 GPU, 1000 structures:
#   $ sbatch -J wbmtest -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 00:40:00 --export=ALL,CKPT=<ckpt.pkl>,TAG=<name>,NSH=1,SHARDS=0,LIMIT=1000 scripts/eval/mbd/relax_wbm.sh
#   full WBM on one node (4 GPUs):  NSH=4,SHARDS="0 1 2 3"   (several nodes: NSH=4*nodes, 4 shards per job)
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
CKPT=${CKPT:?checkpoint}; TAG=${TAG:?tag}; NSH=${NSH:?number of shards}; SHARDS=${SHARDS:?shards of this job}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
WBM=${WBM:-/scratch/t510a13/probabilistic_symmetrization/bundle/scratch/t589a02/mp_data/wbm/2024-08-04-wbm-initial-atoms.extxyz.zip}
OUT=runs/mbd/$TAG/relaxed; mkdir -p $OUT
echo "node $(hostname)  ckpt $CKPT  shards ($SHARDS)/$NSH  limit ${LIMIT:-none}  code $(git -C $(readlink -f race_v2)/.. rev-parse --short HEAD)  $(date '+%F %T')"
g=0; rc=0
for s in $SHARDS; do
  CUDA_VISIBLE_DEVICES=$g python -m race_v2.inference.relax_wbm --ckpt $CKPT --input $WBM --out-dir $OUT \
    --shard $s --num-shards $NSH ${LIMIT:+--limit $LIMIT} --max-structures ${MAXS:-64} --groups 2 \
    --host-workers ${HW:-32} > $OUT/relax_${s}of${NSH}.log 2>&1 &
  g=$((g + 1))
done
for j in $(jobs -p); do wait $j || rc=1; done
for s in $SHARDS; do echo "== shard $s"; grep -E '^\[relax\]|\[batched\] FINAL|Traceback|Error' $OUT/relax_${s}of${NSH}.log | tail -n 6 | cut -c1-330; done
echo "exit $rc  $(date '+%F %T')"
