#!/bin/bash
# Thermal conductivity of the 103 PhononDB structures with a RACEv2 checkpoint on one node (one process per GPU),
# then the kappa SRME (scripts/eval/mbd/kappa.py).  Rerun-safe: finished materials are skipped.
# usage (from the work folder, default /scratch/t589a02/BAM_V2_more/work):
#   test, 1 GPU, 3 structures:
#   $ sbatch -J kappatest -p normal --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 01:00:00 --export=ALL,CKPT=<ckpt.pkl>,TAG=<name>,NSH=1,LIMIT=3 scripts/eval/mbd/kappa.sh
#   all 103:  NSH=4 (no LIMIT)
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd ${RACE_WORK:-/scratch/t589a02/BAM_V2_more/work}
CKPT=${CKPT:?checkpoint}; TAG=${TAG:?tag}; NSH=${NSH:-4}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/home01/t589a02/.conda/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false ${XLA_EXTRA:-}" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=${OMP:-48} PYTHONUNBUFFERED=1
P=${PHONON_DIR:?folder with the PhononDB reference files}
OUT=runs/mbd/$TAG/kappa; mkdir -p $OUT
echo "node $(hostname)  ckpt $CKPT  shards $NSH  limit ${LIMIT:-none}  code $(git -C $(readlink -f race_v2)/.. rev-parse --short HEAD)  $(date '+%F %T')"
rc=0
for s in $(seq 0 $((NSH - 1))); do
  CUDA_VISIBLE_DEVICES=$s python scripts/eval/mbd/kappa.py run --ckpt $CKPT --structures $P/2024-11-09-phononDB-PBE-103-structures.extxyz \
    --out-dir $OUT --shard $s --num-shards $NSH ${LIMIT:+--limit $LIMIT} ${APC:+--atoms-per-call $APC} >> $OUT/run_${s}of${NSH}.log 2>&1 &
done
for j in $(jobs -p); do wait $j || rc=1; done
for s in $(seq 0 $((NSH - 1))); do grep -E '^\[kappa\]|Traceback|Error' $OUT/run_${s}of${NSH}.log | tail -n 40 | cut -c1-260; done
OPENBLAS_NUM_THREADS=1 python -I scripts/eval/mbd/kappa.py score --out-dir $OUT --reference $P/2024-11-09-kappas-phononDB-PBE-noNAC.json.gz
echo "exit $rc  $(date '+%F %T')"
