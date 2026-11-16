#!/bin/bash
# Timing of the MD force call (scripts/eval/mbd/md_profile.py) on one GPU of one node.
# usage (from the work folder, default /scratch/t589a02/BAM_V2_more/work):
#   $ sbatch -J mdprofile -p normal --comment="field=sciai;appl=jax" --nodes=1 --ntasks-per-node=1 --exclusive --gres=gpu:4 -t 00:20:00 --export=ALL,CKPT=<ckpt.pkl>,TAG=<name>,ARGS="<options of md_profile.py>" scripts/eval/mbd/md_profile.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd ${RACE_WORK:-/scratch/t589a02/BAM_V2_more/work}
CKPT=${CKPT:?checkpoint}; TAG=${TAG:?tag}
if [ -z "$MD_INNER" ]; then
  exec srun --nodes=1 --ntasks=1 --cpus-per-task=288 env MD_INNER=1 bash scripts/eval/mbd/md_profile.sh
fi
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${MBD_ENV:-/home01/t589a02/.conda/envs/mbd314}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false ${XLA_EXTRA:-}"
export BAM_NEIGHBORS=${BAM_NEIGHBORS:-vesin} BAM_PREDICT_X64=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=${OMP:-8} PYTHONUNBUFFERED=1
export MBD_REPO=${MBD_REPO:-/scratch/t589a02/BAM_V2_more/ext/matbench-discovery}
export MBD_CACHE_DIR=${MBD_CACHE_DIR:-/scratch/t589a02/BAM_V2_more/ext/mbd_cache}
OUT=runs/mbd/$TAG; mkdir -p $OUT
echo "node $(hostname)  ckpt $CKPT  args ${ARGS:-}  code $(git -C $(readlink -f race_v2)/.. rev-parse --short HEAD)  $(date '+%F %T')"
df -h /dev/shm /tmp 2>/dev/null | tail -n 2
CUDA_VISIBLE_DEVICES=0 taskset -c 0-71 python scripts/eval/mbd/md_profile.py --ckpt $CKPT ${ARGS:-} > $OUT/profile_$SLURM_JOB_ID.log 2>&1; rc=$?
tr '\r' '\n' < $OUT/profile_$SLURM_JOB_ID.log | grep -a -v -E '^\s*$|WARNING|cuda_timer|Nsight|step/s\]|it/s\]' | tail -n 110 | cut -c1-230
echo "exit $rc  $(date '+%F %T')"
exit $rc
