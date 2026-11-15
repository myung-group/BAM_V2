#!/bin/bash
# Per-element energy refit of a checkpoint on a subset of the training split (fair protocol: training data only).
# One GPU: model energies of the training frames (relax_wbm.py with a huge fmax = no relaxation step), then the
# least-squares fit (refit_fit.py) -> <ckpt>.species_refit.json.  The subset comes from refit_subset.py (login node).
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J specrefit -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 00:30:00 --exclude=node[0449-0484,0499-0576] --export=ALL,CKPT=<snap.pkl>,CFG=<config.json>,XYZ=runs/mbd/refit/mptrj_train_20k.extxyz,TAG=<name> scripts/eval/mbd/species_refit.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
CKPT=${CKPT:?checkpoint}; CFG=${CFG:?config json}; XYZ=${XYZ:?training subset extxyz}; TAG=${TAG:?tag}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
OUT=runs/mbd/refit/$TAG; mkdir -p $OUT
echo "node $(hostname)  ckpt $CKPT  subset $XYZ  code $(git -C $(readlink -f race_v2)/.. rev-parse --short HEAD)  $(date '+%F %T')"
python -m race_v2.inference.relax_wbm --ckpt $CKPT --input $XYZ --out-dir $OUT --shard 0 --num-shards 1 --fmax 1e6 --max-steps 1 \
  --max-structures 64 --groups 2 --host-workers 32 > $OUT/energies.log 2>&1; RC=$?
grep -E '^\[relax\]|\[batched\] FINAL|Traceback|Error' $OUT/energies.log | tail -n 6 | cut -c1-330
[ $RC = 0 ] && python -I scripts/eval/mbd/refit_fit.py --xyz $XYZ --relaxed $OUT --ckpt $CKPT --atomic-numbers-from $CFG
echo "exit $RC  $(date '+%F %T')"
