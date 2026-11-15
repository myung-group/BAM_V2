#!/bin/bash
# Memory / speed benchmark versus system size (float32 and float64) and a float64 JAX reference on rattled structures
# for the PyTorch-port comparison.  env: CKPT, TAG.  One node, GPU 0 and 1.
#   $ sbatch -J racebench -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 00:40:00 --exclude=node[0449-0484,0499-0576] --export=ALL,CKPT=<ckpt.pkl>,TAG=<name> scripts/eval/bench/bench.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
CKPT=${CKPT:?checkpoint}; TAG=${TAG:?tag}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
O=runs/bench/$TAG; mkdir -p $O
echo "node $(hostname)  ckpt $CKPT  $(date '+%F %T')"
CUDA_VISIBLE_DEVICES=2 BAM_PREDICT_X64=1 python scripts/eval/bench/ref_rattled.py $CKPT $O/ref_rattled_x64.npz 16 2>&1 | tail -n 2 &
CUDA_VISIBLE_DEVICES=0 python scripts/eval/bench/mem_bench.py $CKPT $O/mem_float32.json > $O/mem_float32.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 BAM_PREDICT_X64=1 python scripts/eval/bench/mem_bench.py $CKPT $O/mem_float64.json > $O/mem_float64.log 2>&1 &
wait
for d in float32 float64; do echo "== $d"; grep -a '^{' $O/mem_$d.log | cut -c1-330; grep -a -E 'Traceback|Error' $O/mem_$d.log | tail -n 2 | cut -c1-200; done
echo "exit 0  $(date '+%F %T')"
