#!/bin/bash
# JAX reference data for the PyTorch port (race_v2_torch/tools/dump_jax_reference.py), one GPU, a few minutes.
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J torchref -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 00:30:00 --exclude=node[0449-0484,0499-0576] --export=ALL,REFS="<ckpt>:<tag> <ckpt>:<tag>" research/diag/torch_port_dump_ref.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
echo "node $(hostname)  code $(git -C /scratch/t510a13/BAM_V2_merged rev-parse --short HEAD)  $(date '+%F %T')"
rc=0
for r in ${REFS:?ckpt:tag list}; do
  CK=${r%%:*}; TG=${r##*:}
  python /scratch/t510a13/BAM_V2_merged/race_v2_torch/tools/dump_jax_reference.py --ckpt $CK --out runs/torch_port/ref_$TG \
    --val /scratch/t510a13/data/mptrj_lmdb/val --n 12 || rc=1
done
echo "exit $rc  $(date '+%F %T')"
