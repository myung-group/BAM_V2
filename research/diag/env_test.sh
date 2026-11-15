#!/bin/bash
# Environment check of a rebuilt bam_cue env on one GPU (2026-11-11, account t510a13): versions, GPU visibility,
# the cue DCE shim, then the whole test suite on the GPU.
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J envtest -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 01:00:00 research/diag/env_test.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
ENV=${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export SLURM_JOB_ID=${SLURM_JOB_ID:-local}
echo "node $(hostname)  arch $(uname -m)  python $(which python)  code $(git -C $(readlink -f race_v2)/.. rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'EOF'
import jax, flax, optax, e3nn_jax, numpy, vesin, lmdb
print("jax", jax.__version__, "flax", flax.__version__, "optax", optax.__version__, "numpy", numpy.__version__)
print("devices", jax.devices())
import cuequivariance_jax as cuex
print("cuequivariance_jax", cuex.__version__)
import race_v2, os
print("race_v2 from", os.path.dirname(race_v2.__file__))
EOF
echo "== pytest tests (GPU)"
/usr/bin/time -f "WALL %e s" python -m pytest -q --durations=10 -p no:cacheprovider tests 2>&1 | tail -n 40
