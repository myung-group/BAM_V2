#!/bin/bash
# Local GPU run of a multi-dataset stream config (docs/HANDOFF_multidataset_thole.md):
#   bash scripts/train/run_multi.sh <name>        -> research/input/multi/<name>.json, log runs/multi/<name>/{loss.out,stdout.txt}
# The config has "restart": true, so the same command continues a stopped run from its last checkpoint.
set -e
cd "$(dirname "$0")/../.."
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate jax
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false      # project rule for every GPU run
export BAM_NEIGHBORS=vesin                               # the size tables are made with the same backend
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "runs/multi/$1"
exec python -m race_v2.training.train "research/input/multi/$1.json" >> "runs/multi/$1/stdout.txt" 2>&1
