#!/bin/bash
# Run one training config on the local GPU with limited use, so that the desktop stays responsive:
#   CUDA MPS (user daemon, private pipe dir): at most GPU_SM_PCT % of the SMs (default 50)
#   XLA memory: no preallocation, at most GPU_MEM_FRAC of the 16 GB (default 0.4)
#   CPU: nice 10
# usage: runs/diag/local_4g/run_limited.sh <config.json> <stdout file>
set -u
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.cache/bam_mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.cache/bam_mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 || nvidia-cuda-mps-control -d
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${GPU_SM_PCT:-50}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${GPU_MEM_FRAC:-0.4}
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
cd "$(dirname "$0")/../../.."
exec nice -n 10 /usr/bin/time -f "WALL %e s" ~/anaconda3/envs/bam_cue/bin/python -m bam_omat24.train_3bpa "$1" > "$2" 2>&1
