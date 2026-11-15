#!/bin/bash
# GPU-limited run of the benchmark (same limits as research/diag/local_4g/run_limited.sh)
# usage: research/diag/bench_stream/run.sh <config.json> <out.json> [benchmark args...]
set -u
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.cache/bam_mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.cache/bam_mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 || nvidia-cuda-mps-control -d
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${GPU_SM_PCT:-50}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${GPU_MEM_FRAC:-0.4}
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
cd "$(dirname "$0")/../../.."
exec nice -n 10 ~/anaconda3/envs/bam_cue/bin/python research/diag/bench_stream/bench_stream_vs_pkl.py "$@"
