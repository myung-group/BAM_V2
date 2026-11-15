#!/bin/bash
# OMol25 pilot queue on the local GPU (docs/local_test_report.md section 10): one run at a time with limited GPU use
# (research/diag/local_4g/run_limited.sh), then per-frame errors on the test and validation files and, for RACE_LR, the
# Qeq charges vs the reference charges.
# usage: setsid nohup research/diag/omol_pilot_queue.sh <name> ... > runs/omol/queue.log 2>&1 < /dev/null &
#        (configs research/input/omol/<name>.json, outputs runs/omol/<name>/)
cd "$(dirname "$0")/../.."
mkdir -p runs/omol
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.cache/bam_mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.cache/bam_mps/log
for n in "$@"; do
  cfg=research/input/omol/$n.json
  test=$(python3 -c "import json; print(json.load(open('$cfg'))['test_paths'][0])")
  valid=$(python3 -c "import json; print(json.load(open('$cfg'))['valid_path'])")
  rm -rf runs/omol/$n; mkdir -p runs/omol/$n
  echo "$(date '+%F %T') start $n"
  research/diag/local_4g/run_limited.sh $cfg runs/omol/$n/stdout.txt
  echo "$(date '+%F %T') end $n exit $?"
  (
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${GPU_SM_PCT:-50} XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=${GPU_MEM_FRAC:-0.4} XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
    PY="nice -n 10 $HOME/anaconda3/envs/bam_cue/bin/python"
    $PY -m race_v2.eval.dump_frame_errors $cfg --ckpt runs/omol/$n/ckpt.pkl --xyz $test --xyz $valid --out runs/omol/$n/frames
    case $n in *racelr*) $PY research/diag/local_4g/charges.py $cfg runs/omol/$n/ckpt.pkl $test runs/omol/$n/charges_test.npz ;; esac
  ) > runs/omol/$n/post.txt 2>&1
  echo "$(date '+%F %T') post $n done"
done
echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 && echo quit | nvidia-cuda-mps-control
echo "$(date '+%F %T') queue done"
