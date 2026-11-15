#!/bin/bash
# Local 4G-HDNNP test queue (docs/local_test_report.md sections 7-9): one run at a time with limited GPU use
# (run_limited.sh), then per-frame errors on the test split and the training subset (dump_frame_errors) and, for
# RACE_LR runs, the Qeq charges vs Hirshfeld (charges.py).
# usage: runs/diag/local_4g/queue.sh [name ...]   (default: <system>_racelr and <system>_race for the 4 systems)
# Configs: input/local_4g/<name>.json; outputs: runs/local_4g/<name>/.
# XYZ_SUFFIX (e.g. _spin): post-evaluation on data_local/4ghdnnp/<system>/{test,train_sub*}<suffix>.xyz.
# Run detached:  setsid nohup runs/diag/local_4g/queue.sh ... > runs/local_4g/queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../../.."
mkdir -p runs/local_4g
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  for s in ag_cluster carbon_chain nacl aumgo; do names+=(${s}_racelr ${s}_race); done
fi
export CUDA_MPS_PIPE_DIRECTORY=$HOME/.cache/bam_mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.cache/bam_mps/log
for n in "${names[@]}"; do
  s=${n%%_race*}; sub=1000; [ "$s" = aumgo ] && sub=100
  rm -rf runs/local_4g/$n; mkdir -p runs/local_4g/$n
  echo "$(date '+%F %T') start $n"
  runs/diag/local_4g/run_limited.sh input/local_4g/$n.json runs/local_4g/$n/stdout.txt
  echo "$(date '+%F %T') end $n exit $?"
  (
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${GPU_SM_PCT:-50} XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=${GPU_MEM_FRAC:-0.4} XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
    PY="nice -n 10 $HOME/anaconda3/envs/bam_cue/bin/python"
    $PY -m bam_omat24.dump_frame_errors input/local_4g/$n.json --ckpt runs/local_4g/$n/ckpt.pkl \
        --xyz data_local/4ghdnnp/$s/test${XYZ_SUFFIX}.xyz --xyz data_local/4ghdnnp/$s/train_sub$sub${XYZ_SUFFIX}.xyz \
        --out runs/local_4g/$n/frames
    case $n in *racelr*) $PY runs/diag/local_4g/charges.py input/local_4g/$n.json runs/local_4g/$n/ckpt.pkl \
        data_local/4ghdnnp/$s/test${XYZ_SUFFIX}.xyz runs/local_4g/$n/charges_test.npz ;; esac
  ) > runs/local_4g/$n/post.txt 2>&1
  echo "$(date '+%F %T') post $n done"
done
echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 && echo quit | nvidia-cuda-mps-control
echo "$(date '+%F %T') queue done"
