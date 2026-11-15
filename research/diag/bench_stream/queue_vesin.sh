#!/bin/bash
# the periodic set again with the vesin neighbour list (BAM_NEIGHBORS=vesin; VESIN_PATH = a directory with vesin)
cd "$(dirname "$0")/../../.."
O=research/diag/bench_stream/out
until grep -q QUEUE_DONE $O/aumgo.log 2>/dev/null; do sleep 20; done
export BENCH_TMP=${BENCH_TMP:-/tmp} BAM_NEIGHBORS=vesin PYTHONPATH=${VESIN_PATH:-}
research/diag/bench_stream/run.sh research/input/local_4g/aumgo_race.json $O/aumgo_vesin.json \
  --override '{"x64": false, "optimizer": "amsgrad", "train_path": "data_local/4ghdnnp/aumgo/train.xyz", "batch_size": 8}' \
  > $O/aumgo_vesin.log 2>&1
echo QUEUE_DONE >> $O/aumgo_vesin.log
