#!/bin/bash
# pickle vs stream benchmark on the two local sets (molecules: OMol25 public subset; periodic: 4G-HDNNP Au2/MgO)
cd "$(dirname "$0")/../../.."
O=runs/diag/bench_stream/out
export BENCH_TMP=${BENCH_TMP:-/tmp}
runs/diag/bench_stream/run.sh input/omol/omoloff10k_race_amsgrad_f32_s42.json $O/omol10k.json > $O/omol10k.log 2>&1
runs/diag/bench_stream/run.sh input/local_4g/aumgo_race.json $O/aumgo.json \
  --override '{"x64": false, "optimizer": "amsgrad", "train_path": "data_local/4ghdnnp/aumgo/train.xyz", "batch_size": 8}' \
  > $O/aumgo.log 2>&1
echo QUEUE_DONE >> $O/aumgo.log
