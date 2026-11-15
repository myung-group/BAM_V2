#!/bin/bash
cd /home/gbsim/prog/BAM_V2
R=research/diag/local_4g/run_limited.sh
for n in ${RUNS:-xyz stream resume_e3 resume_e5}; do
  mkdir -p runs/stream_e2e/${n%_e*}
  $R research/input/stream_e2e/omoloff10k_racelr_q_$n.json runs/stream_e2e/$n.stdout
  echo "DONE $n $?" >> runs/stream_e2e/queue.log
done
echo QUEUE_DONE >> runs/stream_e2e/queue.log
