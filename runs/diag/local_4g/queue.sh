#!/bin/bash
# Local 4G-HDNNP test queue (docs/local_test_report.md section 7): RACE_LR and RACE per system, one at a time, with
# limited GPU use (run_limited.sh). Configs: input/local_4g/<system>_<model>.json; outputs: runs/local_4g/<name>/.
# Run detached:  setsid nohup runs/diag/local_4g/queue.sh > runs/local_4g/queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../../.."
mkdir -p runs/local_4g
for s in ag_cluster carbon_chain nacl aumgo; do for t in racelr race; do
  n=${s}_$t; rm -rf runs/local_4g/$n; mkdir -p runs/local_4g/$n
  echo "$(date '+%F %T') start $n"
  runs/diag/local_4g/run_limited.sh input/local_4g/$n.json runs/local_4g/$n/stdout.txt
  echo "$(date '+%F %T') end $n exit $?"
done; done
echo "$(date '+%F %T') queue done"
