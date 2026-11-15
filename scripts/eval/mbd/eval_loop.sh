#!/bin/bash
# eval_loop.sh <run name> <short tag> <first epoch> <last epoch> <step>: for each epoch, wait for its log line, snapshot the
# fresh .last.pkl (>300 MB, stable 60 s, written within 15 min), run the 3k-sample WBM relaxation on one of our nodes
# (node0485-0498 only), then score (MAE/F1/R2) and RMSD.
cd /scratch/t510a13/race_v2/work
R=$1; T=$2; E0=$3; E1=$4; DE=$5
LOG=runs/${R}_cue/loss_${R}_cue.out; L=runs/${R}_cue/ckpt_${R}_cue.last.pkl
W=/scratch/t510a13/probabilistic_symmetrization/bundle/scratch/t589a02/mp_data/wbm
for E in $(seq $E0 $DE $E1); do
  until grep -a -q -E "^[0-9]+/[0-9]+/[0-9]+ +[0-9:]+ +$E " $LOG 2>/dev/null; do sleep 120; done
  S=runs/mbd/snap/${T}_e$E.pkl
  if [ ! -f $S ]; then
    TS=$(date +%s)
    until M=$(stat -c %Y $L) && [ $M -gt $((TS - 900)) ] && [ $(( $(date +%s) - M )) -gt 60 ] && [ $(stat -c %s $L) -gt 300000000 ]; do sleep 20; done
    command cp -f $L $S
    grep -a -q -E "^[0-9]+/[0-9]+/[0-9]+ +[0-9:]+ +$((E + 1)) " $LOG && echo "WARNING: epoch $((E + 1)) already logged before the snapshot"
  fi
  TG=sample3k_${T}_e$E
  J=$(sbatch --parsable -J wbmsample3k -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 00:40:00 --exclude=node[0449-0484,0499-0576] --export=ALL,CKPT=$S,TAG=$TG,NSH=86,SHARDS=0 scripts/eval/mbd/relax_wbm.sh)
  echo "== $T epoch $E: snapshot $(date '+%F %T'), job $J"
  until St=$(sacct -j $J -X -n -o State 2>/dev/null | awk '{print $1}'); echo "$St" | grep -q -E '^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|DEADLINE|OUT_OF_MEMORY)'; do sleep 60; done
  sacct -j $J -X -n -o JobID,State,Elapsed,Start,NodeList
  if grep -q '^exit 0' runs/slurm/wbmsample3k_$J.out 2>/dev/null; then
    OPENBLAS_NUM_THREADS=1 bash scripts/eval/mbd/score_sample.sh $TG 2>&1 | grep -E '^unique' | cut -c1-230
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /scratch/t510a13/envs/bam_jax/bin/python -I scripts/eval/mbd/rmsd_sample.py --relaxed runs/mbd/$TG/relaxed \
      --mp2020-cse $W/2022-10-19-wbm-computed-structure-entries.jsonl.gz --summary $W/2023-12-13-wbm-summary.csv.gz --output runs/mbd/$TG/rmsd.csv.gz 2>&1 | grep -E '^unique' | cut -c1-200
  else
    echo "relaxation job $J did not finish ($St)"
  fi
done
echo "loop done $(date '+%F %T')"
