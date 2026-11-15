#!/bin/bash
# One-shot (user-approved 2026-11-14): after the next virtual-epoch checkpoint of the OMat pretraining (job $1), cancel
# it and resume the same config on $2 nodes (restart: true; WSD constant LR, so the node count may change), then
# report the resume line, the LR and the step rate of the first 2000 steps.
cd /scratch/t510a13/race_v2/work
OLD=$1; N=$2; R=omat24_v1set_sr32_soap_pre; LOG=runs/${R}_cue/loss_${R}_cue.out; L=runs/${R}_cue/ckpt_${R}_cue.last.pkl
E=$(( $(grep -a -c -E '^[0-9]+/[0-9]+/[0-9]+ ' $LOG) + 1 ))
echo "waiting for virtual epoch $E  $(date '+%F %T')"
until grep -a -q -E "^[0-9]+/[0-9]+/[0-9]+ +[0-9:]+ +$E " $LOG; do sleep 30; done
TS=$(date +%s)
until M=$(stat -c %Y $L) && [ $M -gt $((TS - 600)) ] && [ $(( $(date +%s) - M )) -gt 30 ] && [ $(stat -c %s $L) -gt 1000000000 ]; do sleep 10; done
scancel $OLD; echo "epoch $E checkpointed $(date -d @$M +%T); cancelled $OLD  $(date +%T)"
until ! squeue -j $OLD -h 2>/dev/null | grep -q .; do sleep 10; done
/scratch/t510a13/envs/bam_jax/bin/python -I -c "
import json; f='runs/omat_stream/$R.json'; c=json.load(open(f)); c['restart']=True; json.dump(c, open(f,'w'), indent=1)"
J=$(sbatch --parsable -J omatpre -p exclusive06 --comment="field=sciai;appl=jax" --nodes=$N --ntasks-per-node=1 --exclusive --gres=gpu:4 -t 3-00:00:00 --exclude=node[0449-0484,0499-0576] --export=ALL,XLA_EXTRA=--xla_gpu_shard_autotuning=false scripts/train/omat_stream/pretrain.sh runs/omat_stream/$R.json)
echo "submitted $J on $N nodes  $(date +%T)"
until grep -a -q -E '^epoch [0-9]+ step 2000 ' runs/slurm/omatpre_$J.out 2>/dev/null; do sleep 60; squeue -j $J -h | grep -q . || break; done
squeue -j $J -o '%.9i %.3t %.8M %.5D %R'
grep -a -E 'LR schedule|Resumed|LR at the start|steps per data epoch' $LOG | tail -n 4 | cut -c1-220
grep -a -E '^first training|^epoch [0-9]+ step (1000|2000) ' runs/slurm/omatpre_$J.out; date '+%F %T'
