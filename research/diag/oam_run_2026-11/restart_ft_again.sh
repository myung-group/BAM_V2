#!/bin/bash
# One-shot (user-approved 2026-11-15 19:10): the fine-tuning job $1 runs slow after its restart (344 s per 1000 steps
# instead of ~254; GPU side, no loader wait).  At its next virtual-epoch checkpoint cancel it and resubmit the same
# config once (restart: true), then report the rate of the first 2000 steps.
cd /scratch/t510a13/race_v2/work
OLD=$1; F=oam_v1set_sr32_soap_ft; LOG=runs/${F}_cue/loss_${F}_cue.out; K=runs/${F}_cue/ckpt_${F}_cue.last.pkl
E=$(( $(grep -a -E '^[0-9]+/[0-9]+/[0-9]+ ' $LOG | tail -n 1 | awk '{print $3}') + 1 ))
echo "waiting for virtual epoch $E  $(date +%T)"
until grep -a -q -E "^[0-9]+/[0-9]+/[0-9]+ +[0-9:]+ +$E " $LOG; do sleep 30; done
TS=$(date +%s)
until M=$(stat -c %Y $K) && [ $M -gt $((TS - 600)) ] && [ $(( $(date +%s) - M )) -gt 30 ] && [ $(stat -c %s $K) -gt 1000000000 ]; do sleep 10; done
scancel $OLD; echo "virtual epoch $E checkpointed $(date -d @$M +%T); cancelled $OLD  $(date +%T)"
until ! squeue -j $OLD -h 2>/dev/null | grep -q .; do sleep 10; done
J=$(sbatch --parsable -J oamft -p exclusive06 --comment="field=sciai;appl=jax" --nodes=6 --ntasks-per-node=1 --exclusive --gres=gpu:4 -t 3-00:00:00 --exclude=node[0449-0484,0499-0576] --export=ALL,XLA_EXTRA=--xla_gpu_shard_autotuning=false scripts/train/omat_stream/pretrain.sh runs/omat_stream/$F.json)
echo "submitted $J  $(date +%T)"
until grep -a -q -E '^epoch [0-9]+ step 3000 ' runs/slurm/oamft_$J.out 2>/dev/null; do sleep 60; squeue -j $J -h | grep -q . || break; done
squeue -j $J -o '%.9i %.3t %.8M %.5D %R'
grep -a -E 'Resumed|LR at the start' $LOG | tail -n 2 | cut -c1-200
grep -a -E '^first training|^epoch [0-9]+ step (1000|2000|3000) ' runs/slurm/oamft_$J.out; date '+%F %T'
