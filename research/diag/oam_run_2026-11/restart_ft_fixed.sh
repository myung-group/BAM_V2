#!/bin/bash
# One-shot (user-approved 2026-11-15 ~18:00): after the smoke $2 of the fixed stream.train.parts layout has passed,
# wait for the next virtual-epoch checkpoint of the fine-tuning job $1, cancel it and resume the same config on 6
# nodes with restart: true (the fixed code spreads the 8 MPtrj copies over the data epoch).
cd /scratch/t510a13/race_v2/work
OLD=$1; SM=$2; F=oam_v1set_sr32_soap_ft; LOG=runs/${F}_cue/loss_${F}_cue.out; K=runs/${F}_cue/ckpt_${F}_cue.last.pkl
until St=$(sacct -j $SM -X -n -o State 2>/dev/null | awk '{print $1}'); echo "$St" | grep -q -E '^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)'; do sleep 30; done
SL=runs/oam_ft_smoke2_cue/loss_oam_ft_smoke2_cue.out
grep -a -E 'stream train part|steps per data epoch|valid loss terms' $SL | cut -c1-200
if ! grep -q '^exit 0' runs/slurm/ftsmoke2_$SM.out || [ $(grep -a -c 'valid loss terms' $SL) -lt 2 ]; then
  echo "smoke $SM did not pass ($St): the fine-tuning is NOT restarted"; tail -n 5 runs/slurm/ftsmoke2_$SM.out; exit 1
fi
echo "smoke passed  $(date +%T)"
E=$(( $(grep -a -c -E '^[0-9]+/[0-9]+/[0-9]+ ' $LOG) + 1 ))
until grep -a -q -E "^[0-9]+/[0-9]+/[0-9]+ +[0-9:]+ +$E " $LOG; do sleep 30; done
TS=$(date +%s)
until M=$(stat -c %Y $K) && [ $M -gt $((TS - 600)) ] && [ $(( $(date +%s) - M )) -gt 30 ] && [ $(stat -c %s $K) -gt 1000000000 ]; do sleep 10; done
scancel $OLD; echo "virtual epoch $E checkpointed $(date -d @$M +%T); cancelled $OLD  $(date +%T)"
until ! squeue -j $OLD -h 2>/dev/null | grep -q .; do sleep 10; done
/scratch/t510a13/envs/bam_jax/bin/python -I -c "
import json; f='runs/omat_stream/$F.json'; c=json.load(open(f)); c['restart']=True
c['note'] += '  2026-11-15: resumed at virtual epoch $E with the fixed parts layout (MPtrj copies spread over the epoch).'
json.dump(c, open(f,'w'), indent=1)"
J=$(sbatch --parsable -J oamft -p exclusive06 --comment="field=sciai;appl=jax" --nodes=6 --ntasks-per-node=1 --exclusive --gres=gpu:4 -t 3-00:00:00 --exclude=node[0449-0484,0499-0576] --export=ALL,XLA_EXTRA=--xla_gpu_shard_autotuning=false scripts/train/omat_stream/pretrain.sh runs/omat_stream/$F.json)
echo "submitted $J  $(date +%T)"
until grep -a -q -E '^epoch [0-9]+ step 2000 ' runs/slurm/oamft_$J.out 2>/dev/null; do sleep 60; squeue -j $J -h | grep -q . || break; done
squeue -j $J -o '%.9i %.3t %.8M %.5D %R'
grep -a -E 'stream train part|LR schedule|Resumed|LR at the start|steps per data epoch' $LOG | tail -n 6 | cut -c1-220
grep -a -E '^first training|^epoch [0-9]+ step (1000|2000) ' runs/slurm/oamft_$J.out; date '+%F %T'
