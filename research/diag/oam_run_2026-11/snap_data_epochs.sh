#!/bin/bash
# Snapshot the fine-tuning checkpoint at the first virtual-epoch boundary after each data epoch (user 2026-11-15):
# one data epoch = 29,709 steps on 6 nodes = 5.94 virtual epochs, so data epoch k ends inside virtual epoch 6k;
# the saved checkpoint (end of virtual epoch 6k) is 291 k steps (~1 % of a data epoch) past it.
cd /scratch/t510a13/race_v2/work
F=oam_v1set_sr32_soap_ft; L=runs/${F}_cue/loss_${F}_cue.out; K=runs/${F}_cue/ckpt_${F}_cue.last.pkl
for k in 2 3 4 5 6 7; do
  V=$((6 * k))
  until grep -a -q -E "^[0-9]+/[0-9]+/[0-9]+ +[0-9:]+ +$V " $L; do sleep 60; done
  TS=$(date +%s)
  until M=$(stat -c %Y $K) && [ $M -gt $((TS - 600)) ] && [ $(( $(date +%s) - M )) -gt 30 ] && [ $(stat -c %s $K) -gt 1000000000 ]; do sleep 10; done
  grep -a -q -E "^[0-9]+/[0-9]+/[0-9]+ +[0-9:]+ +$((V + 1)) " $L && echo "WARNING: virtual epoch $((V + 1)) already logged"
  command cp -p $K runs/mbd/snap/oamft_data_ep${k}_v${V}.pkl && echo "saved data epoch $k (virtual $V)  $(date '+%F %T')"
done
