#!/bin/bash
# Back up every evaluated fine-tuning checkpoint (runs/mbd/snap/v1oamft_e*.pkl, oamft_e31_predecay.pkl) to /appsdata
# as soon as it is complete, and at the end of the fine-tuning job $1 its final checkpoint (oamft_final.pkl).
# Each copy is checked with cmp.  User request 2026-11-15 ~16:50.
cd /scratch/t510a13/race_v2/work
FJ=$1; B=/appsdata/t510a13/race_v2_backup_2026-11-15/checkpoints; F=oam_v1set_sr32_soap_ft
copy() {   # copy src if missing at the destination or different
  local s=$1 d=$B/$(basename $1)
  [ -f $d ] && cmp -s $s $d && return
  ionice -c3 cp -p $s $d && cmp -s $s $d && echo "backed up $(basename $s) $(stat -c %s $s) bytes  $(date '+%F %T')"
}
while :; do
  for s in runs/mbd/snap/v1oamft_e*.pkl runs/mbd/snap/oamft_e31_predecay.pkl runs/mbd/snap/oamft_data_ep*.pkl; do
    [ -f $s ] && [ $(( $(date +%s) - $(stat -c %Y $s) )) -gt 120 ] && copy $s
  done
  if ! squeue -j $FJ -h 2>/dev/null | grep -q .; then
    sleep 60
    K=runs/${F}_cue/ckpt_${F}_cue.last.pkl
    LE=$(grep -a -E '^[0-9]+/[0-9]+/[0-9]+ ' runs/${F}_cue/loss_${F}_cue.out | tail -n 1 | awk '{print $3}')
    NE=$(grep -m1 '"n_epochs"' runs/omat_stream/$F.json | tr -dc 0-9)
    if [ "$LE" = "$NE" ]; then command cp -p $K runs/mbd/snap/oamft_final.pkl && copy runs/mbd/snap/oamft_final.pkl
    else echo "job $FJ ended at virtual epoch $LE of $NE (restart?): no final checkpoint saved"; fi
    grep -a -E '^[0-9]+/[0-9]+/[0-9]+ ' runs/${F}_cue/loss_${F}_cue.out | tail -n 1 | cut -c1-40
    for s in runs/mbd/snap/v1oamft_e*.pkl runs/mbd/snap/oamft_e31_predecay.pkl runs/mbd/snap/oamft_data_ep*.pkl; do [ -f $s ] && copy $s; done
    ls -l $B; break
  fi
  sleep 300
done
