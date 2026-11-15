#!/bin/bash
# next_epoch.sh <run> <last seen epoch>: wait until a newer virtual epoch is logged, then print the last epochs.
cd /scratch/t510a13/race_v2/work
R=$1; N=$2; L=runs/${R}_cue/loss_${R}_cue.out
until [ "$(grep -a -E '^[0-9]+/[0-9]+/[0-9]+ ' $L | tail -n 1 | awk '{print $3}')" -gt "$N" ] 2>/dev/null; do
  sleep 60; squeue -u t510a13 -n oamft -h | grep -q . || { echo "no oamft job in the queue"; break; }
done
date +%T; /scratch/t510a13/envs/bam_jax/bin/python -I scripts/eval/mbd/val_table.py $R $((N - 2))
grep -a -E '^[0-9]+/[0-9]+/[0-9]+ ' $L | tail -n 1 | awk '{print "epoch", $3, "took", $4, "s"}'
