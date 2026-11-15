#!/bin/bash
# One-shot (user-approved 2026-11-15 06:15): when the OMat pretraining job $1 has finished virtual epoch 58, copy its
# final checkpoint to runs/mbd/snap/omat_pre_final.pkl, submit the MPtrj+sAlex fine-tuning on 6 nodes, report the
# init line, the LR schedule and the first 2000 steps.
cd /scratch/t510a13/race_v2/work
PJ=$1; P=omat24_v1set_sr32_soap_pre; PL=runs/${P}_cue/loss_${P}_cue.out; PK=runs/${P}_cue/ckpt_${P}_cue.last.pkl
until ! squeue -j $PJ -h 2>/dev/null | grep -q .; do sleep 60; done
sacct -j $PJ -X -n -o JobID,State,Elapsed; tail -n 2 runs/slurm/omatpre_$PJ.out | cut -c1-150
if ! grep -a -q -E '^[0-9]+/[0-9]+/[0-9]+ +[0-9:]+ +58 ' $PL; then echo "pretraining did not reach epoch 58: not starting the fine-tuning"; exit 1; fi
sleep 30; S=runs/mbd/snap/omat_pre_final.pkl; command cp -f $PK $S; ls -l --time-style=+%T $PK $S | awk '{print $5,$6,$7}'
F=oam_v1set_sr32_soap_ft
J=$(sbatch --parsable -J oamft -p exclusive06 --comment="field=sciai;appl=jax" --nodes=6 --ntasks-per-node=1 --exclusive --gres=gpu:4 -t 3-00:00:00 --exclude=node[0449-0484,0499-0576] --export=ALL,XLA_EXTRA=--xla_gpu_shard_autotuning=false scripts/train/omat_stream/pretrain.sh runs/omat_stream/$F.json)
echo "submitted fine-tuning $J  $(date '+%F %T')"
until grep -a -q -E '^epoch 1 step 2000 ' runs/slurm/oamft_$J.out 2>/dev/null; do sleep 60; squeue -j $J -h | grep -q . || break; done
squeue -j $J -o '%.9i %.3t %.8M %.5D %R'
grep -a -E 'Initialised|stream train part|steps per data epoch|LR schedule|LR at the start' runs/${F}_cue/loss_${F}_cue.out | cut -c1-230
grep -a -E '^first training|^epoch 1 step (1000|2000) ' runs/slurm/oamft_$J.out; date '+%F %T'
