#!/bin/bash
# final_eval.sh <fine-tuning job id>: when the job has ended at its last epoch, snapshot the final checkpoint and run
# the float64 evaluation (3k, kappa, full WBM, CPS) with scripts/eval/mbd/eval_x64.sh; then convert it to PyTorch.
cd /scratch/t510a13/race_v2/work
J=$1; F=oam_v1set_sr32_soap_ft; L=runs/${F}_cue/loss_${F}_cue.out; K=runs/${F}_cue/ckpt_${F}_cue.last.pkl
NE=$(grep -m1 '"n_epochs"' runs/omat_stream/$F.json | tr -dc 0-9)
until ! squeue -j $J -h 2>/dev/null | grep -q .; do sleep 30; done
sacct -j $J -X -n -o JobID,State,Elapsed,End; tail -n 2 runs/slurm/oamft_$J.out | cut -c1-120
LE=$(grep -a -E '^[0-9]+/[0-9]+/[0-9]+ ' $L | tail -n 1 | awk '{print $3}')
if [ "$LE" != "$NE" ]; then echo "job ended at epoch $LE of $NE: no final evaluation started"; exit 1; fi
/scratch/t510a13/envs/bam_jax/bin/python -I scripts/eval/mbd/val_table.py $F $((NE - 3))
sleep 45; S=runs/mbd/snap/oamft_final_e$NE.pkl; command cp -p $K $S; ls -l --time-style=+%T $S | awk '{print $5,$6}'
( cd $(dirname $(readlink -f scripts))/race_v2_torch && OMP_NUM_THREADS=4 tools/tpy -m race_v2_torch.convert --ckpt /scratch/t510a13/race_v2/work/$S --out /scratch/t510a13/race_v2/work/runs/torch_port/oam/oamft_final_e$NE.pt 2>&1 | grep -v -i warn | cut -c1-160 )
bash scripts/eval/mbd/eval_x64.sh $S oamft_final_e$NE
