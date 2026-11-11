#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:4
#SBATCH -t 23:00:00
#SBATCH -J qm9_u0f
#SBATCH -o runs/slurm/%x_%j.out
# QM9 U0, standard 110k / 10k / 10,831 split (uncharacterized removed), energy + F=0 stationarity loss (force_weight 1e-3), 4 GPUs (global batch 32)
cd /scratch/t510a13/race_v2/work
bash runs/run_cue.sh qm9_u0f_hill_soap_s42
tail -n1 runs/qm9_u0f_hill_soap_s42_cue/stdout.txt
