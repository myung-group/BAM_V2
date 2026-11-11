#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:4
#SBATCH -t 10:00:00
#SBATCH -J lipsfu_soap
#SBATCH -o runs/slurm/%x_%j.out
# LiPS Fu et al. protocol: train 19000 / valid 1000 / test 5001 (sklearn random_state=123); 4 GPUs, global batch 4
cd /scratch/t510a13/race_v2/work
bash runs/run_cue.sh lips_fu_hill_soap_s42
tail -n1 runs/lips_fu_hill_soap_s42_cue/stdout.txt
