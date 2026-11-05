#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:1
#SBATCH -t 00:01:00
#SBATCH -J skip_lips_lc_adam
#SBATCH -o runs/slurm/%x_%j.out
# disabled 2026-10-30 (user: run only the best recipe, A+hill + SOAP)
exit 0
