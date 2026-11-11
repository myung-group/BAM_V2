#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH -o runs/slurm/%x_%j.out
# Build QM9 U/H/G/ZPVE/Cv extxyz files (same split as U0) from the raw tarball (single-stream read).
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
cd /scratch/t510a13/data/datasets_ext
python -I convert_qm9_props.py qm9_src/dsgdb9nsd.xyz.tar.bz2 qm9 . | tee /scratch/t510a13/race_v2/work/runs/diag/qm9_props_prep.log
