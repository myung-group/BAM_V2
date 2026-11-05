#!/bin/bash
# unit test of the species-Adam SOAP option on a GPU node (no JAX on login nodes), then nothing else
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
python -m pytest -q tests/test_soap_grouped.py 2>&1 | tail -n 5
