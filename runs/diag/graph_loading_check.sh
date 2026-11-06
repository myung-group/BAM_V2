#!/bin/bash
# Check of the pickle-free graph loading (2026-11-06): tests, then the size table of the 100k OMat24 subset with the
# CPUs of a GPU node (no GPU work).  usage: sbatch -J graphchk -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 \
#   -t 00:30:00 runs/diag/graph_loading_check.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m pytest -q tests/test_graph_loading.py 2>&1 | tail -n 3
D=/scratch/t589a02/datasets_ext/oam_subsets_100k
python -I bam_omat24/graph_sizes.py $D/omat24_train.xyz $D/omat24_valid.xyz --cutoff 5.0 --workers 64 \
  --out-dir runs/diag/sizes_oam100k
