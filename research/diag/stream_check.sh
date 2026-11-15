#!/bin/bash
# Streaming (pickle-free) data pipeline check (2026-11-06, written while the allocation was exhausted; not run yet):
# all tests of tests/test_stream_io.py (incl. the JAX ones), the graph-loading tests, then the size pass of the 100k
# OMat24 subset with 64 workers on the CPUs of one GPU node (no GPU work).
# usage: sbatch -J streamchk -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 00:40:00 research/diag/stream_check.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
python -m pytest -q --noconftest tests/test_stream_io.py tests/test_graph_loading.py 2>&1 | tail -n 4
D=/scratch/t510a13/data/datasets_ext/oam_subsets_100k
python -I race_v2/data/stream_io.py sizes --source extxyz:$D/omat24_train.xyz --source extxyz:$D/omat24_valid.xyz \
  --cutoff 5.0 --out research/diag/stream_sizes_oam100k --workers 64
