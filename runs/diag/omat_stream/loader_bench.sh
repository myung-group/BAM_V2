#!/bin/bash
# Loader-only throughput on the MPtrj training shards (no model, no GPU work; runs/diag/stream_t6/loader_bench.py),
# to see whether the trainer (12,282 steps in 52 min = 250 structures/s per process, L5x128) is bound by the loader.
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J loaderbench -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 00:20:00 runs/diag/omat_stream/loader_bench.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export BAM_NEIGHBORS=ase OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
D=/scratch/t510a13/data
echo "node $(hostname)  $(date +%T)"
python -I runs/diag/stream_t6/loader_bench.py --source aselmdb:$D/mptrj_lmdb/train --cutoff 5.0 \
  --sizes $D/oam_sizes/mptrj_train_c5 --atomic-numbers-from runs/omat_stream/mptrj_L5x128_s42.json \
  --workers 8,24,48,96 --steps 300 --graphs-per-pack 16 --out runs/mbd/loader_bench_mptrj.json 2>&1 | grep -E '^W=|mean_|pad_' 
echo "done $(date +%T)"
