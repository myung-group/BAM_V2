#!/bin/bash
# T6 streaming throughput smoke on HANGANG (2026-11-09, docs/HANDOFF_stream_loader.md section 6 item 3), one GPU node:
#   1. the stream / vesin tests on CPU;
#   2. loader only (no model) on the OMat24 100k training subset read from Lustre: ASE neighbour list W=8, then
#      vesin W=2,4,8,16 (500 steps each); the size table is made on the way (vesin, all CPUs);
#   3. 2 x 250 training steps of the sz100k L5x128 amsgrad config with data_format stream (vesin, 8 workers):
#      epoch 2 time = steady state, against the in-memory run of the same config when its log is present.
# needs once in bam_cue:  pip install --no-deps vesin lmdb
# usage: sbatch -J t6stream -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 01:00:00 runs/diag/stream_t6/t6.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
python -c "import vesin, lmdb" || { echo "ABORT: run 'pip install --no-deps vesin lmdb' in bam_cue first"; exit 1; }
echo "python $(which python)  git $(git rev-parse --short HEAD)  node $(hostname)  cpus $(nproc)"
D=/scratch/t589a02/datasets_ext/oam_subsets_100k
BASE=input/input_3bpa_omat24_sz100k_L5x128amsgrad_omat24_v2_s42_cue.json
O=runs/stream_t6; mkdir -p $O
SZ=$O/sizes_omat24_train
CUT=$(python -c "import json; print(json.load(open('$BASE')).get('cutoff', 5.0))")
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "== 1. tests"
JAX_PLATFORMS=cpu python -m pytest -q tests/test_stream_io.py tests/test_stream_train.py tests/test_neighbors_vesin.py \
  2>&1 | tail -n 4

echo "== 2. loader only"
LB="python -I runs/diag/stream_t6/loader_bench.py --source extxyz:$D/omat24_train.xyz --cutoff $CUT --sizes $SZ \
  --atomic-numbers-from $BASE --graphs-per-pack 16"
( export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  BAM_NEIGHBORS=vesin $LB --workers 2,4,8,16 --steps 500 --out $O/loader_vesin.json 2>&1 | grep -E '^W=|size pass'
  BAM_NEIGHBORS=ase $LB --workers 8 --steps 300 --out $O/loader_ase.json 2>&1 | grep '^W=' )

echo "== 3. training, data_format stream"
python -I runs/diag/stream_t6/make_config.py $BASE $O/oam100k_stream.json --run-dir $O/train --sizes $SZ \
  --steps-per-epoch 250 --epochs 2 --workers 8 --valid-max 1000 || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
BAM_NEIGHBORS=vesin /usr/bin/time -f "WALL %e s" python -m bam_omat24.train_3bpa $O/oam100k_stream.json \
  > $O/train/stdout.txt 2>&1
tail -n 2 $O/train/stdout.txt
grep -A1 "^  bucket\|steps per data epoch" $O/train/loss.out | head -n 6
grep -E "^[0-9]{2}/[0-9]{2}/[0-9]{4} .* [12] +[0-9.]+ +\|" $O/train/loss.out

echo "== reference: in-memory run of the same config (if present)"
REF=$(python -c "import json; print(json.load(open('$BASE'))['log_file'])")
if [ -f "$REF" ]; then
  grep -m1 "batches/epoch" $REF
  grep -E "^[0-9]{2}/[0-9]{2}/[0-9]{4} .* [2-4] +[0-9.]+ +\|" $REF | head -n 3
else
  echo "no $REF"
fi
echo T6_DONE
