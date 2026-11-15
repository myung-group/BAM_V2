#!/bin/bash
# OMat24 streaming smoke on the raw aselmdb files, one whole GPU node (2026-11-11, account t510a13):
#   A. size pass + row filters of omat24_val (vesin), drop statistics;
#   B. 2 x 200 training steps, data_format stream, L5x64 recipe, on 1 GPU and then on 4 GPUs (throughput);
#   C. size pass + filters of the whole omat24_train (5300 files) with all CPUs of the node.
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J omatsmoke -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 03:00:00 scripts/train/omat_stream/smoke.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" BAM_NEIGHBORS=ase
D=/scratch/t510a13/data
SZ=$D/oam_sizes; mkdir -p $SZ
BASE=configs/omat_pretrain/omat24_100k_L5x64_screening.json
NCPU=$(nproc)
echo "node $(hostname)  cpus $NCPU  gpus $(nvidia-smi -L | wc -l)  code $(git -C $(readlink -f race_v2)/.. rev-parse --short HEAD)"
FLT="--filters $D/oam_meta/omat24_train/convert_config.json --e0-npz $D/oam_meta/refs/omat24_train_ls.npz"

echo "limits: nproc $(ulimit -u) nofile $(ulimit -n) threads-max $(cat /proc/sys/kernel/threads-max)"
echo "== A. size pass, omat24_val  $(date +%T)"
[ -f $SZ/omat24_val_c5/sizes_meta.json ] || ( export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  python -I race_v2/data/stream_io.py sizes --source aselmdb:$D/oam_lmdb/omat24_val --cutoff 5.0 \
    --out $SZ/omat24_val_c5 --workers $((NCPU < 128 ? NCPU : 128)) $FLT 2>&1 | tail -n 4 )
[ -f $SZ/omat24_val_c5/sizes_meta.json ] || { echo "ABORT: no validation size table"; exit 1; }
python -I - <<EOF || exit 1
import json, sys
m = json.load(open('$SZ/omat24_val_c5/sizes_meta.json'))
d = sum(m['dropped'].values())
print('val kept', m['n'], 'dropped', d, m['dropped'], 'avg_n', round(m['avg_n_neighbors'], 3), 'max atoms', m['max_atoms'])
sys.exit(0 if m['n'] > 0 and d < 0.2 * (m['n'] + d) else 1)
EOF

echo "== B. training smoke, data_format stream (train = validation files here)  $(date +%T)"
for G in 1 4; do
  N=omat_smoke_g$G
  python -I scripts/train/omat_stream/make_config.py $BASE runs/omat_stream/$N.json --name $N \
    --train aselmdb:$D/oam_lmdb/omat24_val --train-sizes $SZ/omat24_val_c5 \
    --valid aselmdb:$D/oam_lmdb/omat24_val --valid-sizes $SZ/omat24_val_c5 \
    --steps-per-epoch 200 --epochs 2 --workers $((24 * G)) --valid-max 1000 --set lr_warmup_epochs=0 || exit 1
  ( export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    [ $G = 1 ] && export CUDA_VISIBLE_DEVICES=0
    /usr/bin/time -f "WALL %e s (G=$G)" python -m race_v2.training.train runs/omat_stream/$N.json 2>&1 | grep -v -E '^\s*\^+\s*$' | tail -n 25 )
  echo "-- log G=$G"; grep -E 'data parallel|data_format stream|bucket|[Ee]poch' runs/${N}_cue/loss_${N}_cue.out | tail -n 12
done

echo "== C. size pass, omat24_train (all files)  $(date +%T)"
( export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  python -I race_v2/data/stream_io.py sizes --source aselmdb:$D/oam_lmdb/omat24_train --cutoff 5.0 \
    --out $SZ/omat24_train_c5 --workers $((NCPU > 16 ? NCPU - 8 : NCPU)) $FLT 2>&1 | tail -n 8 )
python -I -c "
import json; m = json.load(open('$SZ/omat24_train_c5/sizes_meta.json'))
print('train kept', m['n'], 'dropped', m['dropped'], 'atoms', m['sum_atoms'], 'avg_n', round(m['avg_n_neighbors'], 3), 'max atoms/edges', m['max_atoms'], m['max_edges'])"
echo "done $(date +%T)"
