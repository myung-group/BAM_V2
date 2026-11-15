#!/bin/bash
#SBATCH -J dataset_prep
#SBATCH -p exclusive06
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -o /scratch/t510a13/logs/%x_%j.out
#
# Size tables (data_format "stream", cutoff CUT) and per-element E0 of one aselmdb dataset (OMol25 full, OPoly26,
# OC25, ...), for the next-window runs (2026-11-15).  CPU work on one GPU-node quarter (72 cores, GPU unused).
#   NAME   short name; outputs in $OUT (default /appsdata/t510a13/prep/<NAME>_c<CUT>)
#   TRAIN  directory of the training aselmdb files (all of them, or the first SHARDS for a test)
#   VALID  directory (or one .aselmdb file) of the validation split (optional)
#   E0_ROWS rows per file for the E0 fit (default 20000: a sample, enough for least squares)
#   E0     1: fit per-element E0 on the training files (one table per level of theory: OMol25 and OPoly26 share
#          wB97M-V, OC25 is RPBE+D3, MPtrj PBE/PBE+U -- never one E0 for all)
# NOT yet run: a new job; test first with SHARDS=4.
# usage (from /scratch/t510a13):
#   NAME=opoly26 TRAIN=<dir> VALID=<dir> [SHARDS=4] sbatch --exclude=node[0449-0484,0499-0576] jobs/dataset_prep.sh
set -euo pipefail
ENV=/scratch/t510a13/envs/bam_cue
REPO=${REPO:-/scratch/t510a13/BAM_V2_wt_thole}
JOBS=/scratch/t510a13/jobs
CUT=${CUT:-6.0}
NAME=${NAME:?NAME needed}; TRAIN=${TRAIN:?TRAIN needed}
OUT=${OUT:-/appsdata/t510a13/prep/${NAME}_c${CUT}}
echo "host $(hostname)  arch $(uname -m)  job ${SLURM_JOB_ID:-none}  cpus $(nproc)  $(date '+%F %T')  $NAME"
module load miniconda3/py314_26.7.1-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
mapfile -t ALL < <(ls "$TRAIN"/*.aselmdb)
[ "${#ALL[@]}" -gt 0 ] || { echo "ERROR: no .aselmdb in $TRAIN"; exit 1; }
N=${SHARDS:-${#ALL[@]}}; [ "$N" -le "${#ALL[@]}" ] || N=${#ALL[@]}
[ -n "${SHARDS:-}" ] && OUT=${OUT}_test$N
echo "train files: ${#ALL[@]} found, using $N; out $OUT"
mkdir -p "$OUT"
cd "$REPO"
# neighbour backend: the default (matscipy); vesin starts one thread per core in every worker (EAGAIN, job 1257426)
export BAM_REPO=$REPO OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
SRC=()
for f in "${ALL[@]:0:$N}"; do SRC+=(--source "aselmdb:$f"); done
sizes() {   # <name> <source args...>
    local name=$1; shift
    if [ -f "$OUT/sizes_$name/sizes_meta.json" ]; then
        echo "sizes_$name: already there"
    else
        python -I race_v2/data/stream_io.py sizes "$@" --cutoff $CUT --out "$OUT/sizes_$name" --workers 64 | tail -n 2
    fi
}
echo "== size tables  $(date '+%T')"
sizes train "${SRC[@]}"
[ -n "${VALID:-}" ] && sizes valid --source "aselmdb:$VALID"
if [ "${E0:-1}" = 1 ]; then
    echo "== E0 (training files only)  $(date '+%T')"
    python -I "$JOBS/omol4m_e0.py" --sizes "$OUT/sizes_train" --out "$OUT" --workers 64 --max-rows ${E0_ROWS:-20000}
fi
ls -la "$OUT"
echo "DATASET PREP OK $NAME  $(date '+%F %T')"
