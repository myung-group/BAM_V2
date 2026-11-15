#!/bin/bash
# Size tables of the MPtrj lmdb shards for another cutoff (CPU only, one node; 2026-11-13).  Same filters as
# mptrj_prep.sh.  usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J mptrjsz -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 01:30:00 --exclude=node[0449-0484,0499-0576] --export=ALL,CUT=7.0 scripts/data/omat_stream/mptrj_sizes.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BAM_NEIGHBORS=ase PYTHONUNBUFFERED=1
CUT=${CUT:?cutoff}; TAG=c${CUT%.0}; D=/scratch/t510a13/data; OUT=$D/mptrj_lmdb; SZ=$D/oam_sizes
echo "node $(hostname)  cpus $(OMP_NUM_THREADS= nproc)  cutoff $CUT  $(date '+%F %T')"
for S in train val; do        # train first: 32 files in parallel; val is one file (one task)
  echo "== size pass $S  $(date +%T)"
  python -I race_v2/data/stream_io.py sizes --source aselmdb:$OUT/$S --cutoff $CUT --out $SZ/mptrj_${S}_$TAG \
    --workers 250 --filters $OUT/filters.json 2>&1 | tail -n 3 &
done
wait
echo "done $(date '+%F %T')"
