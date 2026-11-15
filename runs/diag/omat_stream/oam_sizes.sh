#!/bin/bash
# Size tables (+ row filters) of OMat24 / sAlex aselmdb splits for a given cutoff, CPU only, one node (2026-11-14).
# Same filters as smoke.sh used for omat24_*_c5: <meta>/convert_config.json + per-dataset least-squares E0 (train refs).
# env: CUT (cutoff, e.g. 7.0), SETS (space-separated list of <dataset>_<split>, e.g. "omat24_train" or "salex_val salex_train")
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J oamsz -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 05:00:00 --exclude=node[0449-0484,0499-0576] --export=ALL,CUT=7.0,SETS="omat24_train" runs/diag/omat_stream/oam_sizes.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BAM_NEIGHBORS=ase PYTHONUNBUFFERED=1
CUT=${CUT:?cutoff}; TAG=c${CUT%.0}; D=/scratch/t510a13/data; SZ=$D/oam_sizes
NCPU=$(OMP_NUM_THREADS= nproc); NW=$((NCPU > 16 ? NCPU - 8 : NCPU))
echo "node $(hostname)  cpus $NCPU  cutoff $CUT  sets ${SETS:?sets}  $(date '+%F %T')"
for S in $SETS; do
  DS=${S%_*}
  echo "== size pass $S -> ${S}_$TAG  $(date +%T)"
  python -I bam_omat24/stream_io.py sizes --source aselmdb:$D/oam_lmdb/$S --cutoff $CUT --out $SZ/${S}_$TAG \
    --workers $NW --filters $D/oam_meta/$S/convert_config.json --e0-npz $D/oam_meta/refs/${DS}_train_ls.npz 2>&1 | tail -n 4 || exit 1
  python -I -c "
import json; m = json.load(open('$SZ/${S}_$TAG/sizes_meta.json'))
print('$S kept', m['n'], 'dropped', m['dropped'], 'atoms', m['sum_atoms'], 'avg_n', round(m['avg_n_neighbors'], 3), 'max atoms/edges', m['max_atoms'], m['max_edges'])" || exit 1
done
echo "exit 0  $(date '+%F %T')"
