#!/bin/bash
# Size table (+ row filters) of a GROUP of subsets of one OMat24/sAlex split, so the 100M-structure omat24_train table
# can be made on several nodes in parallel (2026-11-14; at 7 A one node needs ~7 h).  Same filters as oam_sizes.sh.
# The training config then lists the groups as stream.train.parts (train_3bpa._stream_parts), one size table each.
# env: CUT (cutoff), SPLIT (e.g. omat24_train), SUBS (space-separated subset dirs of that split), NAME (output table name)
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J oamszg -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 04:00:00 --exclude=node[0449-0484,0499-0576] --export=ALL,CUT=7.0,SPLIT=omat24_train,SUBS="rattled-300 rattled-300-subsampled",NAME=omat24_train_g1_c7 runs/diag/omat_stream/oam_sizes_group.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BAM_NEIGHBORS=ase PYTHONUNBUFFERED=1
CUT=${CUT:?cutoff}; D=/scratch/t510a13/data; SZ=$D/oam_sizes; SPLIT=${SPLIT:?split}; NAME=${NAME:?name}; DS=${SPLIT%_*}
NCPU=$(OMP_NUM_THREADS= nproc); NW=$((NCPU > 16 ? NCPU - 8 : NCPU))
SRC=""; for s in ${SUBS:?subsets}; do [ -d $D/oam_lmdb/$SPLIT/$s ] || { echo "missing $D/oam_lmdb/$SPLIT/$s"; exit 1; }; SRC="$SRC --source aselmdb:$D/oam_lmdb/$SPLIT/$s"; done
echo "node $(hostname)  cpus $NCPU  cutoff $CUT  $SPLIT: $SUBS -> $NAME  $(date '+%F %T')"
python -I bam_omat24/stream_io.py sizes $SRC --cutoff $CUT --out $SZ/$NAME \
  --workers $NW --filters $D/oam_meta/$SPLIT/convert_config.json --e0-npz $D/oam_meta/refs/${DS}_train_ls.npz 2>&1 | tail -n 4 || exit 1
python -I -c "
import json; m = json.load(open('$SZ/$NAME/sizes_meta.json'))
print('$NAME kept', m['n'], 'dropped', m['dropped'], 'atoms', m['sum_atoms'], 'avg_n', round(m['avg_n_neighbors'], 3), 'max atoms/edges', m['max_atoms'], m['max_edges'])" || exit 1
echo "exit 0  $(date '+%F %T')"
