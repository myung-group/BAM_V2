#!/bin/bash
# MPtrj preparation on one node (CPU work only; 2026-11-11): raw JSON -> lmdb shards + E0 fit (convert_mptrj.py),
# then the size tables of the training and validation splits with the row filters.
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J mptrjprep -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 03:00:00 scripts/data/omat_stream/mptrj_prep.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BAM_NEIGHBORS=ase PYTHONUNBUFFERED=1
D=/scratch/t510a13/data; OUT=$D/mptrj_lmdb; SZ=$D/oam_sizes
echo "node $(hostname)  cpus $(OMP_NUM_THREADS= nproc)  mem $(free -g | awk '/Mem:/{print $2}') GB  $(date +%T)"
if [ ! -f $OUT/convert_mptrj.json ]; then
  python -I scripts/data/convert_mptrj.py $D/mptrj_raw/MPtrj_2022.9_full.json $OUT || exit 1
fi
ls -l $OUT/train | head -n 4; du -sh $OUT
# row filters: the converter thresholds used for OMat24 (force 50 eV/A, stress 80 GPa, 512 atoms, E/atom > 0),
# residual floor -5 eV/atom against the E0 fitted above
python -I - <<PY
import json
m = json.load(open('$OUT/convert_mptrj.json'))
json.dump({'max_force': 50.0, 'max_stress': 0.49932073007066063, 'max_atoms': 512, 'max_energy_per_atom': 0.0,
           'min_energy_residual': -5.0, 'e0': m['e0_ls_train']}, open('$OUT/filters.json', 'w'), indent=1)
PY
for S in val train; do
  echo "== size pass $S  $(date +%T)"
  python -I race_v2/data/stream_io.py sizes --source aselmdb:$OUT/$S --cutoff 5.0 --out $SZ/mptrj_${S}_c5 \
    --workers 250 --filters $OUT/filters.json 2>&1 | tail -n 6
done
echo "done $(date +%T)"
