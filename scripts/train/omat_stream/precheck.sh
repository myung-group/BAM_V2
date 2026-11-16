#!/bin/bash
# Checks before the chained run (chain_oam.sh) on a new account, one GPU of one node:
#   A. the raw aselmdb files against the size tables (scripts/data/remap_size_tables.py --check --count): same files in
#      the same order, rows of the files = kept + dropped rows of the table, table keys present in the files;
#   B. the unit tests (pytest tests).
# usage (from the work folder, default /scratch/t589a02/BAM_V2_more/work):
#   $ sbatch -J precheck -p normal --comment="field=sciai;appl=jax" --nodes=1 --ntasks-per-node=1 --gres=gpu:1 -t 02:00:00 --export=ALL,TABLES="<a b ...>",MAPS="OLD=NEW OLD=NEW" scripts/train/omat_stream/precheck.sh
#   (TABLES and MAPS are space-separated: a comma would end the variable in --export)
#   SIZES_SRC / SIZES_OUT: original and remapped size-table folders;  SKIP_TESTS=1: part A only;  SKIP_TABLES=1: part B
#   only;  PYTEST_ARGS: pytest targets and options (default: tests), space-separated
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd ${RACE_WORK:-/scratch/t589a02/BAM_V2_more/work}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/home01/t589a02/.conda/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false ${XLA_EXTRA:-}" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
REPO=$(dirname $(readlink -f race_v2))
SRC=${SIZES_SRC:-/scratch/t589a02/BAM_V2_more/prep_data/oam/oam_sizes}; OUT=${SIZES_OUT:-/scratch/t589a02/BAM_V2_more/oam_sizes}
echo "node $(hostname)  cpus $(nproc)  code $(git -C $REPO rev-parse --short HEAD)  $(date '+%F %T')"
# jax.devices() first: the JAX CUDA plugin loads libnvrtc, which libcue_ops needs
python -c "import jax; d = jax.devices(); import flax, e3nn_jax, cuequivariance_jax, lmdb, ase; print('jax', jax.__version__, 'devices', d)"
RA=0
if [ "${SKIP_TABLES:-0}" != 1 ]; then
  echo "== A. raw files against the size tables  $(date +%T)"
  M=""; for m in ${MAPS:?OLD=NEW path maps}; do M="$M --map $m"; done
  python -I scripts/data/remap_size_tables.py --src $SRC --out $OUT $M ${TABLES:+--tables ${TABLES// /,}} --check --count --workers 16; RA=$?
  echo "size-table check exit $RA  $(date +%T)"
fi
RB=0
if [ "${SKIP_TESTS:-0}" != 1 ]; then
  echo "== B. unit tests  $(date +%T)"
  PL=$PWD/runs/slurm/pytest_${SLURM_JOB_ID:-local}.log
  ( cd $REPO && OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m pytest ${PYTEST_ARGS:-tests} -q -p no:cacheprovider -W ignore::DeprecationWarning > $PL 2>&1 ); RB=$?
  grep -a -n -E '^(FAILED|ERROR)|^=+ .* in [0-9.]+s' $PL | tail -n 40; echo "full pytest output: $PL"
  ( cd $REPO && git status --short | head -n 20 )
fi
echo "exit $RA $RB  $(date '+%F %T')"
[ $RA = 0 ] && [ $RB = 0 ]
