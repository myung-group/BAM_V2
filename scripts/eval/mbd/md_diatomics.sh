#!/bin/bash
# Matbench Discovery MD (DynaMat v1.0, 17 NVT rollouts of 20 ps -> CMDS) and diatomics (92 curves -> CDS) with a RACE v2
# checkpoint on one node: one process per GPU for the rollouts, then the diatomic sweep, then the summaries
# (scripts/eval/mbd/md_diatomics.py; float64 inference).  Rerun-safe: finished rollouts are reused.
# usage (from the work folder, default /scratch/t589a02/BAM_V2_more/work):
#   $ sbatch -J mdbench -p normal --comment="field=sciai;appl=jax" --nodes=1 --ntasks-per-node=1 --exclusive --gres=gpu:4 -t 12:00:00 --export=ALL,CKPT=<ckpt.pkl>,TAG=<name> scripts/eval/mbd/md_diatomics.sh
#   TEST=1: MD dry run, 2,000-step rollouts (largest system followed by a small one in one process with vesin, a
#           108-atom one with the ASE neighbour list)
#   and the diatomics dry run: checks the pipeline and gives the time per MD step
#   PROFILE=1: scripts/eval/mbd/md_profile.py only (time per stage of a force call, skin-list check)
#   MBD_ENV: Python >= 3.14 environment; MBD_REPO: clone of janosh/matbench-discovery; MBD_CACHE_DIR: reference files
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd ${RACE_WORK:-/scratch/t589a02/BAM_V2_more/work}
CKPT=${CKPT:?checkpoint}; TAG=${TAG:?tag}
if [ -z "$MD_INNER" ]; then          # the batch step is bound to one core: run everything in one step with all cores
  exec srun --nodes=1 --ntasks=1 --cpus-per-task=288 env MD_INNER=1 bash scripts/eval/mbd/md_diatomics.sh
fi
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${MBD_ENV:-/home01/t589a02/.conda/envs/mbd314}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false ${XLA_EXTRA:-}"
export BAM_NEIGHBORS=${BAM_NEIGHBORS:-vesin} BAM_PREDICT_X64=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=${OMP:-8} PYTHONUNBUFFERED=1
export MBD_REPO=${MBD_REPO:-/scratch/t589a02/BAM_V2_more/ext/matbench-discovery}
export MBD_CACHE_DIR=${MBD_CACHE_DIR:-/scratch/t589a02/BAM_V2_more/ext/mbd_cache}
OUT=runs/mbd/$TAG; mkdir -p $OUT/md $OUT/diatomics
PY="python scripts/eval/mbd/md_diatomics.py"
echo "node $(hostname)  cpus $(nproc)  ckpt $CKPT  tag $TAG  test ${TEST:-0}  neighbours $BAM_NEIGHBORS  code $(git -C $(readlink -f race_v2)/.. rev-parse --short HEAD)  benchmark $(git -C $MBD_REPO rev-parse --short HEAD)  $(date '+%F %T')"
on() { local g=$1; shift; CUDA_VISIBLE_DEVICES=$g taskset -c $((g * 72))-$((g * 72 + 71)) "$@"; }
rc=0
if [ "${PROFILE:-0}" = 1 ]; then      # time per stage of one force call and the skin-list check, one GPU
  on 0 python scripts/eval/mbd/md_profile.py --ckpt $CKPT > $OUT/profile.log 2>&1 || rc=1
  grep -a -v -E '^\s*$|WARNING|cuda_timer|Nsight' $OUT/profile.log | tail -n 40 | cut -c1-300
elif [ "${TEST:-0}" = 1 ]; then
  on 0 $PY md --ckpt $CKPT --out-dir $OUT/md --dry-run > $OUT/md_dry.log 2>&1 &
  on 1 $PY md --ckpt $CKPT --out-dir $OUT/md --systems bulkLiMgAlZnSn_600K_J_Schmidt_VASP bulkMoS2_300K_NO-VdW_J.Kioseoglou_VASP --n-steps 2000 > $OUT/md_g1.log 2>&1 &
  on 2 env BAM_NEIGHBORS=ase $PY md --ckpt $CKPT --out-dir $OUT/md --systems bulkAg_600K_Kapil --n-steps 2000 > $OUT/md_g2.log 2>&1 &
  on 3 $PY diatomics --ckpt $CKPT --out-dir $OUT/diatomics --dry-run > $OUT/diatomics.log 2>&1 &
  for j in $(jobs -p); do wait $j || rc=1; done
  for f in md_dry md_g1 md_g2 diatomics; do echo "== $f"; grep -a -v -E '^\s*$|it/s\]|s/it\]' $OUT/$f.log | tail -n 40 | cut -c1-260; done
else
  # 17 systems, largest first, dealt round-robin to the 4 GPUs
  G0="bulkLiMgAlZnSn_600K_J_Schmidt_VASP naphthalene_295K_Sharma_S bulkCu_1000K_Kapil bulkMoS2_300K_NO-VdW_J.Kioseoglou_VASP anthracene_293K_Sharma_S"
  G1="bulkLiMgAlZnSn_900K_J_Schmidt_VASP CsSnI3_500K_Ivor_VASP bulkPt3Co_300K_J.Kioseoglou_VASP pentacene_295K_Sharma_S"
  G2="bulkAu_1500K_Kapil bulkCuAu_500K-Artrith_VASP TiSe2_400K_Ivor_VASP picene_295K_Sharma_S"
  G3="bulkCuZrAl_1500K_A.Wadowski-J.Schmidt_VASP bulkAg_600K_Kapil MAPbBr3_300K_Ivor_VASP tetracene_295K_Sharma_S"
  g=0
  for S in "$G0" "$G1" "$G2" "$G3"; do
    on $g $PY md --ckpt $CKPT --out-dir $OUT/md --systems $S > $OUT/md_g$g.log 2>&1 &
    g=$((g + 1))
  done
  for j in $(jobs -p); do wait $j || rc=1; done
  for g in 0 1 2 3; do echo "== GPU $g"; grep -a -E '^(calculator|model calls)|Traceback|Error' $OUT/md_g$g.log | tail -n 6 | cut -c1-300; done
  echo "rollouts exit $rc  $(date '+%F %T')"
  on 0 $PY diatomics --ckpt $CKPT --out-dir $OUT/diatomics > $OUT/diatomics.log 2>&1 || rc=1
  grep -a -v -E '^\s*$|it/s\]|s/it\]' $OUT/diatomics.log | tail -n 45 | cut -c1-260
  $PY score-md --out-dir $OUT/md > $OUT/md_score.log 2>&1 || rc=1
  grep -a -A40 -E '^\{' $OUT/md_score.log | cut -c1-260
fi
echo "exit $rc  $(date '+%F %T')"
exit $rc
