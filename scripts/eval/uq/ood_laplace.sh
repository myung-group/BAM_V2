#!/bin/bash
# OOD detection with the SOAP-eigenbasis Laplace posterior of a RACEv2 checkpoint (scripts/eval/uq/ood_laplace.py):
# GGN curvature on MPtrj + sAlex training structures, then sigE / sigF / sigFmax on ID sets (MPtrj val, sAlex val) and
# OOD sets (OMol25, OPoly26, OC25).  One process on one GPU.
# usage (from /scratch/t510a13/race_v2/work):
#   smoke test:
#   $ sbatch -J oodtest -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 --exclude=node[0449-0484,0499-0576] -t 00:40:00 --export=ALL,CKPT=<ckpt.pkl>,TAG=<name>,NFIT=32,NEVAL=48,SAMPLES=4 scripts/eval/uq/ood_laplace.sh
#   full run: no NFIT / NEVAL / SAMPLES (400 / 1000 / 16), -t 03:00:00
# optional: SETS (separate with ':' inside --export, e.g. SETS=mptrj_val:omol25), MAXATOMS, LAM (a number or grid),
#           PROBES, REUSE_NU=1 (read runs/uq/$TAG/ggn_nu.npz of an earlier run), EXTRA (further ood_laplace.py options),
#           WBMDIR (directory with relaxed_*of*.jsonl: adds the set 'wbm' and its error analysis from
#           $WBMDIR/../discovery.csv.gz), NWBM (WBM structures, default all)
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
CKPT=${CKPT:?checkpoint}; TAG=${TAG:?tag}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=${OPENBLAS:-8} OMP_NUM_THREADS=${OMP:-48} PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
OUT=runs/uq/$TAG; mkdir -p $OUT
echo "node $(hostname)  ckpt $CKPT  out $OUT  nfit ${NFIT:-400}  neval ${NEVAL:-1000}  samples ${SAMPLES:-16}  lam ${LAM:-default}  sets ${SETS:-all}  code $(git -C $(dirname $(readlink -f race_v2)) rev-parse --short HEAD)  $(date '+%F %T')"
rc=0
python scripts/eval/uq/ood_laplace.py --ckpt $CKPT --out $OUT --n-fit ${NFIT:-400} --n-eval ${NEVAL:-1000} \
  --samples ${SAMPLES:-16} --max-atoms ${MAXATOMS:-200} ${PROBES:+--ggn-probes $PROBES} ${LAM:+--lam $LAM} \
  ${SETS:+--sets ${SETS//:/,}} ${WBMDIR:+--wbm-relaxed $WBMDIR} ${NWBM:+--n-wbm $NWBM} ${REUSE_NU:+--reuse-nu} \
  ${EXTRA:-} 2>&1 | tee $OUT/run.log
rc=${PIPESTATUS[0]}
echo "exit $rc  $(date '+%F %T')"
