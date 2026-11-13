#!/bin/bash
# Full WBM evaluation on one node: relaxation of all 256,963 structures (bam_omat24/relax_wbm.py, one process per
# GPU, each bound to its own 72 cores; rerun-safe: finished structures are skipped), then the discovery CSV +
# metrics (score_sample.sh) and the geometry RMSD (rmsd_sample.py) inside the same job -- scoring 257k entries is
# too heavy for a login node.
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J wbmfull -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 03:00:00 --exclude=node[0449-0484,0499-0576] --export=ALL,CKPT=<ckpt.pkl>,TAG=wbm_<name> runs/diag/mbd/eval_wbm.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
CKPT=${CKPT:?checkpoint}; TAG=${TAG:?tag}; NSH=4
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
W=/scratch/t510a13/probabilistic_symmetrization/bundle/scratch/t589a02/mp_data/wbm
OUT=runs/mbd/$TAG/relaxed; mkdir -p $OUT
echo "node $(hostname)  ckpt $CKPT  tag $TAG  code $(git -C $(readlink -f bam_omat24)/.. rev-parse --short HEAD)  threads limit $(ulimit -u)  $(date '+%F %T')"
T0=$(date +%s); rc=0
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g taskset -c $((g * 72))-$((g * 72 + 71)) python -m bam_omat24.relax_wbm --ckpt $CKPT \
    --input $W/2024-08-04-wbm-initial-atoms.extxyz.zip --out-dir $OUT --shard $g --num-shards $NSH \
    --max-structures 64 --groups 2 --host-workers 32 > $OUT/relax_${g}of${NSH}.log 2>&1 &
done
for j in $(jobs -p); do wait $j || rc=1; done
for g in 0 1 2 3; do echo "== shard $g"; grep -E '^\[relax\]|\[batched\] FINAL|Traceback|Error' $OUT/relax_${g}of${NSH}.log | tail -n 4 | cut -c1-330; done
echo "relaxation exit $rc, $(( $(date +%s) - T0 )) s  $(date '+%F %T')"
T0=$(date +%s)
bash runs/diag/mbd/score_sample.sh $TAG
echo "scored in $(( $(date +%s) - T0 )) s  $(date '+%F %T')"
T0=$(date +%s)
/scratch/t510a13/envs/bam_jax/bin/python -I runs/diag/mbd/rmsd_sample.py --relaxed $OUT --mp2020-cse $W/2022-10-19-wbm-computed-structure-entries.jsonl.gz \
  --summary $W/2023-12-13-wbm-summary.csv.gz --output runs/mbd/$TAG/rmsd.csv.gz 2>&1 | tail -n 4
echo "RMSD in $(( $(date +%s) - T0 )) s  $(date '+%F %T')"
echo "exit $rc  $(date '+%F %T')"
