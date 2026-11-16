#!/bin/bash
# Full WBM evaluation on one node: relaxation of all 256,963 structures (race_v2/inference/relax_wbm.py, one process per
# GPU, each bound to its own 72 cores; rerun-safe: finished structures are skipped), then the discovery CSV +
# metrics (score_sample.sh) and the geometry RMSD (rmsd_sample.py) inside the same job -- scoring 257k entries is
# too heavy for a login node.
# Several nodes: NSH = 4 x nodes, one job per node with its four shards and STAGE=relax, then one job with STAGE=score
# and --dependency=afterok:<relax jobs> (score and RMSD run side by side there).
# usage (from the work folder, default /scratch/t589a02/BAM_V2_more/work):
#   one node:  $ sbatch -J wbmfull -p normal --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 03:00:00 --export=ALL,CKPT=<ckpt.pkl>,TAG=wbm_<name> scripts/eval/mbd/eval_wbm.sh
#   two nodes: ... --export=ALL,CKPT=..,TAG=..,NSH=8,SHARDS="0 1 2 3",STAGE=relax  and  ..,NSH=8,SHARDS="4 5 6 7",STAGE=relax ; then ..,STAGE=score
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd ${RACE_WORK:-/scratch/t589a02/BAM_V2_more/work}
CKPT=${CKPT:?checkpoint}; TAG=${TAG:?tag}; NSH=${NSH:-4}; SHARDS=${SHARDS:-"0 1 2 3"}; STAGE=${STAGE:-all}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/home01/t589a02/.conda/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false ${XLA_EXTRA:-}" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
W=${WBM_DIR:-/scratch/t589a02/mp_data/wbm}
OUT=runs/mbd/$TAG/relaxed; mkdir -p $OUT
echo "node $(hostname)  ckpt $CKPT  tag $TAG  stage $STAGE  shards ($SHARDS)/$NSH  code $(git -C $(readlink -f race_v2)/.. rev-parse --short HEAD)  threads limit $(ulimit -u)  $(date '+%F %T')"
T0=$(date +%s); rc=0
if [ $STAGE != score ]; then
  g=0
  for s in $SHARDS; do                      # the g-th shard of this job on GPU g with cores 72 g .. 72 g + 71
    CUDA_VISIBLE_DEVICES=$g taskset -c $((g * 72))-$((g * 72 + 71)) python -m race_v2.inference.relax_wbm --ckpt $CKPT \
      --input $W/2024-08-04-wbm-initial-atoms.extxyz.zip --out-dir $OUT --shard $s --num-shards $NSH \
      --max-structures 64 --groups 2 --host-workers 32 >> $OUT/relax_${s}of${NSH}.log 2>&1 &
    g=$((g + 1))
  done
  for j in $(jobs -p); do wait $j || rc=1; done
  for s in $SHARDS; do echo "== shard $s"; grep -E '^\[relax\]|\[batched\] FINAL|Traceback|Error' $OUT/relax_${s}of${NSH}.log | tail -n 4 | cut -c1-330; done
  echo "relaxation exit $rc, $(( $(date +%s) - T0 )) s  $(date '+%F %T')"
fi
if [ $STAGE != relax ]; then
  T0=$(date +%s)
  ${PJ:-/home01/t589a02/.conda/envs/bam_jax/bin/python} -I scripts/eval/mbd/rmsd_sample.py --relaxed $OUT --mp2020-cse $W/2022-10-19-wbm-computed-structure-entries.jsonl.gz \
    --summary $W/2023-12-13-wbm-summary.csv.gz --output runs/mbd/$TAG/rmsd.csv.gz > runs/mbd/$TAG/rmsd.log 2>&1 &
  bash scripts/eval/mbd/score_sample.sh $TAG
  echo "scored in $(( $(date +%s) - T0 )) s  $(date '+%F %T')"
  wait; tail -n 4 runs/mbd/$TAG/rmsd.log
  echo "RMSD in $(( $(date +%s) - T0 )) s  $(date '+%F %T')"
fi
echo "exit $rc  $(date '+%F %T')"
