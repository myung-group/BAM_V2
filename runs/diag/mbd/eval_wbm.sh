#!/bin/bash
# Full WBM evaluation on one node: relaxation (relax_wbm.sh, one process per GPU; rerun-safe) and then the discovery
# CSV + metrics (score_sample.sh) inside the same job -- scoring 257k entries is too heavy for a login node.
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J wbmfull -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 02:00:00 --export=ALL,CKPT=<ckpt.pkl>,TAG=wbm_<name>,NSH=4,SHARDS="0 1 2 3" runs/diag/mbd/eval_wbm.sh
#SBATCH -o runs/slurm/%x_%j.out
cd /scratch/t510a13/race_v2/work
bash runs/diag/mbd/relax_wbm.sh
T0=$(date +%s)
bash runs/diag/mbd/score_sample.sh ${TAG:?tag}
echo "scored in $(( $(date +%s) - T0 )) s  $(date '+%F %T')"
