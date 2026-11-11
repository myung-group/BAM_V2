#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH -o runs/slurm/%x_%j.out
# MD22 prep: convert npz -> extxyz (standard train sizes), then a 2-epoch timing run per molecule
# (reads s/epoch and the training-set avg_n_neighbors from the log) to fix n_epochs before the finale runs.
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
cd /scratch/t510a13/data/datasets_ext && python convert_md22.py
cd /scratch/t510a13/race_v2/work
run4() { i=0; for m in "$@"; do CUDA_VISIBLE_DEVICES=$i bash runs/run_cue.sh md22T_$m & i=$((i+1)); done; wait; }
run4 ala3 dha stach atat
run4 atatcgcg bucky dwnt
for m in ala3 dha stach atat atatcgcg bucky dwnt; do
  f=runs/md22T_${m}_cue/loss_md22T_${m}_cue.out
  echo "$m | $(grep -m1 -o 'n_graphs=.*avg_n_neighbors=[0-9.]*' $f) | $(grep -E '^[0-9/]+ [0-9:]+ +[0-9]+ ' $f | tail -1 | awk '{print "epoch",$3,"s/ep",$4}')"
done | tee runs/diag/md22_prep.log
