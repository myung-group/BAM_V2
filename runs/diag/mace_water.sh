#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=pytorch"
#SBATCH --gres=gpu:2
#SBATCH -t 23:00:00
#SBATCH -J mace_water
#SBATCH -o runs/slurm/%x_%j.out
# MACE baseline (official mace-torch 0.3.17, docs liquid-water example settings = paper's 64-0 model) on OUR water splits:
#   split1 = random seed-42 split (train_only 1290 / valid 144 / test 159); split2 = documented first-1393 / last-200
cd /scratch/t589a02/BAM_V2/runs/mace_water
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mace
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
W=/scratch/t589a02/datasets_ext/water_xyz; S=/scratch/t589a02/datasets_ext/water_split1393
COMMON="--model MACE --num_interactions 2 --num_channels 64 --max_L 0 --correlation 3 --r_max 6.0 --E0s average \
 --energy_key energy --forces_key forces --energy_weight 10 --forces_weight 1000 --batch_size 2 --valid_batch_size 4 \
 --max_num_epochs 800 --swa --start_swa 400 --scheduler_patience 15 --patience 30 --eval_interval 4 --ema \
 --error_table PerAtomRMSE --default_dtype float64 --device cuda --seed 123 --restart_latest --save_cpu"
(CUDA_VISIBLE_DEVICES=0 mace_run_train --name mace64_split1 --train_file $W/train_only.xyz --valid_file $W/valid.xyz --test_file $W/test.xyz $COMMON > split1.out 2>&1; echo "exit $?" >> split1.out) &
(CUDA_VISIBLE_DEVICES=1 mace_run_train --name mace64_split2 --train_file $S/train_first1393.xyz --valid_fraction 0.05 --test_file $S/test_last200.xyz $COMMON > split2.out 2>&1; echo "exit $?" >> split2.out) &
wait
tail -n 25 split1.out split2.out
