#!/bin/bash
# OAM subset screening, step 1 (2026-11-05): build the three subsets, run the new tests, then 4 epochs of RACEv2
# (v2) and V1 on each dataset (pad_mode fixed: one padded shape per run) to measure time per epoch and check the stress loss.  6 short runs on 3 GPUs.
# usage: sbatch -J oam_smoke -p normal --comment="field=sciai;appl=jax" --gres=gpu:3 -t 03:00:00 runs/diag/oam_smoke.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
OUT=/scratch/t589a02/datasets_ext/oam_subsets
N_TRAIN=${N_TRAIN:-10000}
set -x
if [ ! -f $OUT/salex_meta.json ] || [ ! -f $OUT/mptrj_meta.json ] || [ ! -f $OUT/omat24_meta.json ]; then
python -I bam_omat24/oam_subset.py --name omat24 --train-root /scratch/t589a02/oam_data/omat24_train \
  --eval-root /scratch/t589a02/oam_data/omat24_val --atom-ref /scratch/t589a02/oam_data/refs/omat24_train_ls.npz \
  --n-train $N_TRAIN --n-valid 1000 --n-test 1000 --out $OUT > $OUT.omat24.log 2>&1 &
python -I bam_omat24/oam_subset.py --name mptrj --train-root /scratch/t589a02/mp_data/jax_data_v3_raw/train \
  --eval-root /scratch/t589a02/mp_data/jax_data_v3_raw/valid --atom-ref /scratch/t589a02/oam_data/refs/mptrj_raw_ls.npz \
  --n-train $N_TRAIN --n-valid 1000 --n-test 1000 --out $OUT > $OUT.mptrj.log 2>&1 &
python -I bam_omat24/oam_subset.py --name salex --train-root /scratch/t589a02/oam_data/salex_train \
  --eval-root /scratch/t589a02/oam_data/salex_val --atom-ref /scratch/t589a02/oam_data/refs/salex_train_ls.npz \
  --n-train $N_TRAIN --n-valid 1000 --n-test 1000 --out $OUT > $OUT.salex.log 2>&1 &
wait
fi
tail -n 4 $OUT.*.log
for d in omat24 mptrj salex; do [ -f $OUT/${d}_meta.json ] || { echo "ABORT: subset $d missing"; exit 1; }; done
python -I runs/diag/make_oam_configs.py $OUT --tag ${TAG:-oamfix} --epochs ${EPOCHS:-4} || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/t589a02/jax_cache_racev2
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -x tests/test_oam_stress.py tests/test_periodic.py 2>&1 | tail -n 15
set +x
# training-step NaN hunt (SOAP vs AMSGrad, 60 real fixed-shape steps) on two datasets, then the short trainings
CUDA_VISIBLE_DEVICES=0 python runs/diag/oam_step_debug.py input/input_3bpa_omat24_${TAG:-oamfix}_salex_v2_s42_cue.json 60 2>&1 | grep -E "^\[|pads|done|Error|error" > runs/slurm/${TAG:-oamfix}_stepdebug_salex.txt &
CUDA_VISIBLE_DEVICES=1 python runs/diag/oam_step_debug.py input/input_3bpa_omat24_${TAG:-oamfix}_omat24_v2_s42_cue.json 60 2>&1 | grep -E "^\[|pads|done|Error|error" > runs/slurm/${TAG:-oamfix}_stepdebug_omat24.txt &
wait
cat runs/slurm/${TAG:-oamfix}_stepdebug_*.txt
# EPOCHS (default 4) each: v2 on GPU 0-2 in parallel, then v1
i=0; for d in omat24 mptrj salex; do CUDA_VISIBLE_DEVICES=$i bash runs/run_cue.sh ${TAG:-oamfix}_${d}_v2_s42 & i=$((i+1)); done; wait
i=0; for d in omat24 mptrj salex; do CUDA_VISIBLE_DEVICES=$i bash runs/run_cue.sh ${TAG:-oamfix}_${d}_v1_s42 & i=$((i+1)); done; wait
for d in omat24 mptrj salex; do for v in v2 v1; do N=${TAG:-oamfix}_${d}_${v}_s42
  echo "== $N"; tail -n 3 runs/${N}_cue/stdout.txt; grep "^[0-9][0-9]/[0-9][0-9]/20" runs/${N}_cue/loss_${N}_cue.out | tail -n 2
done; done
