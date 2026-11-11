#!/bin/bash
# OAM model-size screening on 100k OMat24 structures (2026-11-06), RACEv2 adopted recipe: float64, species tables by
# Adam (soap_species_adam), cutoff 5 A, stress weight 10, batch 16, lr warmup 2 epochs.
# Selection rule (docs/size_screening_protocol_2026-11-03.md, fixed before results, validation only):
#   candidates = final-epoch valid E RMSE within +10 % of the best; pick the lowest valid F RMSE; ties (< 2 %) -> smaller.
# usage: sbatch -J <job> -p normal --comment="field=sciai;appl=jax" --gres=gpu:<n> -t <time> runs/diag/size_screen.sh <EPOCHS> <TAG_PREFIX> <L>x<H> [...]
#   e.g. ... size_screen.sh 1 szsmk L5x192 L6x128 L5x128 L4x128     (smoke: 1 epoch, time per epoch + memory)
#        ... size_screen.sh 30 sz100k L5x64 L5x128 L5x192 L4x128    (screening)
#        ... size_screen.sh 30 sz100k L6x128 L5x128:amsgrad           (deeper model; optimizer control, not a size candidate)
#        ... size_screen.sh 30 opt100k L5x128:adam L5x128:amsgrad:f32    (optimizer / precision screening; SEED=43 for repeats)
#        ... size_screen.sh 30 v1100k L5x128:v1                         (RACE V1 recipe on the same 100k data, float64)
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
EPOCHS=$1; PREFIX=$2; shift 2; SIZES=("$@")
OUT=/scratch/t510a13/data/datasets_ext/oam_subsets_100k
if [ ! -f $OUT/omat24_meta.json ]; then
  python -I bam_omat24/oam_subset.py --name omat24 --train-root /scratch/t510a13/data/oam_data/omat24_train \
    --eval-root /scratch/t510a13/data/oam_data/omat24_val --atom-ref /scratch/t510a13/data/oam_data/refs/omat24_train_ls.npz \
    --n-train 100000 --n-valid 2000 --n-test 1000 --per-shard 50 --out $OUT 2>&1 | tail -n 5
fi
[ -f $OUT/omat24_meta.json ] || { echo "ABORT: subset missing"; exit 1; }
RUNS=()
SEED=${SEED:-42}
for tok in "${SIZES[@]}"; do
  # L<layers>x<hidden>[:<optimizer>][:f32]   optimizer: soap (default, species tables by Adam) | amsgrad | adam | v1
  IFS=: read -r s opt prec <<< "$tok"
  [ "$opt" = f32 ] && { prec=f32; opt=; }
  NAME=$s; OPT=(--species-adam); X64=(--x64); VAR=v2
  if [ "$opt" = v1 ]; then OPT=(); VAR=v1                 # RACE V1 recipe (all RACEv2 additions removed, AMSGrad)
  elif [ -n "$opt" ] && [ "$opt" != soap ]; then OPT=(--optimizer $opt); NAME=${NAME}${opt}; fi
  if [ "$prec" = f32 ]; then X64=(); NAME=${NAME}f32; fi
  L=${s%x*}; L=${L#L}; H=${s#*x}
  python -I runs/diag/make_oam_configs.py $OUT --tag ${PREFIX}_${NAME} --datasets omat24 --variants $VAR "${X64[@]}" "${OPT[@]}" \
    --layers $L --hidden $H --warmup 2 --epochs $EPOCHS --seeds $SEED || exit 1
  RUNS+=(${PREFIX}_${NAME}_omat24_${VAR}_s${SEED})
done
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/scratch/t510a13/jax_cache_racev2
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
for i in "${!RUNS[@]}"; do CUDA_VISIBLE_DEVICES=$i bash runs/run_cue.sh ${RUNS[$i]} & done
wait
for N in "${RUNS[@]}"; do echo "== $N"; grep -a -E "Model parameters|s/epoch|sec" runs/${N}_cue/stdout.txt | head -n 3; tail -n 2 runs/${N}_cue/stdout.txt; done
