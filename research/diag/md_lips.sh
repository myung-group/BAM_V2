#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH -o runs/slurm/%x_%j.out
# LiPS MD observables (Fu et al. TMLR 2023 protocol): 5 NVT runs (520 K, 0.25 fs, Nose-Hoover ttime 20, 200k steps)
# from 5 random test configurations with the final LiPS model; restartable; analysis at the end.
# usage: sbatch -J mdL --gres=gpu:4 -t 24:00:00 research/diag/md_lips.sh [run name, default lipsF_fu_s42]
N=${1:-lipsF_fu_s42}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
# 1st attempt (job 1219565) aborted in 3 min: "pthread_create failed: Resource temporarily unavailable" while XLA/LLVM
# compiled in 5 processes at once (parallel compilation spawns one thread per core). Fix: serial compilation,
# 1 BLAS/OMP thread, and at most 4 processes at a time (run 4 starts when the first run finishes).
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_force_compilation_parallelism=1"
cd /scratch/t510a13/race_v2/work
C=runs/${N}_cue/ckpt_${N}_cue.gauge.pkl; [ -f $C ] || C=runs/${N}_cue/ckpt_${N}_cue.pkl
STEPS=${MD_STEPS:-200000}; OUT=runs/md_lips_${N}${MD_TAG:-}   # MD_STEPS/MD_TAG: short test runs in a separate folder
for k in 0 1 2 3 4; do
  if [ $k -eq 4 ]; then wait -n; fi      # never more than 4 MD processes at once
  CUDA_VISIBLE_DEVICES=$((k % 4)) python -m race_v2.inference.md_run research/input/input_3bpa_omat24_${N}_cue.json --ckpt $C \
    --init-xyz /scratch/t510a13/data/datasets_ext/lips/fu_test.xyz --out $OUT --run $k --n-runs 5 --init-seed 123 \
    --steps $STEPS --dt 0.25 --temp 520 --ttime 20 --save-freq 50 --chunk 400 --max-hours 23.3 > $OUT.run$k.log 2>&1 &
done
wait
python -m race_v2.eval.md_lips_analysis $OUT 2>&1 | tee $OUT/analysis.log
