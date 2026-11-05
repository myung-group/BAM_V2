#!/bin/bash
# Mixed-precision SOAP check (2026-11-06): (1) float64-promotion audit of the float32 model under jax_enable_x64,
# (2) 300 real fixed-shape steps on sAlex and OMat24 for: SOAP float32, SOAP float32 + highest matmul precision,
# SOAP with a float64 state, AMSGrad, all-float64 (model + SOAP, the benchmark setting; last); each reports divergence, the drift of parameters whose gradient stayed exactly
# 0, and ms/step.
# usage: sbatch -J oam_mixed -p normal --comment="field=sciai;appl=jax" --gres=gpu:3 -t 01:30:00 runs/diag/oam_mixed.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
F="^\[|pads|done|Error|error"
V="${V:-soap_fp32: soap_fp32_highest:matmul_precision=highest amsgrad:optimizer=amsgrad soap_state64:soap_state_dtype=float64 soap_fp64_all:x64=true}"
TAG=${TAG:-mixed}
[ "${AUDIT:-1}" = 1 ] && CUDA_VISIBLE_DEVICES=0 python runs/diag/x64_audit.py input/input_3bpa_omat24_oamfix2_salex_v2_s42_cue.json 2>&1 | grep -v "^graphs\|it/s\]\|cuda_timer" | tail -n 50 > runs/slurm/${TAG}_audit.txt &
CUDA_VISIBLE_DEVICES=1 python runs/diag/oam_step_debug.py input/input_3bpa_omat24_oamfix2_salex_v2_s42_cue.json 300 $V 2>&1 | grep -E "$F" > runs/slurm/${TAG}_salex.txt &
CUDA_VISIBLE_DEVICES=2 python runs/diag/oam_step_debug.py input/input_3bpa_omat24_oamfix2_omat24_v2_s42_cue.json 300 $V 2>&1 | grep -E "$F" > runs/slurm/${TAG}_omat24.txt &
wait
for f in runs/slurm/${TAG}_*.txt; do echo "##### $f"; cat $f; done
