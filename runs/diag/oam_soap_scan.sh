#!/bin/bash
# SOAP stability scan on the OAM subsets (2026-11-06): 200 real fixed-shape steps per optimizer setting, 3 GPUs,
# (float64 SOAP last on GPU 2), plus the padding regression test with full output.
# usage: sbatch -J oam_soap -p normal --comment="field=sciai;appl=jax" --gres=gpu:3 -t 01:30:00 runs/diag/oam_soap_scan.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
C=input/input_3bpa_omat24_oamfix2_salex_v2_s42_cue.json
F="^\[|pads|done|Error|error"
CUDA_VISIBLE_DEVICES=0 python runs/diag/oam_step_debug.py $C 200 soap_lr1e-2: soap_lr3e-3:learning_rate=0.003 soap_reference:soap_impl=reference 2>&1 | grep -E "$F" > runs/slurm/soapscan_g0.txt &
CUDA_VISIBLE_DEVICES=1 python runs/diag/oam_step_debug.py $C 200 soap_lr1e-3:learning_rate=0.001 soap_b2_0.99:soap_b2=0.99 2>&1 | grep -E "$F" > runs/slurm/soapscan_g1.txt &
CUDA_VISIBLE_DEVICES=2 python runs/diag/oam_step_debug.py $C 200 soap_cholqr:soap_orth=cholqr amsgrad:optimizer=amsgrad soap_x64:x64=true 2>&1 | grep -E "$F" > runs/slurm/soapscan_g2.txt &
wait
cat runs/slurm/soapscan_g*.txt
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_oam_stress.py -k many_padding 2>&1 | grep -v "Warning\|warn" | tail -n 40
