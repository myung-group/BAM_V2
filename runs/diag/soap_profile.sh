#!/bin/bash
# Parameter breakdown, then SOAP / model time breakdown (2026-11-06): float64 on GPU 0, float32 on GPU 1; RACEv2 (sAlex, MPtrj batches) and V1.
# usage: sbatch -J soap_prof -p normal --comment="field=sciai;appl=jax" --gres=gpu:2 -t 01:00:00 runs/diag/soap_profile.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
C="input/input_3bpa_omat24_oam64_salex_v2_s42_cue.json input/input_3bpa_omat24_oam64_mptrj_v2_s42_cue.json input/input_3bpa_omat24_oam64_salex_v1_s42_cue.json"
CUDA_VISIBLE_DEVICES=0 python runs/diag/param_breakdown.py input/input_3bpa_omat24_oam64_salex_v2_s42_cue.json input/input_3bpa_omat24_oam64_salex_v1_s42_cue.json input/input_3bpa_omat24_abl3_full_s42_cue.json 2>&1 | grep -E "^==|^   |Error|error" > runs/slurm/param_breakdown.txt
CUDA_VISIBLE_DEVICES=0 python runs/diag/soap_profile.py --dtype float64 $C 2>&1 | grep -E "^\[|Error|error" > runs/slurm/soapprof_f64.txt &
CUDA_VISIBLE_DEVICES=1 python runs/diag/soap_profile.py --dtype float32 $C 2>&1 | grep -E "^\[|Error|error" > runs/slurm/soapprof_f32.txt &
wait
cat runs/slurm/param_breakdown.txt runs/slurm/soapprof_f64.txt runs/slurm/soapprof_f32.txt
