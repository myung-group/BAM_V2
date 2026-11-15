#!/bin/bash
# node_energies split + staged halo exchange check (2026-11-06), one GPU
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
C=research/input/input_3bpa_omat24_lipsF_fu_s42_cue.json
K=runs/lipsF_fu_s42_cue/ckpt_lipsF_fu_s42_cue.gauge.pkl
python research/diag/staged_check.py $C $K /scratch/t510a13/data/datasets_ext/lips/fu_test.xyz runs/lammps_export/lips_e3nn e3nn 1 2>&1 | grep -a -E "monolithic|halo|staged|Error|error|Traceback"
python research/diag/staged_check.py $C $K /scratch/t510a13/data/datasets_ext/lips/fu_test.xyz runs/lammps_export/lips_e3nn cue 1 2>&1 | grep -a -E "staged|halo|Error|error|Traceback"
[ "${RUN_TESTS:-0}" = 1 ] && python -m pytest -q tests/test_oam_stress.py tests/test_periodic.py tests/test_equivariance.py tests/test_forces_are_gradient.py tests/test_indexed_linear.py tests/test_round15.py 2>&1 | tail -n 3
