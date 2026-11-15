#!/bin/bash
# LAMMPS step 4 (2026-11-06): pair_style racev2 run 0 vs Python, 1 ps NVE drift, speed (LiPS, one GPU).
# usage: sbatch -J rv2_lmp -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 00:40:00 research/diag/lammps_step4.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH}
export PATH=$SP/nvidia/cuda_nvcc/bin:$PATH
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cuda_data_dir=$SP/nvidia/cuda_nvcc"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python research/diag/lammps_validate.py /scratch/t510a13/src/lammps-racev2-build/lmp_racev2 \
  $SP/jax_plugins/xla_cuda12/xla_cuda_plugin.so runs/lammps_export/lips_e3nn \
  research/input/input_3bpa_omat24_lipsF_fu_s42_cue.json runs/lipsF_fu_s42_cue/ckpt_lipsF_fu_s42_cue.gauge.pkl \
  /scratch/t510a13/data/datasets_ext/lips/fu_test.xyz 2>&1 | grep -E "^run 0|^NVE|performance|Error|ERROR|Traceback|error"
