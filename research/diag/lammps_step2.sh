#!/bin/bash
# LAMMPS plan step 2 (2026-11-06): export the LiPS model (e3nn backend, float64) with C++ test cases, then run the
# PJRT C++ runtime (lammps/test_pjrt, built on glogin) on them and compare with the JAX outputs.
# usage: sbatch -J rv2_cpp -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 00:40:00 runs/diag/lammps_step2.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
OUT=runs/lammps_export/lips_e3nn
python -m bam_omat24.export_lammps input/input_3bpa_omat24_lipsF_fu_s42_cue.json \
  --ckpt runs/lipsF_fu_s42_cue/ckpt_lipsF_fu_s42_cue.gauge.pkl --out $OUT --backend e3nn \
  --buckets 128:4096 256:8192 --cases /scratch/t510a13/data/datasets_ext/lips/fu_test.xyz 2>&1 \
  | grep -E "exported|written|Error|error|Traceback"
# C++ side: CUDA libraries from the pip nvidia packages (what jax_plugins/xla_cuda12 preloads in Python), ptxas
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH}
export PATH=$SP/nvidia/cuda_nvcc/bin:$PATH
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cuda_data_dir=$SP/nvidia/cuda_nvcc"
./lammps/test_pjrt $SP/jax_plugins/xla_cuda12/xla_cuda_plugin.so $OUT 3
