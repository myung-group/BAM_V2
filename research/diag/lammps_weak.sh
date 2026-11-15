#!/bin/bash
# pair_style race mode halo weak scaling (2026-11-06): 5312 LiPS atoms per GPU, 1 / 2 / 4 GPUs on one node, 8 GPUs on
# two nodes (Slingshot); energies / forces checked against the 1-GPU run (replicated frame).
# usage: sbatch -J rv2_weak -p normal --comment="field=sciai;appl=jax" -N 2 --exclusive --ntasks-per-node=4 \
#          --gres=gpu:4 -t 01:30:00 runs/diag/lammps_weak.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
module load PrgEnv-gnu cray-mpich >/dev/null 2>&1
SP=$(python -c "import site; print(site.getsitepackages()[0])")
OUT=${OUT:-runs/lammps_export/lips_cue_halo4}
export RACEV2_TIMING=1
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$SP/cuequivariance_ops/lib:${LD_LIBRARY_PATH}
export PATH=$SP/nvidia/cuda_nvcc/bin:$PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cuda_data_dir=$SP/nvidia/cuda_nvcc"
python runs/diag/lammps_weak.py --mpi /scratch/t510a13/src/lammps-racev2-mpi-build/lmp_racev2_mpi \
  --plugin $SP/jax_plugins/xla_cuda12/xla_cuda_plugin.so --export $OUT \
  --xyz /scratch/t510a13/data/datasets_ext/lips/fu_test.xyz --base 4 4 4 --steps 200 \
  --runs 1:1:1x1x1 2:1:2x1x1 4:1:2x2x1 8:2:2x2x2
