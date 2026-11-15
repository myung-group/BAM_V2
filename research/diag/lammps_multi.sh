#!/bin/bash
# pair_style race multi-GPU check on one node (2026-11-06): re-export (n_local signature, large buckets), C++ test,
# then LiPS 2x2x2 run 0 + NVE with 1 rank owner / 1 rank ghost / 2 and 4 ranks ghost (one GPU per rank).
# usage: sbatch -J rv2_multi -p normal --comment="field=sciai;appl=jax" -N 1 --ntasks-per-node=4 --gres=gpu:4 \
#          -t 01:30:00 research/diag/lammps_multi.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
module load PrgEnv-gnu cray-mpich >/dev/null 2>&1
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
OUT=${OUT:-runs/lammps_export/lips_cue_big}
BACKEND=${BACKEND:-cue}
CUDA_VISIBLE_DEVICES=0 python -m race_v2.export.lammps research/input/input_3bpa_omat24_lipsF_fu_s42_cue.json \
  --ckpt runs/lipsF_fu_s42_cue/ckpt_lipsF_fu_s42_cue.gauge.pkl --out $OUT --backend $BACKEND \
  --buckets 128:4096 1024:32768 4096:131072 16384:524288 32768:1048576 \
  --cases /scratch/t510a13/data/datasets_ext/lips/fu_test.xyz 2>&1 | grep -a -E "exported|recorded|written|Error|error|Traceback"
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$SP/cuequivariance_ops/lib:${LD_LIBRARY_PATH}
export PATH=$SP/nvidia/cuda_nvcc/bin:$PATH
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cuda_data_dir=$SP/nvidia/cuda_nvcc"
CUDA_VISIBLE_DEVICES=0 ./lammps/test_pjrt $SP/jax_plugins/xla_cuda12/xla_cuda_plugin.so $OUT 3 2>&1 | grep -a -E "case|ms per|ALL|FAIL|what"
python research/diag/lammps_multi.py --serial /scratch/t510a13/src/lammps-racev2-build/lmp_racev2 \
  --mpi /scratch/t510a13/src/lammps-racev2-mpi-build/lmp_racev2_mpi \
  --plugin $SP/jax_plugins/xla_cuda12/xla_cuda_plugin.so --export $OUT \
  --xyz /scratch/t510a13/data/datasets_ext/lips/fu_test.xyz --rep 2 2 2 --ranks 2 4 --steps 200
