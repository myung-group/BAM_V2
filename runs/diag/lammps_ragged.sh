#!/bin/bash
# species-indexed Linear via ragged_dot (2026-11-06): unit tests, cue float64 export with BAM_INDEXED_LINEAR=ragged,
# C++ check against the e3nn/gather reference cases, LAMMPS multi-GPU check (owner / ghost 1, 2, 4 ranks).
# usage: sbatch -J rv2_ragged -p normal --comment="field=sciai;appl=jax" -N 1 --ntasks-per-node=4 --gres=gpu:4 \
#          -t 01:30:00 runs/diag/lammps_ragged.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
module load PrgEnv-gnu cray-mpich >/dev/null 2>&1
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_indexed_linear.py tests/test_oam_stress.py 2>&1 | tail -n 4
OUT=runs/lammps_export/lips_cue_ragged
CUDA_VISIBLE_DEVICES=0 BAM_INDEXED_LINEAR=ragged python -m bam_omat24.export_lammps input/input_3bpa_omat24_lipsF_fu_s42_cue.json \
  --ckpt runs/lipsF_fu_s42_cue/ckpt_lipsF_fu_s42_cue.gauge.pkl --out $OUT --backend cue \
  --buckets 128:4096 1024:32768 4096:131072 16384:524288 32768:1048576 2>&1 | grep -a -E "exported|recorded|Error|error|Traceback"
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$SP/cuequivariance_ops/lib:${LD_LIBRARY_PATH}
export PATH=$SP/nvidia/cuda_nvcc/bin:$PATH
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cuda_data_dir=$SP/nvidia/cuda_nvcc"
export TF_CPP_MIN_LOG_LEVEL=1
CUDA_VISIBLE_DEVICES=0 ./lammps/test_pjrt $SP/jax_plugins/xla_cuda12/xla_cuda_plugin.so $OUT 3 runs/lammps_export/lips_e3nn 1e-8 2>&1 \
  | grep -a -E "case|ms per|ALL|FAIL|what"
python runs/diag/lammps_multi.py --serial /scratch/t589a02/src/lammps-racev2-build/lmp_racev2 \
  --mpi /scratch/t589a02/src/lammps-racev2-mpi-build/lmp_racev2_mpi \
  --plugin $SP/jax_plugins/xla_cuda12/xla_cuda_plugin.so --export $OUT --tag ragged \
  --xyz /scratch/t589a02/datasets_ext/lips/fu_test.xyz --rep 2 2 2 --ranks 2 4 --steps 200 2>&1 | grep -a -v cuda_timer
