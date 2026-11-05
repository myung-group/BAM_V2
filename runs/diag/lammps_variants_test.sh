#!/bin/bash
# C++ runtime test of the 4 exported variants only (exports from runs/diag/lammps_variants.sh), unfiltered output.
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$SP/cuequivariance_ops/lib:${LD_LIBRARY_PATH}
export PATH=$SP/nvidia/cuda_nvcc/bin:$PATH
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cuda_data_dir=$SP/nvidia/cuda_nvcc"
export TF_CPP_MIN_LOG_LEVEL=1
REF=runs/lammps_export/lips_e3nn
for V in e3nn_float32 cue_float64 cue_float32; do
  echo "== $V"; ./lammps/test_pjrt $SP/jax_plugins/xla_cuda12/xla_cuda_plugin.so runs/lammps_export/lips_$V 3 $REF 1e-2 2>&1 | grep -a -v cuda_timer | tail -n 20
done
