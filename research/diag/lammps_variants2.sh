#!/bin/bash
# re-export the float32 variants (outputs cast to the model dtype) and test e3nn_float32, cue_float64, cue_float32
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
C=research/input/input_3bpa_omat24_lipsF_fu_s42_cue.json
K=runs/lipsF_fu_s42_cue/ckpt_lipsF_fu_s42_cue.gauge.pkl
for B in e3nn cue; do
  python -m race_v2.export.lammps $C --ckpt $K --out runs/lammps_export/lips_${B}_float32 --backend $B --dtype float32 \
    --buckets 128:4096 1024:32768 4096:131072 2>&1 | grep -a -E "exported|recorded|Error|error|Traceback" | head -4
done
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$SP/cuequivariance_ops/lib:${LD_LIBRARY_PATH}
export PATH=$SP/nvidia/cuda_nvcc/bin:$PATH
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cuda_data_dir=$SP/nvidia/cuda_nvcc"
export TF_CPP_MIN_LOG_LEVEL=1
REF=runs/lammps_export/lips_e3nn
for V in e3nn_float32 cue_float64 cue_float32; do
  echo "== $V"; ./lammps/test_pjrt $SP/jax_plugins/xla_cuda12/xla_cuda_plugin.so runs/lammps_export/lips_$V 3 $REF 1e-2 2>&1 | grep -a -v cuda_timer | tail -n 12
done
