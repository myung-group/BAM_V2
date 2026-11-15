#!/bin/bash
# NVE MD of a checkpoint from the Q = +1 test frame of a system, several velocity seeds (md_nve.py, runaway guard).
# usage: research/diag/local_4g/md_series.sh <system> <run name> [seeds...]
cd "$(dirname "$0")/../../.."
s=$1; n=$2; shift 2; seeds=${@:-0 1 2}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
f=$(.venv/bin/python -I -c "import ase.io; fr=ase.io.read('data_local/4ghdnnp/$s/test.xyz',':'); print(next(i for i,a in enumerate(fr) if a.info['charge']==1))")
mkdir -p runs/local_4g/md
for sd in $seeds; do
  timeout 1800 nice -n 10 ~/anaconda3/envs/bam_cue/bin/python -W ignore research/diag/local_4g/md_nve.py runs/local_4g/$n/ckpt.pkl \
      data_local/4ghdnnp/$s/test.xyz --frame $f --seed $sd --out runs/local_4g/md/${n}_v$sd.npz 2>&1 | grep -vE "CUDA|cuda_timer|W11" | tail -1
done
