#!/bin/bash
# Build LAMMPS with pair_style racev2 (serial, no MPI) on a GPU login node (aarch64 = GPU nodes), <= 16 jobs.
#   bash lammps/build_lammps.sh /scratch/t589a02/src/lammps-stable_30Sep2026_update1 /scratch/t589a02/src/lammps-racev2-build
set -e
LMP_SRC=${1:?LAMMPS source dir}
BUILD=${2:?build dir}
HERE=$(cd "$(dirname "$0")" && pwd)
module load gcc-native/14 >/dev/null 2>&1 || true
# install the pair style and the PJRT runtime into the LAMMPS source tree (top-level src/*.cpp are always built)
cp "$HERE/pair_racev2.h" "$HERE/pair_racev2.cpp" "$LMP_SRC/src/"
cp "$HERE/racev2_pjrt.h" "$LMP_SRC/src/"
cp "$HERE/racev2_pjrt.cc" "$LMP_SRC/src/racev2_pjrt.cpp"
mkdir -p "$LMP_SRC/src/third_party"
cp "$HERE/third_party/pjrt_c_api.h" "$LMP_SRC/src/third_party/"
mkdir -p "$BUILD" && cd "$BUILD"
cmake "$LMP_SRC/cmake" -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=off -DBUILD_OMP=off \
  -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_EXE_LINKER_FLAGS="-ldl" -DLAMMPS_MACHINE=racev2 > cmake.log 2>&1
make -j 16 > make.log 2>&1
ls -la lmp_racev2
