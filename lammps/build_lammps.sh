#!/bin/bash
# Build LAMMPS with pair_style race on a GPU login node (aarch64 = GPU nodes), <= 16 jobs.
#   bash lammps/build_lammps.sh <LAMMPS source> <build dir> [mpi|serial]
#   mpi    : Cray MPICH (PrgEnv-gnu + cray-mpich, compiler wrappers CC / cc), binary lmp_racev2_mpi
#   serial : no MPI (STUBS), binary lmp_racev2
set -e
LMP_SRC=${1:?LAMMPS source dir}
BUILD=${2:?build dir}
KIND=${3:-serial}
HERE=$(cd "$(dirname "$0")" && pwd)
# install the pair style and the PJRT runtime into the LAMMPS source tree (top-level src/*.cpp are always built)
rm -f "$LMP_SRC/src/pair_racev2.h" "$LMP_SRC/src/pair_racev2.cpp"   # old name (2026-11-06)
cp "$HERE/pair_race.h" "$HERE/pair_race.cpp" "$LMP_SRC/src/"
cp "$HERE/racev2_pjrt.h" "$LMP_SRC/src/"
cp "$HERE/racev2_pjrt.cc" "$LMP_SRC/src/racev2_pjrt.cpp"
mkdir -p "$LMP_SRC/src/third_party"
cp "$HERE/third_party/pjrt_c_api.h" "$HERE/third_party/pjrt_c_api_ffi_extension.h" "$LMP_SRC/src/third_party/"
mkdir -p "$BUILD" && cd "$BUILD"
if [ "$KIND" = mpi ]; then
  module load PrgEnv-gnu cray-mpich >/dev/null 2>&1
  OPTS="-DBUILD_MPI=on -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc -DLAMMPS_MACHINE=racev2_mpi"
else
  module load gcc-native/14 >/dev/null 2>&1 || true
  OPTS="-DBUILD_MPI=off -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DLAMMPS_MACHINE=racev2"
fi
cmake "$LMP_SRC/cmake" -DCMAKE_BUILD_TYPE=Release -DBUILD_OMP=off -DCMAKE_EXE_LINKER_FLAGS="-ldl" $OPTS > cmake.log 2>&1
make -j 16 > make.log 2>&1
grep -i "warning" make.log | grep -i "racev2\|pjrt" | head -5 || true
ls -la lmp_racev2*
