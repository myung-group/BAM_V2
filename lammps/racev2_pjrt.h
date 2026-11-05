// RACEv2 C++ runtime: runs a model exported by bam_omat24/export_lammps.py (StableHLO, float64) on one GPU through
// the PJRT C API of the CUDA plugin that ships with jaxlib (jax_plugins/xla_cuda12/xla_cuda_plugin.so).
// No Python, no JAX at run time.  Step 2 of docs/RACEv2_lammps_export_plan_2026-11-06.md.
//
// Model function (per padded bucket n_pad atoms / e_pad edges):
//   (species[n_pad] i32, receivers[e_pad] i32, senders[e_pad] i32, rvec[e_pad*3] f64, n_real i32, e_real i32,
//    n_local i32) -> (e_local f64, e_atom[n_pad] f64, dE_local/drvec[e_pad*3] f64)
// e_local sums the atomic energies of nodes 0..n_local-1 (atoms owned by this MPI rank; ghosts follow).
// receivers = centre atom i, senders = neighbour j (owned index), rvec = x_j(image) - x_i.  Energies exclude the
// per-species E0 (see atom_energy0()).  Forces: F_i += g, F_j -= g; virial W = -sum r (x) g.
#pragma once

#include <string>
#include <vector>

struct PJRT_Api;
struct PJRT_Client;
struct PJRT_Device;
struct PJRT_LoadedExecutable;

namespace racev2 {

struct Bucket {
  int n_pad = 0, e_pad = 0;
  std::string file;
  PJRT_LoadedExecutable* exe = nullptr;  // compiled on first use
};

class Model {
 public:
  // plugin_so: path of xla_cuda_plugin.so; export_dir: directory with racev2_lammps_meta.txt, compile_options.pb
  // and the .mlirbc files; device: index among the addressable GPUs.
  Model(const std::string& plugin_so, const std::string& export_dir, int device = 0, bool verbose = true);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // Species are indices into atomic_numbers() (sorted Z).  Returns e_local (without E0) of nodes < n_local;
  // e_atom has n entries, dEdr has 3*e entries.  Picks the smallest bucket with n < n_pad, e <= e_pad and pads.
  double compute(int n, int e, int n_local, const int* species, const int* receivers, const int* senders,
                 const double* rvec, std::vector<double>& e_atom, std::vector<double>& dEdr);

  // Same, with caller-supplied padded arrays of exactly one bucket size (used by the tests).
  double compute_padded(int n_pad, int e_pad, int n, int e, int n_local, const int* species, const int* receivers,
                        const int* senders, const double* rvec, std::vector<double>& e_atom_pad,
                        std::vector<double>& dEdr_pad);

  double cutoff() const { return cutoff_; }
  int n_layers() const { return n_layers_; }
  const std::vector<int>& atomic_numbers() const { return z_; }
  const std::vector<double>& atom_energy0() const { return e0_; }
  int api_major() const { return api_major_; }
  int api_minor() const { return api_minor_; }

 private:
  Bucket& bucket_for(int n, int e);
  void compile(Bucket& b);

  const PJRT_Api* api_ = nullptr;
  void* dl_ = nullptr;
  PJRT_Client* client_ = nullptr;
  PJRT_Device* device_ = nullptr;
  std::string dir_, compile_options_;
  std::vector<Bucket> buckets_;
  std::vector<int> z_;
  std::vector<double> e0_;
  double cutoff_ = 0.0;
  int n_layers_ = 0, api_major_ = -1, api_minor_ = -1;
  bool verbose_ = true;
};

}  // namespace racev2
