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

#include <cstdint>
#include <utility>
#include <string>
#include <vector>

struct PJRT_Api;
struct PJRT_Client;
struct PJRT_Device;
struct PJRT_LoadedExecutable;
struct PJRT_Buffer;

namespace racev2 {

struct Bucket {
  int n_pad = 0, e_pad = 0;
  std::string file;
  PJRT_LoadedExecutable* exe = nullptr;  // compiled on first use
};

// Per-layer functions of one bucket for the halo-exchange mode (export_lammps.py --staged-buckets).
struct StagedBucket {
  int n_pad = 0, e_pad = 0, k_pad = 0;   // k_pad: owned-row bucket (node-wise work on k_pad rows), = n_pad if absent
  std::string stem;
  PJRT_LoadedExecutable *prelude = nullptr, *prelude_vjp = nullptr;
  std::vector<PJRT_LoadedExecutable*> layer, layer_vjp;   // compiled on first use
  std::vector<PJRT_LoadedExecutable*> layer_fwdres, layer_bwdres;   // saved-residual variant
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

  // ---- staged (per-layer halo exchange) evaluation, one call sequence per force evaluation:
  //   staged_begin -> staged_prelude -> [forward comm h_0] -> for i: staged_layer(i) -> [forward comm h_{i+1}]
  //   -> for i = L-1..0: staged_layer_vjp(i) -> [reverse comm of ct_h_i, ghost rows zeroed] -> staged_finish
  // Node rows: 0..n_local-1 owned, n_local..n-1 ghosts (copies of atoms owned elsewhere or periodic images),
  // edges only for owned receivers.  Host feature arrays have staged_n_pad() rows of staged_dims()[i] values.
  bool has_staged() const { return !staged_.empty(); }
  const std::vector<int>& staged_dims() const { return st_dims_; }
  int staged_dim_a() const { return st_dim_a_; }
  bool staged_chebyshev() const { return st_cheb_; }
  // saved-residual mode: the forward keeps the VJP residuals of every layer on the device and the backward uses them
  // (no recomputation; more memory).  On when the export has them (--staged-residuals) unless
  // RACEV2_STAGED_RESIDUALS=0.
  bool staged_residuals() const { return st_res_; }
  void staged_begin(int n, int e, int n_local, const int* species, const int* receivers, const int* senders,
                    const double* rvec);
  int staged_n_pad() const;
  double staged_prelude(std::vector<double>& h0);
  // h_prev: the layer input after forward communication (kept on the device for the backward pass)
  double staged_layer(int i, const double* h_prev, std::vector<double>& h_new);
  // ct_h_new: cotangent of the layer output (after reverse communication); returns the cotangent of the layer
  // input (including the Chebyshev contribution of layer i+1, accumulated on the device), to be reverse
  // communicated.  The cotangents of the atomic basis and of rvec are accumulated on the device.
  void staged_layer_vjp(int i, const double* ct_h_new, std::vector<double>& ct_h_prev);
  // seconds spent in executions, uploads and downloads since construction (RACEV2_TIMING=1 prints them at exit)
  double t_exec = 0.0, t_up = 0.0, t_down = 0.0;
  long n_exec = 0;
  // dE_local/drvec of the e real edges (3*e values); releases the device buffers of this evaluation
  void staged_finish(std::vector<double>& dEdr);

  double cutoff() const { return cutoff_; }
  int n_layers() const { return n_layers_; }
  const std::vector<int>& atomic_numbers() const { return z_; }
  const std::vector<double>& atom_energy0() const { return e0_; }
  bool float32() const { return f32_; }
  int api_major() const { return api_major_; }
  int api_minor() const { return api_minor_; }

 private:
  Bucket& bucket_for(int n, int e);
  void compile(Bucket& b);
  void setup_cue(const std::string& kernels, const std::string& lib, const std::string& dep);
  PJRT_LoadedExecutable* compile_file(const std::string& file);
  PJRT_Buffer* upload(const void* data, int type, const std::vector<int64_t>& dims);
  PJRT_Buffer* upload_f(const double* data, const std::vector<int64_t>& dims);   // f64 or converted to f32
  void download_f(PJRT_Buffer* b, double* dst, size_t count);
  std::vector<PJRT_Buffer*> run(PJRT_LoadedExecutable* exe, const std::vector<PJRT_Buffer*>& args, int n_out);
  void staged_release();

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
  bool f32_ = false;   // model exported in float32: rvec converted on input, outputs converted to double
  // staged
  std::vector<StagedBucket> staged_;
  std::vector<int> st_dims_;
  int st_dim_a_ = 1;
  bool st_cheb_ = false;
  StagedBucket* sb_ = nullptr;
  int st_e_ = 0;
  std::vector<PJRT_Buffer*> st_graph_, st_hs_;
  PJRT_Buffer *st_a_ = nullptr, *st_dummy_ = nullptr;
  PJRT_Buffer *st_cta_ = nullptr, *st_ctrv_ = nullptr, *st_ctadd_ = nullptr;   // device accumulators
  std::vector<std::pair<std::pair<int64_t, int64_t>, PJRT_Buffer*>> zeros_;   // cached zero arrays (immutable)
  bool st_res_ = false;
  std::vector<int> st_res_counts_;
  std::vector<std::vector<PJRT_Buffer*>> st_resbuf_;
  PJRT_Buffer* zeros(int64_t rows, int64_t cols);
  bool timing_ = false;
};

}  // namespace racev2
