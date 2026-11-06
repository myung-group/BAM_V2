// RACEv2 C++ runtime over the PJRT C API (see racev2_pjrt.h).
#include "racev2_pjrt.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "third_party/pjrt_c_api.h"
#include "third_party/pjrt_c_api_ffi_extension.h"

namespace racev2 {
namespace {

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("racev2: cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void check(const PJRT_Api* api, PJRT_Error* err, const char* what) {
  if (!err) return;
  PJRT_Error_Message_Args m;
  std::memset(&m, 0, sizeof(m));
  m.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  m.error = err;
  api->PJRT_Error_Message(&m);
  std::string msg(m.message, m.message_size);
  PJRT_Error_Destroy_Args d;
  std::memset(&d, 0, sizeof(d));
  d.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  d.error = err;
  api->PJRT_Error_Destroy(&d);
  throw std::runtime_error(std::string("racev2: ") + what + ": " + msg);
}

void await_and_destroy(const PJRT_Api* api, PJRT_Event* ev, const char* what) {
  if (!ev) return;
  PJRT_Event_Await_Args a;
  std::memset(&a, 0, sizeof(a));
  a.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  a.event = ev;
  PJRT_Error* err = api->PJRT_Event_Await(&a);
  PJRT_Event_Destroy_Args d;
  std::memset(&d, 0, sizeof(d));
  d.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  d.event = ev;
  api->PJRT_Event_Destroy(&d);
  check(api, err, what);
}

void destroy_buffer(const PJRT_Api* api, PJRT_Buffer* b) {
  if (!b) return;
  PJRT_Buffer_Destroy_Args d;
  std::memset(&d, 0, sizeof(d));
  d.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  d.buffer = b;
  api->PJRT_Buffer_Destroy(&d);
}

}  // namespace

Model::Model(const std::string& plugin_so, const std::string& export_dir, int device, bool verbose)
    : dir_(export_dir), verbose_(verbose) {
  // ---- metadata
  std::istringstream meta(read_file(dir_ + "/racev2_lammps_meta.txt"));
  std::string line;
  int signature = 6, st_sig = 1;
  const char* tm = std::getenv("RACEV2_TIMING");
  timing_ = tm && tm[0] == '1';
  std::string cue_kernels, cue_lib, cue_dep;
  while (std::getline(meta, line)) {
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key == "cutoff") ls >> cutoff_;
    else if (key == "n_layers") ls >> n_layers_;
    else if (key == "atomic_numbers") { int z; while (ls >> z) z_.push_back(z); }
    else if (key == "e0") { double v; while (ls >> v) e0_.push_back(v); }
    else if (key == "bucket") { Bucket b; ls >> b.n_pad >> b.e_pad >> b.file; buckets_.push_back(b); }
    else if (key == "signature") ls >> signature;
    else if (key == "dtype") { std::string d; ls >> d; f32_ = (d == "float32"); }
    else if (key == "cue_kernels") ls >> cue_kernels;
    else if (key == "cue_lib") ls >> cue_lib;
    else if (key == "cue_dep") ls >> cue_dep;
    else if (key == "staged_dims") { int d; while (ls >> d) st_dims_.push_back(d); }
    else if (key == "staged_dim_a") ls >> st_dim_a_;
    else if (key == "staged_cheb") { int c; ls >> c; st_cheb_ = c != 0; }
    else if (key == "staged_signature") ls >> st_sig;
    else if (key == "staged_res_counts") { int c; while (ls >> c) st_res_counts_.push_back(c); }
    else if (key == "staged_bucket") {
      StagedBucket b;
      ls >> b.n_pad >> b.e_pad >> b.stem;
      if (!(ls >> b.k_pad)) b.k_pad = b.n_pad;
      staged_.push_back(b);
    }
  }
  if (!staged_.empty() && st_sig != 2)
    throw std::runtime_error("racev2: " + dir_ + " has staged functions of an older signature; re-export it");
  if (!staged_.empty() && (int) st_dims_.size() != n_layers_ + 1)
    throw std::runtime_error("racev2: staged_dims does not have n_layers + 1 entries in " + dir_);
  for (auto& b : staged_) {
    b.layer.assign(n_layers_, nullptr);
    b.layer_vjp.assign(n_layers_, nullptr);
    b.layer_fwdres.assign(n_layers_, nullptr);
    b.layer_bwdres.assign(n_layers_, nullptr);
  }
  {
    const char* rs = std::getenv("RACEV2_STAGED_RESIDUALS");
    st_res_ = (int) st_res_counts_.size() == n_layers_ && !(rs && rs[0] == '0');
  }
  std::sort(staged_.begin(), staged_.end(), [](const StagedBucket& a, const StagedBucket& b) {
    if (a.n_pad != b.n_pad) return a.n_pad < b.n_pad;
    if (a.k_pad != b.k_pad) return a.k_pad < b.k_pad;
    return a.e_pad < b.e_pad;
  });
  if (signature != 7)
    throw std::runtime_error("racev2: " + dir_ + " was exported with an older function signature (no n_local); "
                             "re-export it with bam_omat24/export_lammps.py");
  if (buckets_.empty() || z_.empty() || cutoff_ <= 0.0) throw std::runtime_error("racev2: bad metadata in " + dir_);
  std::sort(buckets_.begin(), buckets_.end(),
            [](const Bucket& a, const Bucket& b) { return a.n_pad < b.n_pad || (a.n_pad == b.n_pad && a.e_pad < b.e_pad); });
  compile_options_ = read_file(dir_ + "/compile_options.pb");

  // ---- plugin
  dl_ = dlopen(plugin_so.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!dl_) throw std::runtime_error(std::string("racev2: dlopen failed: ") + dlerror());
  using GetApi = const PJRT_Api* (*)();
  auto get_api = reinterpret_cast<GetApi>(dlsym(dl_, "GetPjrtApi"));
  if (!get_api) throw std::runtime_error("racev2: GetPjrtApi not found in " + plugin_so);
  api_ = get_api();
  api_major_ = api_->pjrt_api_version.major_version;
  api_minor_ = api_->pjrt_api_version.minor_version;
  if (verbose_)
    std::fprintf(stderr, "racev2: PJRT plugin API %d.%d, header %d.%d\n", api_major_, api_minor_, PJRT_API_MAJOR,
                 PJRT_API_MINOR);
  if (api_major_ != PJRT_API_MAJOR) throw std::runtime_error("racev2: PJRT major version mismatch");

  PJRT_Plugin_Initialize_Args init;
  std::memset(&init, 0, sizeof(init));
  init.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  check(api_, api_->PJRT_Plugin_Initialize(&init), "PJRT_Plugin_Initialize");

  // no up-front grab of 75 % of the GPU (several ranks may share a node; LAMMPS needs little device memory);
  // RACEV2_PREALLOCATE=1 restores the XLA default
  const char* pre = std::getenv("RACEV2_PREALLOCATE");
  PJRT_NamedValue opt;
  std::memset(&opt, 0, sizeof(opt));
  opt.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  static const char kPre[] = "preallocate";
  opt.name = kPre;
  opt.name_size = sizeof(kPre) - 1;
  opt.type = PJRT_NamedValue_kBool;
  opt.bool_value = pre && pre[0] == '1';
  opt.value_size = 1;
  PJRT_Client_Create_Args cc;
  std::memset(&cc, 0, sizeof(cc));
  cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  cc.create_options = &opt;
  cc.num_options = 1;
  check(api_, api_->PJRT_Client_Create(&cc), "PJRT_Client_Create");
  client_ = cc.client;

  PJRT_Client_AddressableDevices_Args ad;
  std::memset(&ad, 0, sizeof(ad));
  ad.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  ad.client = client_;
  check(api_, api_->PJRT_Client_AddressableDevices(&ad), "PJRT_Client_AddressableDevices");
  if (device < 0 || static_cast<size_t>(device) >= ad.num_addressable_devices)
    throw std::runtime_error("racev2: device index out of range");
  device_ = ad.addressable_devices[device];
  if (!cue_kernels.empty()) setup_cue(cue_kernels, cue_lib, cue_dep);
  if (verbose_)
    std::fprintf(stderr, "racev2: %zu GPU(s), using %d; %zu buckets, cutoff %.3f A, %zu species\n",
                 ad.num_addressable_devices, device, buckets_.size(), cutoff_, z_.size());
}

Model::~Model() {
  if (!api_) return;
  if (timing_)
    std::fprintf(stderr, "racev2 timing: %ld executions %.3f s, uploads %.3f s, downloads %.3f s\n", n_exec, t_exec,
                 t_up, t_down);
  staged_release();
  for (auto& z : zeros_) destroy_buffer(api_, z.second);
  auto destroy_exe = [this](PJRT_LoadedExecutable* e) {
    if (!e) return;
    PJRT_LoadedExecutable_Destroy_Args d;
    std::memset(&d, 0, sizeof(d));
    d.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    d.executable = e;
    api_->PJRT_LoadedExecutable_Destroy(&d);
  };
  for (auto& b : staged_) {
    destroy_exe(b.prelude);
    destroy_exe(b.prelude_vjp);
    for (auto* e : b.layer) destroy_exe(e);
    for (auto* e : b.layer_vjp) destroy_exe(e);
    for (auto* e : b.layer_fwdres) destroy_exe(e);
    for (auto* e : b.layer_bwdres) destroy_exe(e);
  }
  for (auto& b : buckets_) {
    if (b.exe) {
      PJRT_LoadedExecutable_Destroy_Args d;
      std::memset(&d, 0, sizeof(d));
      d.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
      d.executable = b.exe;
      api_->PJRT_LoadedExecutable_Destroy(&d);
    }
  }
  if (client_) {
    PJRT_Client_Destroy_Args d;
    std::memset(&d, 0, sizeof(d));
    d.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    d.client = client_;
    api_->PJRT_Client_Destroy(&d);
  }
  // the plugin is not dlclose()d: CUDA / XLA keep global state alive until process exit
}

// cuequivariance kernels: the exported HLO calls the FFI target "uniform_1d_cuda" with kernel handles that the
// Python process created with compile_uniform_1d_ctypes at export time.  Recompile the recorded kernels in the same
// order (handles are assigned sequentially per process) and register the CUDA handler with the plugin.
void Model::setup_cue(const std::string& kernels, const std::string& lib, const std::string& dep) {
  if (!dlopen(dep.c_str(), RTLD_NOW | RTLD_GLOBAL))
    throw std::runtime_error(std::string("racev2: cannot load ") + dep + ": " + dlerror());
  void* h = dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!h) throw std::runtime_error(std::string("racev2: cannot load ") + lib + ": " + dlerror());
  // the exact arity depends on the cuequivariance-ops version (34 integer-class arguments after the name in 0.13):
  // declare 40 and pass zeros for the unused ones (extra integer arguments are harmless in the AAPCS64 / SysV ABIs)
  using Compile = int64_t (*)(const char*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t);
  auto compile_fn = reinterpret_cast<Compile>(dlsym(h, "compile_uniform_1d_ctypes"));
  void* handler = dlsym(h, "execute_uniform_1d_cuda_handler");
  if (!compile_fn || !handler) throw std::runtime_error("racev2: cue symbols not found in " + lib);

  std::istringstream in(read_file(dir_ + "/" + kernels));
  std::string line;
  int replayed = 0;
  bool have = static_cast<bool>(std::getline(in, line));
  while (have) {
    std::istringstream hl(line);
    std::string tok; int64_t handle_ref; int nargs; std::string name;
    hl >> tok >> handle_ref >> nargs >> name;
    if (tok != "call") throw std::runtime_error("racev2: bad cue_kernels file");
    std::vector<std::vector<int64_t>> arrays;
    arrays.reserve(64);
    std::vector<int64_t> args;
    while ((have = static_cast<bool>(std::getline(in, line))) && line.rfind("call ", 0) != 0) {
      std::istringstream ls(line);
      std::string kind; ls >> kind;
      if (kind == "s") { int64_t v; ls >> v; args.push_back(v); }
      else if (kind == "a") {
        int64_t n; ls >> n; arrays.emplace_back(n);
        for (int64_t q = 0; q < n; ++q) ls >> arrays.back()[q];
        args.push_back(reinterpret_cast<int64_t>(arrays.back().data()));
        args.push_back(n);
      }
    }
    if ((int) args.size() != nargs - 1 || args.size() > 40)
      throw std::runtime_error("racev2: cue_kernels entry does not match its argument count");
    args.resize(40, 0);
    const int64_t* a = args.data();
    int64_t got = compile_fn(name.c_str(), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11],
                             a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23],
                             a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35],
                             a[36], a[37], a[38], a[39]);
    if (got != handle_ref) {
      std::ostringstream ss;
      ss << "racev2: cue kernel handle mismatch (" << got << " vs recorded " << handle_ref << ")";
      throw std::runtime_error(ss.str());
    }
    ++replayed;
  }

  const PJRT_Extension_Base* ext = api_->extension_start;
  while (ext && ext->type != PJRT_Extension_Type_FFI) ext = ext->next;
  if (!ext) throw std::runtime_error("racev2: the PJRT plugin has no FFI extension");
  const auto* ffi = reinterpret_cast<const PJRT_FFI_Extension*>(ext);
  static const char kTarget[] = "uniform_1d_cuda";
  static const char kPlatform[] = "CUDA";
  PJRT_FFI_Register_Handler_Args r;
  std::memset(&r, 0, sizeof(r));
  r.struct_size = PJRT_FFI_Register_Handler_Args_STRUCT_SIZE;
  r.target_name = kTarget;
  r.target_name_size = sizeof(kTarget) - 1;
  r.handler = handler;
  r.platform_name = kPlatform;
  r.platform_name_size = sizeof(kPlatform) - 1;
  check(api_, ffi->register_handler(&r), "PJRT FFI register uniform_1d_cuda");
  if (verbose_) std::fprintf(stderr, "racev2: cue kernels replayed (%d), FFI handler registered\n", replayed);
}

void Model::compile(Bucket& b) {
  b.exe = compile_file(b.file);
  if (verbose_) std::fprintf(stderr, "racev2: compiled bucket n_pad %d e_pad %d\n", b.n_pad, b.e_pad);
}

PJRT_LoadedExecutable* Model::compile_file(const std::string& file) {
  std::string code = read_file(dir_ + "/" + file);
  static const char kFormat[] = "mlir";
  PJRT_Program prog;
  std::memset(&prog, 0, sizeof(prog));
  prog.struct_size = PJRT_Program_STRUCT_SIZE;
  prog.code = code.data();
  prog.code_size = code.size();
  prog.format = kFormat;
  prog.format_size = sizeof(kFormat) - 1;
  PJRT_Client_Compile_Args ca;
  std::memset(&ca, 0, sizeof(ca));
  ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  ca.client = client_;
  ca.program = &prog;
  ca.compile_options = compile_options_.data();
  ca.compile_options_size = compile_options_.size();
  check(api_, api_->PJRT_Client_Compile(&ca), ("PJRT_Client_Compile " + file).c_str());
  return ca.executable;
}

Bucket& Model::bucket_for(int n, int e) {
  for (auto& b : buckets_)
    if (n < b.n_pad && e <= b.e_pad) return b;
  std::ostringstream ss;
  ss << "racev2: no exported bucket fits " << n << " atoms / " << e << " edges (largest " << buckets_.back().n_pad
     << " / " << buckets_.back().e_pad << "); export a larger bucket";
  throw std::runtime_error(ss.str());
}

double Model::compute(int n, int e, int n_local, const int* species, const int* receivers, const int* senders,
                      const double* rvec, std::vector<double>& e_atom, std::vector<double>& dEdr) {
  Bucket& b = bucket_for(n, e);
  std::vector<int> sp(b.n_pad, 0), rc(b.e_pad, n), sd(b.e_pad, n);
  std::vector<double> rv(3 * static_cast<size_t>(b.e_pad), 0.0);
  std::copy(species, species + n, sp.begin());
  std::copy(receivers, receivers + e, rc.begin());
  std::copy(senders, senders + e, sd.begin());
  std::copy(rvec, rvec + 3 * static_cast<size_t>(e), rv.begin());
  std::vector<double> ea, g;
  double et = compute_padded(b.n_pad, b.e_pad, n, e, n_local, sp.data(), rc.data(), sd.data(), rv.data(), ea, g);
  e_atom.assign(ea.begin(), ea.begin() + n);
  dEdr.assign(g.begin(), g.begin() + 3 * static_cast<size_t>(e));
  return et;
}

double Model::compute_padded(int n_pad, int e_pad, int n, int e, int n_local, const int* species, const int* receivers,
                             const int* senders, const double* rvec, std::vector<double>& e_atom_pad,
                             std::vector<double>& dEdr_pad) {
  Bucket* bp = nullptr;
  for (auto& b : buckets_)
    if (b.n_pad == n_pad && b.e_pad == e_pad) bp = &b;
  if (!bp) throw std::runtime_error("racev2: no bucket of this exact size");
  if (!bp->exe) compile(*bp);

  // ---- inputs -> device
  struct In {
    const void* data;
    PJRT_Buffer_Type type;
    std::vector<int64_t> dims;
  };
  const int32_t n32 = n, e32 = e, l32 = n_local;
  std::vector<float> rv32;
  if (f32_) rv32.assign(rvec, rvec + 3 * static_cast<size_t>(e_pad));
  const void* rv_data = f32_ ? static_cast<const void*>(rv32.data()) : static_cast<const void*>(rvec);
  const PJRT_Buffer_Type ftype = f32_ ? PJRT_Buffer_Type_F32 : PJRT_Buffer_Type_F64;
  std::vector<In> ins = {
      {species, PJRT_Buffer_Type_S32, {n_pad}},       {receivers, PJRT_Buffer_Type_S32, {e_pad}},
      {senders, PJRT_Buffer_Type_S32, {e_pad}},       {rv_data, ftype, {e_pad, 3}},
      {&n32, PJRT_Buffer_Type_S32, {}},               {&e32, PJRT_Buffer_Type_S32, {}},
      {&l32, PJRT_Buffer_Type_S32, {}}};
  std::vector<PJRT_Buffer*> args(ins.size(), nullptr);
  for (size_t k = 0; k < ins.size(); ++k) {
    PJRT_Client_BufferFromHostBuffer_Args h;
    std::memset(&h, 0, sizeof(h));
    h.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    h.client = client_;
    h.data = ins[k].data;
    h.type = ins[k].type;
    h.dims = ins[k].dims.data();
    h.num_dims = ins[k].dims.size();
    h.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    h.device = device_;
    check(api_, api_->PJRT_Client_BufferFromHostBuffer(&h), "BufferFromHostBuffer");
    await_and_destroy(api_, h.done_with_host_buffer, "host->device transfer");
    args[k] = h.buffer;
  }

  // ---- execute
  PJRT_Buffer* outs[3] = {nullptr, nullptr, nullptr};
  PJRT_Buffer* const* arg_list[1] = {args.data()};
  PJRT_Buffer** out_list[1] = {outs};
  PJRT_Event* done[1] = {nullptr};
  PJRT_ExecuteOptions opts;
  std::memset(&opts, 0, sizeof(opts));
  opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  PJRT_LoadedExecutable_Execute_Args ex;
  std::memset(&ex, 0, sizeof(ex));
  ex.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  ex.executable = bp->exe;
  ex.options = &opts;
  ex.argument_lists = arg_list;
  ex.num_devices = 1;
  ex.num_args = args.size();
  ex.output_lists = out_list;
  ex.device_complete_events = done;
  check(api_, api_->PJRT_LoadedExecutable_Execute(&ex), "Execute");
  await_and_destroy(api_, done[0], "execute");
  for (auto* a : args) destroy_buffer(api_, a);

  // ---- outputs -> host
  double et = 0.0;
  e_atom_pad.assign(n_pad, 0.0);
  dEdr_pad.assign(3 * static_cast<size_t>(e_pad), 0.0);
  float et32 = 0.0f;
  std::vector<float> ea32, g32;
  void* dst[3] = {&et, e_atom_pad.data(), dEdr_pad.data()};
  size_t sz[3] = {sizeof(double), sizeof(double) * n_pad, sizeof(double) * 3 * static_cast<size_t>(e_pad)};
  if (f32_) {
    ea32.assign(n_pad, 0.0f);
    g32.assign(3 * static_cast<size_t>(e_pad), 0.0f);
    dst[0] = &et32; dst[1] = ea32.data(); dst[2] = g32.data();
    sz[0] = sizeof(float); sz[1] = sizeof(float) * n_pad; sz[2] = sizeof(float) * 3 * static_cast<size_t>(e_pad);
  }
  for (int k = 0; k < 3; ++k) {
    PJRT_Buffer_ToHostBuffer_Args t;
    std::memset(&t, 0, sizeof(t));
    t.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    t.src = outs[k];
    t.dst = dst[k];
    t.dst_size = sz[k];
    check(api_, api_->PJRT_Buffer_ToHostBuffer(&t), "ToHostBuffer");
    await_and_destroy(api_, t.event, "device->host transfer");
    destroy_buffer(api_, outs[k]);
  }
  if (f32_) {
    et = et32;
    std::copy(ea32.begin(), ea32.end(), e_atom_pad.begin());
    std::copy(g32.begin(), g32.end(), dEdr_pad.begin());
  }
  return et;
}

// ----------------------------------------------------------------------------------------------- staged mode
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

PJRT_Buffer* Model::upload(const void* data, int type, const std::vector<int64_t>& dims) {
  const auto t0 = Clock::now();
  PJRT_Client_BufferFromHostBuffer_Args h;
  std::memset(&h, 0, sizeof(h));
  h.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  h.client = client_;
  h.data = data;
  h.type = static_cast<PJRT_Buffer_Type>(type);
  h.dims = dims.data();
  h.num_dims = dims.size();
  h.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
  h.device = device_;
  check(api_, api_->PJRT_Client_BufferFromHostBuffer(&h), "BufferFromHostBuffer");
  await_and_destroy(api_, h.done_with_host_buffer, "host->device transfer");
  if (timing_) {    // wait until the data is on the device, so that t_up is the real transfer time
    PJRT_Buffer_ReadyEvent_Args r;
    std::memset(&r, 0, sizeof(r));
    r.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    r.buffer = h.buffer;
    check(api_, api_->PJRT_Buffer_ReadyEvent(&r), "Buffer_ReadyEvent");
    await_and_destroy(api_, r.event, "buffer ready");
  }
  t_up += secs(t0);
  return h.buffer;
}

PJRT_Buffer* Model::zeros(int64_t rows, int64_t cols) {
  for (auto& z : zeros_)
    if (z.first.first == rows && z.first.second == cols) return z.second;
  std::vector<double> v(static_cast<size_t>(rows) * cols, 0.0);
  PJRT_Buffer* b = upload_f(v.data(), {rows, cols});
  zeros_.push_back({{rows, cols}, b});
  return b;
}

PJRT_Buffer* Model::upload_f(const double* data, const std::vector<int64_t>& dims) {
  if (!f32_) return upload(data, PJRT_Buffer_Type_F64, dims);
  size_t n = 1;
  for (auto d : dims) n *= static_cast<size_t>(d);
  std::vector<float> v(data, data + n);
  return upload(v.data(), PJRT_Buffer_Type_F32, dims);   // synchronous: v may go out of scope afterwards
}

void Model::download_f(PJRT_Buffer* b, double* dst, size_t count) {
  const auto t0 = Clock::now();
  std::vector<float> v32;
  PJRT_Buffer_ToHostBuffer_Args t;
  std::memset(&t, 0, sizeof(t));
  t.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  t.src = b;
  if (f32_) {
    v32.resize(count);
    t.dst = v32.data();
    t.dst_size = sizeof(float) * count;
  } else {
    t.dst = dst;
    t.dst_size = sizeof(double) * count;
  }
  check(api_, api_->PJRT_Buffer_ToHostBuffer(&t), "ToHostBuffer");
  await_and_destroy(api_, t.event, "device->host transfer");
  if (f32_) std::copy(v32.begin(), v32.end(), dst);
  t_down += secs(t0);
}

std::vector<PJRT_Buffer*> Model::run(PJRT_LoadedExecutable* exe, const std::vector<PJRT_Buffer*>& args, int n_out) {
  const auto t0 = Clock::now();
  std::vector<PJRT_Buffer*> outs(n_out, nullptr);
  PJRT_Buffer* const* arg_list[1] = {args.data()};
  PJRT_Buffer** out_list[1] = {outs.data()};
  PJRT_Event* done[1] = {nullptr};
  PJRT_ExecuteOptions opts;
  std::memset(&opts, 0, sizeof(opts));
  opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  PJRT_LoadedExecutable_Execute_Args ex;
  std::memset(&ex, 0, sizeof(ex));
  ex.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  ex.executable = exe;
  ex.options = &opts;
  ex.argument_lists = arg_list;
  ex.num_devices = 1;
  ex.num_args = args.size();
  ex.output_lists = out_list;
  ex.device_complete_events = done;
  check(api_, api_->PJRT_LoadedExecutable_Execute(&ex), "Execute (staged)");
  await_and_destroy(api_, done[0], "execute (staged)");
  t_exec += secs(t0);
  ++n_exec;
  return outs;
}

void Model::staged_release() {
  for (auto* b : st_graph_) destroy_buffer(api_, b);
  for (auto* b : st_hs_) destroy_buffer(api_, b);
  for (auto& v : st_resbuf_)
    for (auto* b : v) destroy_buffer(api_, b);
  st_resbuf_.clear();
  destroy_buffer(api_, st_a_);
  destroy_buffer(api_, st_cta_);
  destroy_buffer(api_, st_ctrv_);
  destroy_buffer(api_, st_ctadd_);
  st_graph_.clear();
  st_hs_.clear();
  st_a_ = st_cta_ = st_ctrv_ = st_ctadd_ = nullptr;
  sb_ = nullptr;
}

int Model::staged_n_pad() const { return sb_ ? sb_->n_pad : 0; }

void Model::staged_begin(int n, int e, int n_local, const int* species, const int* receivers, const int* senders,
                         const double* rvec) {
  staged_release();
  if (staged_.empty()) throw std::runtime_error("racev2: " + dir_ + " has no staged (halo exchange) functions; "
                                                "re-export with --staged-buckets");
  for (auto& b : staged_)
    if (n < b.n_pad && e <= b.e_pad && n_local <= b.k_pad) { sb_ = &b; break; }
  if (!sb_) {
    std::ostringstream ss;
    ss << "racev2: no staged bucket fits " << n << " atoms (" << n_local << " owned) / " << e << " edges (largest "
       << staged_.back().n_pad
       << " / " << staged_.back().e_pad << "); export a larger --staged-buckets entry";
    throw std::runtime_error(ss.str());
  }
  const int np = sb_->n_pad, ep = sb_->e_pad;
  std::vector<int> sp(np, 0), rc(ep, n), sd(ep, n);
  std::vector<double> rv(3 * static_cast<size_t>(ep), 0.0);
  std::copy(species, species + n, sp.begin());
  std::copy(receivers, receivers + e, rc.begin());
  std::copy(senders, senders + e, sd.begin());
  std::copy(rvec, rvec + 3 * static_cast<size_t>(e), rv.begin());
  const int32_t n32 = n, e32 = e, l32 = n_local;
  st_graph_ = {upload(sp.data(), PJRT_Buffer_Type_S32, {np}), upload(rc.data(), PJRT_Buffer_Type_S32, {ep}),
               upload(sd.data(), PJRT_Buffer_Type_S32, {ep}), upload_f(rv.data(), {ep, 3}),
               upload(&n32, PJRT_Buffer_Type_S32, {}), upload(&e32, PJRT_Buffer_Type_S32, {}),
               upload(&l32, PJRT_Buffer_Type_S32, {})};
  st_dummy_ = zeros(np, 1);                 // cached, not released per evaluation
  st_hs_.assign(n_layers_ + 1, nullptr);
  st_resbuf_.assign(n_layers_, {});
  st_e_ = e;
  const std::string base = sb_->stem;
  if (!sb_->prelude) {
    sb_->prelude = compile_file(base + "_prelude.mlirbc");
    sb_->prelude_vjp = compile_file(base + "_prelude_vjp.mlirbc");
    for (int i = 0; i < n_layers_; ++i) {
      const std::string li = base + "_layer" + std::to_string(i);
      if (st_res_) {
        sb_->layer_fwdres[i] = compile_file(li + "_fwdres.mlirbc");
        sb_->layer_bwdres[i] = compile_file(li + "_bwdres.mlirbc");
      } else {
        sb_->layer[i] = compile_file(li + ".mlirbc");
        sb_->layer_vjp[i] = compile_file(li + "_vjp.mlirbc");
      }
    }
    if (verbose_) std::fprintf(stderr, "racev2: compiled staged bucket n_pad %d e_pad %d k_pad %d (%d functions%s) for "
                               "%d nodes (%d owned) / %d edges\n", np, ep, sb_->k_pad, 2 + 2 * n_layers_,
                               st_res_ ? ", saved residuals" : "", n, n_local, e);
  }
}

double Model::staged_prelude(std::vector<double>& h0) {
  const size_t np = sb_->n_pad;
  auto outs = run(sb_->prelude, st_graph_, 3);
  h0.resize(np * st_dims_[0]);
  download_f(outs[0], h0.data(), h0.size());
  double e = 0.0;
  download_f(outs[2], &e, 1);
  destroy_buffer(api_, outs[0]);
  destroy_buffer(api_, outs[2]);
  st_a_ = outs[1];                                   // atomic basis stays on the device
  return e;
}

double Model::staged_layer(int i, const double* h_prev, std::vector<double>& h_new) {
  const int64_t np = sb_->n_pad;
  destroy_buffer(api_, st_hs_[i]);
  st_hs_[i] = upload_f(h_prev, {np, st_dims_[i]});
  PJRT_Buffer* hpp = (st_cheb_ && i >= 2) ? st_hs_[i - 1] : st_dummy_;
  std::vector<PJRT_Buffer*> args = {st_hs_[i], hpp, st_a_};
  args.insert(args.end(), st_graph_.begin(), st_graph_.end());
  auto outs = st_res_ ? run(sb_->layer_fwdres[i], args, 2 + st_res_counts_[i]) : run(sb_->layer[i], args, 2);
  if (st_res_) st_resbuf_[i].assign(outs.begin() + 2, outs.end());   // released after the layer's backward
  h_new.resize(static_cast<size_t>(np) * st_dims_[i + 1]);
  download_f(outs[0], h_new.data(), h_new.size());
  double e = 0.0;
  download_f(outs[1], &e, 1);
  destroy_buffer(api_, outs[0]);
  destroy_buffer(api_, outs[1]);
  return e;
}

void Model::staged_layer_vjp(int i, const double* ct_h_new, std::vector<double>& ct_h_prev) {
  const int64_t np = sb_->n_pad, ep = sb_->e_pad;
  const bool use_pp = st_cheb_ && i >= 2;
  PJRT_Buffer* hpp = use_pp ? st_hs_[i - 1] : st_dummy_;
  PJRT_Buffer* ct = upload_f(ct_h_new, {np, st_dims_[i + 1]});
  PJRT_Buffer* add = st_ctadd_ ? st_ctadd_ : zeros(np, st_dims_[i]);
  PJRT_Buffer* cta = st_cta_ ? st_cta_ : zeros(np, st_dim_a_);
  PJRT_Buffer* ctrv = st_ctrv_ ? st_ctrv_ : zeros(ep, 3);
  std::vector<PJRT_Buffer*> args;
  if (st_res_) {
    args = st_resbuf_[i];
  } else {
    args = {st_hs_[i], hpp, st_a_};
    args.insert(args.end(), st_graph_.begin(), st_graph_.end());
  }
  args.insert(args.end(), {ct, add, cta, ctrv});
  auto outs = run(st_res_ ? sb_->layer_bwdres[i] : sb_->layer_vjp[i], args, 4);
  destroy_buffer(api_, ct);
  if (st_res_) {
    for (auto* b : st_resbuf_[i]) destroy_buffer(api_, b);
    st_resbuf_[i].clear();
  }
  destroy_buffer(api_, st_ctadd_);
  destroy_buffer(api_, st_cta_);
  destroy_buffer(api_, st_ctrv_);
  ct_h_prev.resize(static_cast<size_t>(np) * st_dims_[i]);
  download_f(outs[0], ct_h_prev.data(), ct_h_prev.size());
  destroy_buffer(api_, outs[0]);
  if (use_pp) st_ctadd_ = outs[1];          // cotangent of h_{i-1}: added in the vjp of layer i-1
  else { destroy_buffer(api_, outs[1]); st_ctadd_ = nullptr; }
  st_cta_ = outs[2];
  st_ctrv_ = outs[3];
}

void Model::staged_finish(std::vector<double>& dEdr) {
  const int64_t np = sb_->n_pad, ep = sb_->e_pad;
  std::vector<PJRT_Buffer*> args = st_graph_;
  args.push_back(st_cta_ ? st_cta_ : zeros(np, st_dim_a_));
  args.push_back(st_ctrv_ ? st_ctrv_ : zeros(ep, 3));
  auto outs = run(sb_->prelude_vjp, args, 1);
  std::vector<double> g(3 * static_cast<size_t>(ep));
  download_f(outs[0], g.data(), g.size());
  destroy_buffer(api_, outs[0]);
  dEdr.assign(g.begin(), g.begin() + 3 * static_cast<size_t>(st_e_));
  staged_release();
}

}  // namespace racev2
