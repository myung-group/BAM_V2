// RACEv2 C++ runtime over the PJRT C API (see racev2_pjrt.h).
#include "racev2_pjrt.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "third_party/pjrt_c_api.h"

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
  while (std::getline(meta, line)) {
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key == "cutoff") ls >> cutoff_;
    else if (key == "n_layers") ls >> n_layers_;
    else if (key == "atomic_numbers") { int z; while (ls >> z) z_.push_back(z); }
    else if (key == "e0") { double v; while (ls >> v) e0_.push_back(v); }
    else if (key == "bucket") { Bucket b; ls >> b.n_pad >> b.e_pad >> b.file; buckets_.push_back(b); }
  }
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

  PJRT_Client_Create_Args cc;
  std::memset(&cc, 0, sizeof(cc));
  cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
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
  if (verbose_)
    std::fprintf(stderr, "racev2: %zu GPU(s), using %d; %zu buckets, cutoff %.3f A, %zu species\n",
                 ad.num_addressable_devices, device, buckets_.size(), cutoff_, z_.size());
}

Model::~Model() {
  if (!api_) return;
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

void Model::compile(Bucket& b) {
  std::string code = read_file(dir_ + "/" + b.file);
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
  check(api_, api_->PJRT_Client_Compile(&ca), "PJRT_Client_Compile");
  b.exe = ca.executable;
  if (verbose_) std::fprintf(stderr, "racev2: compiled bucket n_pad %d e_pad %d\n", b.n_pad, b.e_pad);
}

Bucket& Model::bucket_for(int n, int e) {
  for (auto& b : buckets_)
    if (n < b.n_pad && e <= b.e_pad) return b;
  std::ostringstream ss;
  ss << "racev2: no exported bucket fits " << n << " atoms / " << e << " edges (largest " << buckets_.back().n_pad
     << " / " << buckets_.back().e_pad << "); export a larger bucket";
  throw std::runtime_error(ss.str());
}

double Model::compute(int n, int e, const int* species, const int* receivers, const int* senders, const double* rvec,
                      std::vector<double>& e_atom, std::vector<double>& dEdr) {
  Bucket& b = bucket_for(n, e);
  std::vector<int> sp(b.n_pad, 0), rc(b.e_pad, n), sd(b.e_pad, n);
  std::vector<double> rv(3 * static_cast<size_t>(b.e_pad), 0.0);
  std::copy(species, species + n, sp.begin());
  std::copy(receivers, receivers + e, rc.begin());
  std::copy(senders, senders + e, sd.begin());
  std::copy(rvec, rvec + 3 * static_cast<size_t>(e), rv.begin());
  std::vector<double> ea, g;
  double et = compute_padded(b.n_pad, b.e_pad, n, e, sp.data(), rc.data(), sd.data(), rv.data(), ea, g);
  e_atom.assign(ea.begin(), ea.begin() + n);
  dEdr.assign(g.begin(), g.begin() + 3 * static_cast<size_t>(e));
  return et;
}

double Model::compute_padded(int n_pad, int e_pad, int n, int e, const int* species, const int* receivers,
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
  const int32_t n32 = n, e32 = e;
  std::vector<In> ins = {
      {species, PJRT_Buffer_Type_S32, {n_pad}},       {receivers, PJRT_Buffer_Type_S32, {e_pad}},
      {senders, PJRT_Buffer_Type_S32, {e_pad}},       {rvec, PJRT_Buffer_Type_F64, {e_pad, 3}},
      {&n32, PJRT_Buffer_Type_S32, {}},               {&e32, PJRT_Buffer_Type_S32, {}}};
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
  void* dst[3] = {&et, e_atom_pad.data(), dEdr_pad.data()};
  size_t sz[3] = {sizeof(double), sizeof(double) * n_pad, sizeof(double) * 3 * static_cast<size_t>(e_pad)};
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
  return et;
}

}  // namespace racev2
