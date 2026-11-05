// Runs the case_<k>.bin test cases written by `bam_omat24.export_lammps --cases` through the C++ runtime and
// compares with the outputs of the exported function computed by JAX; also times repeated calls.
//   test_pjrt <xla_cuda_plugin.so> <export_dir> [n_cases]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "racev2_pjrt.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <xla_cuda_plugin.so> <export_dir> [n_cases]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[2];
  const int n_cases = argc > 3 ? std::atoi(argv[3]) : 3;
  racev2::Model model(argv[1], dir);
  int fails = 0;
  for (int k = 0; k < n_cases; ++k) {
    std::ifstream f(dir + "/case_" + std::to_string(k) + ".bin", std::ios::binary);
    if (!f) { std::fprintf(stderr, "no case_%d.bin\n", k); break; }
    int32_t hdr[4];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    const int n_pad = hdr[0], e_pad = hdr[1], n = hdr[2], e = hdr[3];
    std::vector<int> sp(n_pad), rc(e_pad), sd(e_pad);
    std::vector<double> rv(3 * e_pad), ea_ref(n_pad), g_ref(3 * e_pad);
    double et_ref = 0.0;
    f.read(reinterpret_cast<char*>(sp.data()), 4 * n_pad);
    f.read(reinterpret_cast<char*>(rc.data()), 4 * e_pad);
    f.read(reinterpret_cast<char*>(sd.data()), 4 * e_pad);
    f.read(reinterpret_cast<char*>(rv.data()), 8 * 3 * e_pad);
    f.read(reinterpret_cast<char*>(&et_ref), 8);
    f.read(reinterpret_cast<char*>(ea_ref.data()), 8 * n_pad);
    f.read(reinterpret_cast<char*>(g_ref.data()), 8 * 3 * e_pad);

    std::vector<double> ea, g;
    double et = model.compute(n, e, sp.data(), rc.data(), sd.data(), rv.data(), ea, g);
    double de = std::fabs(et - et_ref), dea = 0.0, dg = 0.0, gmax = 0.0;
    for (int i = 0; i < n; ++i) dea = std::max(dea, std::fabs(ea[i] - ea_ref[i]));
    for (int i = 0; i < 3 * e; ++i) {
      dg = std::max(dg, std::fabs(g[i] - g_ref[i]));
      gmax = std::max(gmax, std::fabs(g_ref[i]));
    }
    // forces from dE/dr: F_i += g, F_j -= g
    std::vector<double> F(3 * n, 0.0), Fr(3 * n, 0.0);
    for (int q = 0; q < e; ++q)
      for (int c = 0; c < 3; ++c) {
        F[3 * rc[q] + c] += g[3 * q + c]; F[3 * sd[q] + c] -= g[3 * q + c];
        Fr[3 * rc[q] + c] += g_ref[3 * q + c]; Fr[3 * sd[q] + c] -= g_ref[3 * q + c];
      }
    double df = 0.0;
    for (int i = 0; i < 3 * n; ++i) df = std::max(df, std::fabs(F[i] - Fr[i]));
    const bool ok = de < 1e-8 && dea < 1e-8 && df < 1e-8;
    fails += !ok;
    std::printf("case %d: %d atoms %d edges bucket %d/%d | E %.10f (ref %.10f) |dE| %.2e |dE_atom| %.2e "
                "|d dE/dr| %.2e (max |dE/dr| %.3f) |dF| %.2e  %s\n",
                k, n, e, n_pad, e_pad, et, et_ref, de, dea, dg, gmax, df, ok ? "OK" : "MISMATCH");
    // timing (compiled already)
    const int reps = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) model.compute(n, e, sp.data(), rc.data(), sd.data(), rv.data(), ea, g);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    std::printf("        %.2f ms per call (energy + dE/dr, incl. host<->device copies)\n", ms);
  }
  std::printf("%s\n", fails ? "SOME CASES FAILED" : "ALL CASES OK");
  return fails ? 1 : 0;
}
