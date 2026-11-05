/* -*- c++ -*- ----------------------------------------------------------
   pair_style racev2: RACEv2 machine-learning potential (JAX-trained, exported to StableHLO by
   bam_omat24/export_lammps.py) evaluated on one GPU through the PJRT C API (racev2_pjrt.h).

     pair_style racev2 <xla_cuda_plugin.so> <export_dir> [gpu_index]
     pair_coeff * * <element of type 1> <element of type 2> ...

   Step 3 of docs/RACEv2_lammps_export_plan_2026-11-06.md: one MPI rank (the periodic graph is built from the
   owned atoms; every ghost is mapped back to its owner with atom->map), metal units, atom_modify map yes.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(racev2,PairRACEv2);
// clang-format on
#else

#ifndef LMP_PAIR_RACEV2_H
#define LMP_PAIR_RACEV2_H

#include "pair.h"

#include <memory>
#include <string>
#include <vector>

namespace racev2 {
class Model;
}

namespace LAMMPS_NS {

class PairRACEv2 : public Pair {
 public:
  PairRACEv2(class LAMMPS *);
  ~PairRACEv2() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  void allocate();

  std::unique_ptr<racev2::Model> model;
  std::string plugin_so, export_dir;
  int gpu_index;
  double cutoff, cutoffsq;
  std::vector<int> type2species;    // LAMMPS type (1-based) -> model species index, -1 = unmapped
  std::vector<double> species_e0;   // per-species reference energy added to the model energy
  // work arrays
  std::vector<int> species, receivers, senders;
  std::vector<double> rvec, e_atom, dEdr;
};

}    // namespace LAMMPS_NS

#endif
#endif
