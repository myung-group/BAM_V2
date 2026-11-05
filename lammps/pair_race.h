/* -*- c++ -*- ----------------------------------------------------------
   pair_style race: RACEv2 machine-learning potential (JAX-trained, exported to StableHLO by
   bam_omat24/export_lammps.py) evaluated on GPUs through the PJRT C API (racev2_pjrt.h).

     pair_style race <xla_cuda_plugin.so> <export_dir> [gpu auto|<index>] [mode auto|owner|ghost]
     pair_coeff * * <element of type 1> <element of type 2> ...

   modes
     owner : one MPI rank; the graph has the owned atoms only, every ghost neighbour is mapped to its owned image
             (periodic graph as in training).  Fastest.
     ghost : any number of ranks / GPUs / nodes; the graph has owned + ghost atoms, the ghost cutoff is raised to
             n_layers * cutoff + skin so the owned atoms see their full receptive field; the energy is the sum over
             the owned atoms, forces on ghosts are sent back to their owners by LAMMPS (newton pair on).
     auto  : owner for one rank, ghost otherwise.
   gpu auto: device = node-local rank (SLURM_LOCALID / OMPI_COMM_WORLD_LOCAL_RANK / MPI_LOCALRANKID / PMI_LOCAL_RANK)
             modulo the visible GPUs; with one visible GPU per rank (recommended) it is 0.
   units   : metal (eV, A) or real (kcal/mol, A; energies, forces and virial scaled).
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(race,PairRACE);
// clang-format on
#else

#ifndef LMP_PAIR_RACE_H
#define LMP_PAIR_RACE_H

#include "pair.h"

#include <memory>
#include <string>
#include <vector>

namespace racev2 {
class Model;
}

namespace LAMMPS_NS {

class PairRACE : public Pair {
 public:
  PairRACE(class LAMMPS *);
  ~PairRACE() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  void allocate();
  int local_rank() const;

  std::unique_ptr<racev2::Model> model;
  std::string plugin_so, export_dir;
  int gpu_index;              // -1 = auto
  int mode;                   // 0 auto, 1 owner, 2 ghost
  bool ghost_mode;            // resolved in init_style
  double cutoff, cutoffsq, energy_unit;
  std::vector<int> type2species;    // LAMMPS type (1-based) -> model species index, -1 = unmapped
  std::vector<double> species_e0;   // per-species reference energy added to the model energy (eV)
  // work arrays
  std::vector<int> species, receivers, senders;
  std::vector<double> rvec, e_atom, dEdr;
};

}    // namespace LAMMPS_NS

#endif
#endif
