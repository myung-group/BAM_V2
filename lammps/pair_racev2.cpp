// clang-format off
/* ----------------------------------------------------------------------
   pair_style racev2 (see pair_racev2.h)
------------------------------------------------------------------------- */

#include "pair_racev2.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include "racev2_pjrt.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

namespace {
const char *kElements[] = {"X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al",
                           "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co",
                           "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb",
                           "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs",
                           "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
                           "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
                           "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk",
                           "Cf", "Es", "Fm", "Md", "No", "Lr"};
int atomic_number(const char *sym)
{
  for (int z = 1; z < (int) (sizeof(kElements) / sizeof(kElements[0])); ++z)
    if (strcmp(sym, kElements[z]) == 0) return z;
  return -1;
}
}    // namespace

/* ---------------------------------------------------------------------- */

PairRACEv2::PairRACEv2(LAMMPS *lmp) : Pair(lmp), gpu_index(0), cutoff(0.0), cutoffsq(0.0)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;    // forces live on owned atoms only: virial is accumulated explicitly
}

PairRACEv2::~PairRACEv2()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairRACEv2::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;
  memory->create(setflag, n, n, "pair:setflag");
  memory->create(cutsq, n, n, "pair:cutsq");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;
}

/* ---------------------------------------------------------------------- */

void PairRACEv2::settings(int narg, char **arg)
{
  if (narg < 2 || narg > 3)
    error->all(FLERR, "Illegal pair_style racev2 command: pair_style racev2 <plugin.so> <export_dir> [gpu]");
  plugin_so = arg[0];
  export_dir = arg[1];
  gpu_index = (narg == 3) ? utils::inumeric(FLERR, arg[2], false, lmp) : 0;
}

void PairRACEv2::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg != 2 + atom->ntypes)
    error->all(FLERR, "pair_coeff * * needs one element symbol per atom type for pair_style racev2");
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_style racev2 requires pair_coeff * *");

  if (!model) {
    try {
      model = std::make_unique<racev2::Model>(plugin_so, export_dir, gpu_index, comm->me == 0);
    } catch (std::exception &ex) {
      error->one(FLERR, ex.what());
    }
  }
  cutoff = model->cutoff();
  cutoffsq = cutoff * cutoff;
  const auto &zs = model->atomic_numbers();
  const auto &e0 = model->atom_energy0();
  species_e0 = e0;
  type2species.assign(atom->ntypes + 1, -1);
  for (int t = 1; t <= atom->ntypes; ++t) {
    const char *sym = arg[1 + t];
    if (strcmp(sym, "NULL") == 0) continue;
    int z = atomic_number(sym);
    int s = -1;
    for (int k = 0; k < (int) zs.size(); ++k)
      if (zs[k] == z) s = k;
    if (s < 0) error->all(FLERR, "pair_style racev2: element {} is not in the model", sym);
    type2species[t] = s;
  }
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      setflag[i][j] = (type2species[i] >= 0 && type2species[j] >= 0) ? 1 : 0;
}

void PairRACEv2::init_style()
{
  if (comm->nprocs != 1)
    error->all(FLERR, "pair_style racev2 currently runs on one MPI rank (message passing across subdomains "
                      "is not implemented)");
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, "pair_style racev2 requires an atom map (atom_modify map yes)");
  if (force->newton_pair == 0) error->all(FLERR, "pair_style racev2 requires newton pair on");
  if (strcmp(update->unit_style, "metal") != 0) error->all(FLERR, "pair_style racev2 requires metal units");
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairRACEv2::init_one(int, int)
{
  return cutoff;
}

/* ---------------------------------------------------------------------- */

void PairRACEv2::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (vflag_atom) error->all(FLERR, "pair_style racev2 does not compute per-atom virials");

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  tagint *tag = atom->tag;
  const int nlocal = atom->nlocal;

  // ---- graph of the owned atoms: every ghost neighbour is an image of an owned atom (one rank, periodic)
  species.resize(nlocal);
  for (int i = 0; i < nlocal; ++i) {
    species[i] = type2species[type[i]];
    if (species[i] < 0) error->one(FLERR, "pair_style racev2: atom of an unmapped type");
  }
  receivers.clear();
  senders.clear();
  rvec.clear();
  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  for (int ii = 0; ii < inum; ++ii) {
    const int i = ilist[ii];
    const int *jlist = firstneigh[i];
    for (int jj = 0; jj < numneigh[i]; ++jj) {
      const int j = jlist[jj] & NEIGHMASK;
      const double dx = x[j][0] - x[i][0], dy = x[j][1] - x[i][1], dz = x[j][2] - x[i][2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= cutoffsq) continue;
      int jo = (j < nlocal) ? j : atom->map(tag[j]);
      if (jo < 0 || jo >= nlocal) error->one(FLERR, "pair_style racev2: ghost atom without an owned image");
      receivers.push_back(i);
      senders.push_back(jo);
      rvec.push_back(dx);
      rvec.push_back(dy);
      rvec.push_back(dz);
    }
  }
  const int ne = (int) receivers.size();

  double etot = 0.0;
  try {
    etot = model->compute(nlocal, ne, species.data(), receivers.data(), senders.data(), rvec.data(), e_atom, dEdr);
  } catch (std::exception &ex) {
    error->one(FLERR, ex.what());
  }

  // ---- forces (F_i += g, F_j -= g, g = dE/dr_ij, r_ij = x_j - x_i) and virial W = -sum r (x) g
  double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int q = 0; q < ne; ++q) {
    const double *g = &dEdr[3 * q];
    const double *r = &rvec[3 * q];
    const int i = receivers[q], j = senders[q];
    f[i][0] += g[0]; f[i][1] += g[1]; f[i][2] += g[2];
    f[j][0] -= g[0]; f[j][1] -= g[1]; f[j][2] -= g[2];
    v[0] -= r[0] * g[0]; v[1] -= r[1] * g[1]; v[2] -= r[2] * g[2];
    v[3] -= r[0] * g[1]; v[4] -= r[0] * g[2]; v[5] -= r[1] * g[2];
  }

  if (eflag_global) {
    double e0 = 0.0;
    for (int i = 0; i < nlocal; ++i) e0 += species_e0[species[i]];
    eng_vdwl += etot + e0;
  }
  if (eflag_atom)
    for (int i = 0; i < nlocal; ++i) eatom[i] += e_atom[i] + species_e0[species[i]];
  if (vflag_global)
    for (int k = 0; k < 6; ++k) virial[k] += v[k];
}
