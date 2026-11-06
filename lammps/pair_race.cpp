// clang-format off
/* ----------------------------------------------------------------------
   pair_style race (see pair_race.h)
------------------------------------------------------------------------- */

#include "pair_race.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include "racev2_pjrt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
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
const double kEvToKcalMol = 23.060547830619;    // 1 eV in kcal/mol (CODATA 2018)
}    // namespace

/* ---------------------------------------------------------------------- */

PairRACE::PairRACE(LAMMPS *lmp) :
    Pair(lmp), gpu_index(-1), mode(0), ghost_mode(false), halo_mode(false), cutoff(0.0), cutoffsq(0.0), energy_unit(1.0)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;    // virial accumulated explicitly from the edge vectors
}

PairRACE::~PairRACE()
{
  if (model && halo_mode && std::getenv("RACEV2_TIMING"))
    utils::logmesg(lmp, "pair_style race halo timing (rank {}): executions {:.3f} s ({}), uploads {:.3f} s, "
                        "downloads {:.3f} s, MPI comm {:.3f} s\n", comm->me, model->t_exec, model->n_exec,
                   model->t_up, model->t_down, t_comm);
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutghost);
  }
}

void PairRACE::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;
  memory->create(setflag, n, n, "pair:setflag");
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(cutghost, n, n, "pair:cutghost");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;
}

int PairRACE::local_rank() const
{
  for (const char *v : {"SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "PMI_LOCAL_RANK"}) {
    const char *s = std::getenv(v);
    if (s && *s) return std::atoi(s);
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

void PairRACE::settings(int narg, char **arg)
{
  if (narg < 2)
    error->all(FLERR, "Illegal pair_style race command: pair_style race <plugin.so> <export_dir> "
                      "[gpu auto|<index>] [mode auto|owner|ghost|halo]");
  plugin_so = arg[0];
  export_dir = arg[1];
  int iarg = 2;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "gpu") == 0 && iarg + 1 < narg) {
      gpu_index = (strcmp(arg[iarg + 1], "auto") == 0) ? -1 : utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "mode") == 0 && iarg + 1 < narg) {
      if (strcmp(arg[iarg + 1], "auto") == 0) mode = 0;
      else if (strcmp(arg[iarg + 1], "owner") == 0) mode = 1;
      else if (strcmp(arg[iarg + 1], "ghost") == 0) mode = 2;
      else if (strcmp(arg[iarg + 1], "halo") == 0) mode = 3;
      else error->all(FLERR, "pair_style race: mode must be auto, owner, ghost or halo");
      iarg += 2;
    } else if (iarg == 2 && narg == 3) {    // backward compatible: third positional argument = gpu index
      gpu_index = utils::inumeric(FLERR, arg[iarg], false, lmp);
      iarg += 1;
    } else {
      error->all(FLERR, "pair_style race: unknown keyword {}", arg[iarg]);
    }
  }
}

void PairRACE::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg != 2 + atom->ntypes)
    error->all(FLERR, "pair_coeff * * needs one element symbol per atom type for pair_style race");
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_style race requires pair_coeff * *");

  if (!model) {
    int dev = gpu_index;
    if (dev < 0) dev = local_rank();    // modulo the visible GPUs inside Model (index check)
    try {
      // with auto: try the local rank, fall back to 0 when only one GPU is visible to this rank
      try {
        model = std::make_unique<racev2::Model>(plugin_so, export_dir, dev, comm->me == 0);
      } catch (std::exception &) {
        if (gpu_index >= 0 || dev == 0) throw;
        model = std::make_unique<racev2::Model>(plugin_so, export_dir, 0, comm->me == 0);
      }
    } catch (std::exception &ex) {
      error->one(FLERR, ex.what());
    }
  }
  cutoff = model->cutoff();
  cutoffsq = cutoff * cutoff;
  const auto &zs = model->atomic_numbers();
  species_e0 = model->atom_energy0();
  type2species.assign(atom->ntypes + 1, -1);
  for (int t = 1; t <= atom->ntypes; ++t) {
    const char *sym = arg[1 + t];
    if (strcmp(sym, "NULL") == 0) continue;
    int z = atomic_number(sym);
    int s = -1;
    for (int k = 0; k < (int) zs.size(); ++k)
      if (zs[k] == z) s = k;
    if (s < 0) error->all(FLERR, "pair_style race: element {} is not in the model", sym);
    type2species[t] = s;
  }
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      setflag[i][j] = (type2species[i] >= 0 && type2species[j] >= 0) ? 1 : 0;
}

void PairRACE::init_style()
{
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, "pair_style race requires an atom map (atom_modify map yes)");
  if (force->newton_pair == 0) error->all(FLERR, "pair_style race requires newton pair on");
  if (strcmp(update->unit_style, "metal") == 0) energy_unit = 1.0;
  else if (strcmp(update->unit_style, "real") == 0) energy_unit = kEvToKcalMol;
  else error->all(FLERR, "pair_style race supports metal and real units");

  halo_mode = (mode == 3) || (mode == 0 && comm->nprocs > 1 && model->has_staged());
  ghost_mode = (mode == 2) || (mode == 0 && comm->nprocs > 1 && !model->has_staged());
  if (mode == 1 && comm->nprocs > 1)
    error->all(FLERR, "pair_style race mode owner needs one MPI rank; use mode halo, ghost or auto");
  if (halo_mode && !model->has_staged())
    error->all(FLERR, "pair_style race mode halo: the export has no staged functions (export with --staged-buckets)");
  if (comm->me == 0)
    utils::logmesg(lmp, "pair_style race: mode {}{}\n", halo_mode ? "halo" : ghost_mode ? "ghost" : "owner",
                   halo_mode ? (model->staged_residuals() ? " (saved residuals)" : " (recompute in backward)") : "");
  comm_forward = comm_reverse = 0;
  if (halo_mode) {
    for (int d : model->staged_dims()) comm_forward = std::max(comm_forward, d);
    comm_reverse = comm_forward;
  }

  if (ghost_mode) {
    // every owned atom needs its full receptive field: n_layers message-passing steps of `cutoff`
    const double need = model->n_layers() * cutoff + neighbor->skin;
    if (comm->cutghostuser < need) {
      comm->cutghostuser = need;
      if (comm->me == 0)
        utils::logmesg(lmp, "pair_style race (ghost mode): ghost cutoff set to {:.3f} A "
                            "({} layers x {:.3f} A + skin)\n", need, model->n_layers(), cutoff);
    }
    ghostneigh = 1;
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  } else {
    ghostneigh = 0;
    neighbor->add_request(this, NeighConst::REQ_FULL);
  }
}

double PairRACE::init_one(int i, int j)
{
  if (ghostneigh) {
    cutghost[i][j] = cutoff;
    cutghost[j][i] = cutoff;
  }
  return cutoff;
}

/* ---------------------------------------------------------------------- */

void PairRACE::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (vflag_atom) error->all(FLERR, "pair_style race does not compute per-atom virials");
  if (halo_mode) {
    compute_halo(eflag, vflag);
    return;
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  tagint *tag = atom->tag;
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const int nnodes = ghost_mode ? nall : nlocal;

  species.resize(nnodes);
  for (int i = 0; i < nnodes; ++i) {
    species[i] = type2species[type[i]];
    if (species[i] < 0) error->one(FLERR, "pair_style race: atom of an unmapped type");
  }
  receivers.clear();
  senders.clear();
  rvec.clear();
  // owner mode: neighbour lists of the owned atoms, ghosts mapped to their owned image (one rank, periodic)
  // ghost mode: neighbour lists of owned and ghost atoms (REQ_GHOST), ghosts stay separate nodes
  const int ncentres = ghost_mode ? list->inum + list->gnum : list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  for (int ii = 0; ii < ncentres; ++ii) {
    const int i = ilist[ii];
    const int *jlist = firstneigh[i];
    for (int jj = 0; jj < numneigh[i]; ++jj) {
      const int j = jlist[jj] & NEIGHMASK;
      const double dx = x[j][0] - x[i][0], dy = x[j][1] - x[i][1], dz = x[j][2] - x[i][2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= cutoffsq) continue;
      int jn = j;
      if (!ghost_mode && j >= nlocal) {
        jn = atom->map(tag[j]);
        if (jn < 0 || jn >= nlocal) error->one(FLERR, "pair_style race: ghost atom without an owned image");
      }
      receivers.push_back(i);
      senders.push_back(jn);
      rvec.push_back(dx);
      rvec.push_back(dy);
      rvec.push_back(dz);
    }
  }
  const int ne = (int) receivers.size();

  double elocal = 0.0;
  try {
    elocal = model->compute(nnodes, ne, nlocal, species.data(), receivers.data(), senders.data(), rvec.data(),
                            e_atom, dEdr);
  } catch (std::exception &ex) {
    error->one(FLERR, ex.what());
  }

  // ---- forces (F_i += g, F_j -= g, g = dE_local/dr_ij, r_ij = x_j - x_i) and virial W = -sum r (x) g.
  // In ghost mode forces land on ghosts too and LAMMPS sums them into the owners (reverse communication).
  const double u = energy_unit;
  double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int q = 0; q < ne; ++q) {
    const double g0 = u * dEdr[3 * q], g1 = u * dEdr[3 * q + 1], g2 = u * dEdr[3 * q + 2];
    const double *r = &rvec[3 * q];
    const int i = receivers[q], j = senders[q];
    f[i][0] += g0; f[i][1] += g1; f[i][2] += g2;
    f[j][0] -= g0; f[j][1] -= g1; f[j][2] -= g2;
    v[0] -= r[0] * g0; v[1] -= r[1] * g1; v[2] -= r[2] * g2;
    v[3] -= r[0] * g1; v[4] -= r[0] * g2; v[5] -= r[1] * g2;
  }

  if (eflag_global) {
    double e0 = 0.0;
    for (int i = 0; i < nlocal; ++i) e0 += species_e0[species[i]];
    eng_vdwl += u * (elocal + e0);
  }
  if (eflag_atom)
    for (int i = 0; i < nlocal; ++i) eatom[i] += u * (e_atom[i] + species_e0[species[i]]);
  if (vflag_global)
    for (int k = 0; k < 6; ++k) virial[k] += v[k];
}

/* ----------------------------------------------------------------------
   halo mode: per-layer functions with forward / reverse communication of the node features
------------------------------------------------------------------------- */

void PairRACE::comm_rows(std::vector<double> &a, int dim, bool forward)
{
  comm_ptr = a.data();
  comm_dim = dim;
  if (forward) comm->forward_comm(this, dim);
  else comm->reverse_comm(this, dim);
  comm_ptr = nullptr;
}

void PairRACE::compute_halo(int /*eflag*/, int /*vflag*/)
{
  if (eflag_atom) error->all(FLERR, "pair_style race mode halo does not compute per-atom energies");
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;

  species.resize(nall);
  for (int i = 0; i < nall; ++i) {
    species[i] = type2species[type[i]];
    if (species[i] < 0) error->one(FLERR, "pair_style race: atom of an unmapped type");
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
      if (dx * dx + dy * dy + dz * dz >= cutoffsq) continue;
      receivers.push_back(i);
      senders.push_back(j);
      rvec.push_back(dx);
      rvec.push_back(dy);
      rvec.push_back(dz);
    }
  }
  const int ne = (int) receivers.size();
  const int L = model->n_layers();
  const auto &dims = model->staged_dims();
  double elocal = 0.0;
  try {
    model->staged_begin(nall, ne, nlocal, species.data(), receivers.data(), senders.data(), rvec.data());
    const size_t np = model->staged_n_pad();
    hs.resize(L + 1);
    // forward: h_0, then every layer output; ghost rows are refreshed from their owners before they are used
    elocal = model->staged_prelude(hs[0]);
    auto tc = std::chrono::steady_clock::now();
    comm_rows(hs[0], dims[0], true);
    t_comm += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count();
    for (int i = 0; i < L; ++i) {
      elocal += model->staged_layer(i, hs[i].data(), hs[i + 1]);
      tc = std::chrono::steady_clock::now();
      if (i + 1 < L) comm_rows(hs[i + 1], dims[i + 1], true);   // h_L is not an input of any layer
      t_comm += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count();
    }
    // backward: cotangents of h_i; the ghost-row cotangents belong to the owners (reverse comm), then zeroed
    cts.resize(L + 1);
    cts[L].assign(np * dims[L], 0.0);      // the energy does not depend on h_L
    for (int i = L - 1; i >= 0; --i) {
      model->staged_layer_vjp(i, cts[i + 1].data(), cts[i]);   // includes the Chebyshev term of layer i+1
      if (i > 0) {    // h_0 depends on the species only: its cotangent is not needed
        const auto tc = std::chrono::steady_clock::now();
        comm_rows(cts[i], dims[i], false);
        std::fill(cts[i].begin() + (size_t) nlocal * dims[i], cts[i].end(), 0.0);
        t_comm += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count();
      }
    }
    model->staged_finish(dEdr);
  } catch (std::exception &ex) {
    error->one(FLERR, ex.what());
  }

  const double u = energy_unit;
  double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int q = 0; q < ne; ++q) {
    const double g0 = u * dEdr[3 * q], g1 = u * dEdr[3 * q + 1], g2 = u * dEdr[3 * q + 2];
    const double *r = &rvec[3 * q];
    const int i = receivers[q], j = senders[q];
    f[i][0] += g0; f[i][1] += g1; f[i][2] += g2;
    f[j][0] -= g0; f[j][1] -= g1; f[j][2] -= g2;
    v[0] -= r[0] * g0; v[1] -= r[1] * g1; v[2] -= r[2] * g2;
    v[3] -= r[0] * g1; v[4] -= r[0] * g2; v[5] -= r[1] * g2;
  }
  if (eflag_global) {
    double e0 = 0.0;
    for (int i = 0; i < nlocal; ++i) e0 += species_e0[species[i]];
    eng_vdwl += u * (elocal + e0);
  }
  if (vflag_global)
    for (int k = 0; k < 6; ++k) virial[k] += v[k];
}

int PairRACE::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  // features are translation invariant: periodic images get the owner's row unchanged
  int m = 0;
  for (int k = 0; k < n; ++k) {
    const double *row = comm_ptr + (size_t) list[k] * comm_dim;
    for (int c = 0; c < comm_dim; ++c) buf[m++] = row[c];
  }
  return m;
}

void PairRACE::unpack_forward_comm(int n, int first, double *buf)
{
  int m = 0;
  for (int i = first; i < first + n; ++i) {
    double *row = comm_ptr + (size_t) i * comm_dim;
    for (int c = 0; c < comm_dim; ++c) row[c] = buf[m++];
  }
}

int PairRACE::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  for (int i = first; i < first + n; ++i) {
    const double *row = comm_ptr + (size_t) i * comm_dim;
    for (int c = 0; c < comm_dim; ++c) buf[m++] = row[c];
  }
  return m;
}

void PairRACE::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  for (int k = 0; k < n; ++k) {
    double *row = comm_ptr + (size_t) list[k] * comm_dim;
    for (int c = 0; c < comm_dim; ++c) row[c] += buf[m++];
  }
}
