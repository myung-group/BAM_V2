# LAMMPS speed-up for RACEv2: notes of 2026-11-13 (deferred, nothing run)

Recorded on the user's request; the project is to be taken up later. Based on reading `lammps/pair_race.cpp`,
`lammps/racev2_pjrt.*`, `bam_omat24/export_lammps.py`, `docs/HANDOFF.md`, `docs/torch_port_scope.md` and on the
SevenNet paper (Park et al., JCTC 2024, arXiv:2402.03789). Measured numbers below are from the earlier documents
(previous account); nothing was measured on 2026-11-13.

## What exists

`pair_style race <plugin.so> <export dir> [gpu ...] [mode auto|owner|ghost|halo]`

| mode | scheme | status |
|---|---|---|
| owner | one MPI rank, periodic graph as in training | validated against Python |
| ghost | ghost shell widened to n_layers x cutoff (5 x 5 A) | out of memory above ~1k atoms per rank |
| halo | ghost shell = one cutoff; node features are exchanged between ranks after every layer, cotangents are returned in the backward pass | validated on 1 / 2 / 4 ranks (dF <= 1e-14 eV/A vs owner); weak scaling 88 % from 1 to 8 GPUs (42,496 LiPS atoms) |

The halo mode is the algorithm of SevenNet's `pair_style e3gnn/parallel`: one exported function per layer, a forward
communication of the ghost rows after each layer (T - 1 needed), a reverse communication of dE/dh per layer in the
backward pass, and the usual reverse communication of ghost forces. The RACE recipe used for MPtrj (5 layers, x_mode
basis, Chebyshev residual, hill norm, gated readout, UFF repulsion) needs from neighbours only the feature vector
h_t (128 x 16 = 2048 values per atom, float64); everything else is local to the receiving atom, and there is no
graph-level normalisation or long-range term. So the scheme applies to the model as it is.

## The gap

Halo is 3.3x slower than owner on one GPU; no timing breakdown exists. Candidates seen in the code:

1. The backward pass recomputes every layer (`layer{i}_vjp` calls `jax.vjp` on the layer). The saved-residual
   variant (`--staged-residuals`, `RACEV2_STAGED_RESIDUALS`) and the owned-row buckets (`--staged-buckets n:e:k`,
   `race_nnx.NODE_LIMIT`) are written but have never been run (`runs/diag/lammps_halo_res.sh` not submitted).
2. Every staged function re-runs the prelude, so the edge radial and spherical-harmonic features are recomputed in
   each of the ~10 calls of a step.
3. Each layer output and each cotangent is downloaded to the host as a full N x 2048 array, communicated and uploaded
   again; only the border and ghost rows are needed.
4. Redundant transfers: forward communication of h_0 (a function of the species only), download of h_L, upload of a
   zero cotangent for h_L, download of the cotangent of h_0.

## Plan (when resumed)

0. Fetch the LAMMPS source (the documents used `stable_30Sep2026_update1`) and build with `lammps/build_lammps.sh`
   (serial and MPI). On account t510a13 there is no LAMMPS source, build, module or exported model.
1. Export a trained model with staged buckets (and residuals) and check it against Python (`--check`, GPU job).
2. Timing breakdown with `RACEV2_TIMING=1` (execution / upload / download / MPI) for owner, halo (recompute) and halo
   (saved residuals) on the same system, one and several GPUs.
3. Optimise in the order of the measured cost (items 1-4 above). Figures of merit: halo / owner time ratio (now 3.3)
   and the weak-scaling efficiency.

Alternative kept open in `docs/torch_port_scope.md`: a PyTorch port with `pair_style mliap unified` + KOKKOS
(estimated 10-15 person-days). Run the already written optimisations first.
