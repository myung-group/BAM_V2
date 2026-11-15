"""Phase-2 reference values of BAM-torch for tests/test_thole_phase2.py: electrode / charge-group constraints,
contact charge relaxation, fragment unwrapping and the slab dipole correction (mode pbc_split).  Run in the env with
torch and BAM-torch:  python make_thole_torch_ref2.py <out.npz>"""
import sys, numpy as np, torch, ase, ase.neighborlist
from e3nn import o3
from bam_ufo.model.thole import TholeElectrostaticsBlock, unwrap_fragments
from bam_ufo.model.electrostatics import charge_constraint_groups
from bam_ufo.utils.contact_charges import contact_charges
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(11)
cutoff = 6.0
pos, cells, batch, pbc, ei, es = [], [], [], [], [], []
off = 0
def add(x, cell, periodic):
    global off
    a = ase.Atoms('H' * len(x), positions=x, cell=cell, pbc=periodic)
    if periodic:
        i, j, S = ase.neighborlist.neighbor_list('ijS', a, cutoff)
    else:
        i, j = ase.neighborlist.neighbor_list('ij', a, cutoff); S = np.zeros((len(i), 3), int)
    g = len(cells)
    pos.append(x); cells.append(cell); batch.extend([g] * len(x)); pbc.append(periodic)
    ei.append(np.stack([i, j]) + off); es.append(S); off += len(x)
c = np.diag([9.0, 8.1, 10.8]) + rng.normal(0, 0.5, (3, 3)); add(rng.random((18, 3)) @ c, c, True)            # dense cell
add(rng.normal(0, 2.5, (14, 3)) + 3.0, np.eye(3) * 50.0, False)                                                  # molecule
c = np.diag([20.0, 21.0, 24.0]) + rng.normal(0, 0.3, (3, 3))                                                     # two clusters
x = np.concatenate([np.array([19.6, 5.0, 5.0]) + rng.normal(0, 0.9, (5, 3)), np.array([9.0, 14.0, 23.7]) + rng.normal(0, 0.9, (4, 3))])
x = (np.linalg.solve(c.T, x.T).T % 1.0) @ c                                                                      # wrapped into the cell
add(x, c, True)
add(rng.normal(0, 2.0, (7, 3)), np.eye(3) * 50.0, False)
pos, cell, batch, pbc = np.concatenate(pos), np.stack(cells), np.array(batch), np.array(pbc)
ei, es = np.concatenate(ei, 1), np.concatenate(es)
N, G = len(pos), len(cell)
T = lambda x, **k: torch.tensor(x, **k)
# charge groups: the first atoms of every graph are the electrode / reservoir, the rest groups of <= 3 atoms
em = np.zeros(N, bool); cg = np.full(N, -1); fc = np.zeros(N)
for g in range(G):
    idx = np.nonzero(batch == g)[0]; ne = max(2, len(idx) // 3)
    em[idx[:ne]] = True
    rest = idx[ne:]
    for k in range(0, len(rest), 3):
        grp = rest[k:k + 3]; cg[grp] = k // 3
        if (k // 3) % 3 == 1:
            fc[grp] = (1.0 if (k // 3) % 2 else -1.0) / len(grp)
total_charge = np.array([0.0, 1.0, 0.0, -1.0])
raw = rng.normal(0, 0.3, N); fukui = rng.normal(0, 1.5, N)
group, ng, targets = charge_constraint_groups('electrode', T(ei), T(batch), G, T(total_charge), N, T(em), T(cg), T(fc))
q_group = TholeElectrostaticsBlock._equilibrate(T(raw), T(fukui), group, ng, targets, 'fukui', 'softmax')
src, dst = ei
d = pos[src] - pos[dst] - np.einsum('ei,eij->ej', es.astype(float), cell[batch[src]])
dist = np.linalg.norm(d, axis=1)
save = dict(pos=pos, cell=cell, batch=batch, pbc=pbc, edge_index=ei, edge_shifts=es, cutoff=cutoff, electrode_mask=em,
            charge_group=cg, fixed_charge=fc, total_charge=total_charge, raw=raw, fukui=fukui, q_group=q_group.numpy())
for pairs in ('charged', 'all'):
    p = T(pos, requires_grad=True)
    dd = p[T(src)] - p[T(dst)] - torch.einsum('ei,eij->ej', T(es.astype(float)), T(cell)[T(batch)[T(src)]])
    o = contact_charges(T(raw), T(fukui), group, targets, T(batch), T(ei), dd.norm(dim=1), 2.5, 4.0, 'fukui', 'softmax', pairs)
    save[f'q_contact_{pairs}'] = o['charges'].detach().numpy()
    w = T(rng.normal(size=N)) if pairs == 'charged' else T(np.linspace(-1, 1, N))
    save[f'w_{pairs}'] = w.numpy()
    save[f'grad_contact_{pairs}'] = torch.autograd.grad((o['charges'] * w).sum(), p)[0].numpy()
    print(pairs, 'max |q_contact - q_group|', float((o['charges'].detach() - q_group).abs().max()), 'contacts', int((o['transfer_contacts'] > 0).sum()))
offsets, perc = unwrap_fragments(T(ei), T(es.astype(float)), N)
save['offsets'] = offsets.numpy(); save['percolating'] = perc.numpy()
print('offsets nonzero atoms', int((offsets != 0).any(-1).sum()), 'percolating', int(perc.sum()))
# slab correction: neutral charges on the periodic graphs
q = q_group.numpy().copy()
for g in range(G):
    m = batch == g
    if pbc[g]: q[m] -= q[m].sum() / m.sum()
mu = rng.normal(0, 0.2, (N, 3)); alpha = rng.uniform(0.3, 3.0, N)
th = rng.normal(0, 0.3, (N, 3, 3)); th = 0.5 * (th + th.transpose(0, 2, 1)); th -= np.trace(th, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
save.update(q=q, mu=mu, theta=th, alpha=alpha)
slab_on = np.array([True, False, True, False])
for name, kw in {'slab': {}, 'slab_nomutual': dict(mutual_induction=False), 'slab_noind': dict(induction=False), 'slab_axis0': dict(slab_axis=0)}.items():
    blk = TholeElectrostaticsBlock(o3.Irreps('8x0e+8x1o+8x2e'), mode='pbc_split', cutoff=cutoff, quadrupoles=True, max_polarizability=4.0,
                                   min_polarizability=0.001, cg_max_iterations=300, cg_tolerance=1e-13, cap_induced_width=True,
                                   slab_correction=True, **kw)
    p = T(pos, requires_grad=True); c = T(cell, requires_grad=True)
    o = blk._forward_pbc_split(p, T(batch), c, G, T(q), T(mu), T(alpha), T(ei), T(es), torch.zeros(G, 3), offsets,
                               T(slab_on), T(pbc), T(th))
    gp, gc = torch.autograd.grad(o['coulomb_energy'].sum(), (p, c))
    for k in ('coulomb_energy', 'node_coulomb', 'slab_energy', 'induced_dipoles', 'total_dipole', 'normal_dipole'):
        save[f'{name}:{k}'] = o[k].detach().numpy()
    save[f'{name}:forces'] = -gp.numpy(); save[f'{name}:cellgrad'] = gc.numpy()
    print(name, 'E', o['coulomb_energy'].detach().numpy().round(6), 'Eslab', o['slab_energy'].detach().numpy().round(6), 'Mn', o['normal_dipole'].detach().numpy().round(4))
save['slab_on'] = slab_on
np.savez(sys.argv[1], **save)
