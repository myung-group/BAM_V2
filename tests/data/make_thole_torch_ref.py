"""Reference values of the BAM-torch Thole block (mode pbc_split) for tests/test_thole.py.  Run in the env with torch."""
import sys, numpy as np, torch, ase, ase.neighborlist
from e3nn import o3
from bam_ufo.model.thole import TholeElectrostaticsBlock
torch.set_default_dtype(torch.float64)
out_path = sys.argv[1]
rng = np.random.default_rng(7)
cutoff = 6.0

def system():
    pos, cells, batch, pbc, ei, es = [], [], [], [], [], []
    specs = [('p', 18, 9.0), ('m', 14, 0), ('p', 25, 11.0), ('m', 7, 0)]
    off = 0
    for g, (kind, n, L) in enumerate(specs):
        if kind == 'p':
            cell = np.diag([L, L * 0.9, L * 1.2]) + rng.normal(0, 0.6, (3, 3))
            x = rng.random((n, 3)) @ cell
            a = ase.Atoms('H' * n, positions=x, cell=cell, pbc=True)
            i, j, S = ase.neighborlist.neighbor_list('ijS', a, cutoff)
        else:
            cell = np.eye(3) * 50.0
            x = rng.normal(0, 2.5, (n, 3)) + 3.0
            a = ase.Atoms('H' * n, positions=x, cell=cell, pbc=False)
            i, j = ase.neighborlist.neighbor_list('ij', a, cutoff); S = np.zeros((len(i), 3), int)
        # keep atoms apart (no pathological r -> 0)
        pos.append(x); cells.append(cell); batch += [g] * n; pbc.append(kind == 'p')
        ei.append(np.stack([i, j]) + off); es.append(S); off += n
    return (np.concatenate(pos), np.stack(cells), np.array(batch), np.array(pbc), np.concatenate(ei, 1),
            np.concatenate(es))

pos, cell, batch, pbc, ei, es = system()
N, G = len(pos), len(cell)
q = rng.normal(0, 0.4, N)
# graph charges: 0, +1, -1 (charged periodic cell), 0
for g, Q in enumerate([0.0, 1.0, -1.0, 0.0]):
    m = batch == g; q[m] += (Q - q[m].sum()) / m.sum()
mu = rng.normal(0, 0.2, (N, 3))
th = rng.normal(0, 0.3, (N, 3, 3)); th = 0.5 * (th + th.transpose(0, 2, 1)); th -= np.trace(th, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
alpha = rng.uniform(0.3, 3.0, N)
save = dict(pos=pos, cell=cell, batch=batch, pbc=pbc, edge_index=ei, edge_shifts=es, q=q, mu=mu, theta=th, alpha=alpha, cutoff=cutoff)
T = lambda x, **k: torch.tensor(x, **k)
cases = {'s10': dict(sigma=1.0), 's05': dict(sigma=0.5), 's10_noquad': dict(sigma=1.0, quad=False),
         's10_nomutual': dict(sigma=1.0, mutual_induction=False), 's10_noind': dict(sigma=1.0, induction=False),
         's10_nocap_ktol': dict(sigma=1.0, cap_induced_width=False, ewald_k_tolerance=1e-5)}
for name, kw in cases.items():
    quad = kw.pop('quad', True)
    blk = TholeElectrostaticsBlock(o3.Irreps('8x0e+8x1o+8x2e'), mode='pbc_split', cutoff=cutoff, quadrupoles=quad,
                                   max_polarizability=4.0, min_polarizability=0.001, thole_width=2.0,
                                   cg_max_iterations=200, cg_tolerance=1e-12, diagonal_shift=1e-6,
                                   **{'cap_induced_width': True, **kw})
    p = T(pos, requires_grad=True); c = T(cell, requires_grad=True)
    o = blk._forward_pbc_split(p, T(batch), c, G, T(q), T(mu), T(alpha), T(ei), T(es), torch.zeros(G, 3), None,
                               torch.zeros(G, dtype=torch.bool), T(pbc), T(th) if quad else None)
    e = o['coulomb_energy']
    gp, gc = torch.autograd.grad(e.sum(), (p, c))
    for k in ('coulomb_energy', 'node_coulomb', 'permanent_energy', 'polarization_energy', 'induced_dipoles', 'total_dipole',
              'electrostatic_potential', 'scf_iterations'):
        save[f'{name}:{k}'] = o[k].detach().numpy()
    save[f'{name}:forces'] = -gp.numpy(); save[f'{name}:cellgrad'] = gc.numpy()
    save[f'{name}:eta'] = blk.ewald_width; save[f'{name}:k_max'] = blk.ewald_k_max
    print(name, 'eta', blk.ewald_width, 'kmax', round(blk.ewald_k_max, 4), 'E', e.detach().numpy().round(6), 'Epol', o['polarization_energy'].detach().numpy().round(6), 'it', o['scf_iterations'].tolist())
np.savez(out_path, **save)
