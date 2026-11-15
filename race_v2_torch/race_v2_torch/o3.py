"""O(3) building blocks in plain numpy / torch with the conventions of e3nn-jax 0.21 (the library the JAX RACEv2 was
trained with).  No e3nn dependency.

* ``Irreps``: list of (mul, l, p) with e3nn-jax ordering, ``regroup`` = stable sort + merge (e3nn_jax/_src/irreps.py).
* ``clebsch_gordan(l1, l2, l3)``: verbatim numpy port of e3nn_jax/_src/su2.py + so3.py (real basis, Frobenius norm 1).
* ``spherical_harmonics``: e3nn-jax "recursive, dense" algorithm (spherical_harmonics/recursive.py), normalize=True,
  normalization="component" (|Y^l|^2 = 2l+1, Y^1 = sqrt(3) (x, y, z)).
* ``normalize_constant(fn)``: the constant c of e3nn_jax.normalize_function, c = sqrt(mean(fn(x)^2)) over the
  deterministic grid x = normalspace(1_000_001) = sqrt(2) erfinv(linspace(-1, 1, n + 2)[1:-1]).
"""
from __future__ import annotations

import functools
import math
from fractions import Fraction
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch


# --------------------------------------------------------------------------------------------------- irreps
def _ir_key(l: int, p: int):
    """e3nn-jax Irrep.__lt__ order: (l, -p (-1)^l) -> natural parity first."""
    return (l, -p * (-1) ** l)


class Irreps:
    """Minimal e3nn-jax compatible irreps: a tuple of (mul, l, p), p = +1 (e) or -1 (o)."""

    def __init__(self, x: Union[str, "Irreps", Sequence[Tuple[int, int, int]], None] = None):
        if x is None:
            items = []
        elif isinstance(x, Irreps):
            items = list(x.items)
        elif isinstance(x, str):
            items = []
            s = x.strip()
            if s.startswith('Irreps:'):
                s = s[len('Irreps:'):]
            for tok in s.split('+'):
                tok = tok.strip()
                if not tok:
                    continue
                if 'x' in tok:
                    mul, ir = tok.split('x')
                    mul = int(mul)
                else:
                    mul, ir = 1, tok
                l, p = int(ir[:-1]), {'e': 1, 'o': -1}[ir[-1]]
                items.append((mul, l, p))
        else:
            items = [(int(m), int(l), int(p)) for m, l, p in x]
        self.items: Tuple[Tuple[int, int, int], ...] = tuple(items)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __eq__(self, other):
        return isinstance(other, Irreps) and self.items == other.items

    def __hash__(self):
        return hash(self.items)

    @staticmethod
    def ir_str(l: int, p: int) -> str:
        return f"{l}{'e' if p == 1 else 'o'}"

    @staticmethod
    def mulir_str(mul: int, l: int, p: int) -> str:
        return f"{mul}x{Irreps.ir_str(l, p)}"

    def __str__(self):
        return '+'.join(self.mulir_str(*t) for t in self.items)

    __repr__ = __str__

    @property
    def dim(self) -> int:
        return sum(m * (2 * l + 1) for m, l, _ in self.items)

    @property
    def num_irreps(self) -> int:
        return sum(m for m, _, _ in self.items)

    @property
    def lmax(self) -> int:
        return max(l for _, l, _ in self.items)

    def simplify(self) -> "Irreps":
        out: List[List[int]] = []
        for m, l, p in self.items:
            if out and out[-1][1] == l and out[-1][2] == p:
                out[-1][0] += m
            elif m > 0:
                out.append([m, l, p])
        return Irreps([tuple(t) for t in out if t[0] > 0])

    def sort(self) -> "Irreps":
        order = sorted(range(len(self.items)), key=lambda i: (_ir_key(self.items[i][1], self.items[i][2]), i))
        return Irreps([self.items[i] for i in order])

    def regroup(self) -> "Irreps":
        return self.sort().simplify()

    def irs(self):
        return {(l, p) for _, l, p in self.items}

    def filter_keep(self, irs: Iterable[Tuple[int, int]]) -> "Irreps":
        irs = set(irs)
        return Irreps([t for t in self.items if (t[1], t[2]) in irs])

    def slices(self) -> List[Tuple[int, int]]:
        out, i = [], 0
        for m, l, _ in self.items:
            n = m * (2 * l + 1)
            out.append((i, i + n))
            i += n
        return out

    def scale_mul(self, k: int) -> "Irreps":
        return Irreps([(m * k, l, p) for m, l, p in self.items])

    def div_mul(self, k: int) -> "Irreps":
        """e3nn-jax ``irreps // k``."""
        return Irreps([(m // k, l, p) for m, l, p in self.items])


def s2_irreps(lmax: int, p_val: int = 1, p_arg: int = -1) -> Irreps:
    """e3nn_jax.s2_irreps(lmax): 1x0e + 1x1o + 1x2e + ..."""
    return Irreps([(1, l, p_val * p_arg ** l) for l in range(lmax + 1)])


# --------------------------------------------------------------------------------------------------- Clebsch-Gordan
def _su2_cg(idx1, idx2, idx3):
    """Clebsch-Gordan coefficient (e3nn_jax/_src/su2.py ``_su2_cg``, copied from qutip)."""
    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3
    if m3 != m1 + m2:
        return 0
    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    def f(n):
        assert n == round(n)
        return math.factorial(round(n))

    C = (
        (2.0 * j3 + 1.0)
        * Fraction(
            f(j3 + j1 - j2) * f(j3 - j1 + j2) * f(j1 + j2 - j3) * f(j3 + m3) * f(j3 - m3),
            f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2),
        )
    ) ** 0.5
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v),
            f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3),
        )
    return C * S


def su2_clebsch_gordan(j1, j2, j3) -> np.ndarray:
    mat = np.zeros((int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)))
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)] = _su2_cg((j1, m1), (j2, m2), (j3, m1 + m2))
    return mat / math.sqrt(2 * j3 + 1)


def change_basis_real_to_complex(l: int) -> np.ndarray:
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / np.sqrt(2)
        q[l + m, l - abs(m)] = -1j / np.sqrt(2)
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / np.sqrt(2)
        q[l + m, l - abs(m)] = 1j * (-1) ** m / np.sqrt(2)
    return (-1j) ** l * q


@functools.lru_cache(maxsize=None)
def _clebsch_gordan(l1: int, l2: int, l3: int) -> np.ndarray:
    C = su2_clebsch_gordan(l1, l2, l3)
    Q1 = change_basis_real_to_complex(l1)
    Q2 = change_basis_real_to_complex(l2)
    Q3 = change_basis_real_to_complex(l3)
    C = np.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, np.conj(Q3.T), C)
    assert np.all(np.abs(np.imag(C)) < 1e-5)
    return np.real(C)


def clebsch_gordan(l1: int, l2: int, l3: int) -> np.ndarray:
    """Real Clebsch-Gordan tensor (2l1+1, 2l2+1, 2l3+1) of e3nn-jax (``e3nn.clebsch_gordan``), Frobenius norm 1."""
    return _clebsch_gordan(int(l1), int(l2), int(l3)).copy()


# --------------------------------------------------------------------------------------------------- spherical harmonics
def _biggest_power_of_two(n: int) -> int:
    return 2 ** (n.bit_length() - 1)


@functools.lru_cache(maxsize=None)
def _sh_recursion(lmax: int):
    """[(l, l1, l2, C)] with C = float(x / norm) * CG(l1, l2, l) of e3nn-jax recursive.py, normalization 'component'."""
    steps = []
    for l in range(2, lmax + 1):
        l2 = _biggest_power_of_two(l - 1)
        l1 = l - l2
        cg = clebsch_gordan(l1, l2, l)
        cst = cg[l1, l2, :]                     # Y^l1 = e_{m=0}, Y^l2 = e_{m=0} substituted into the CG contraction
        norm = math.sqrt(float(np.sum(cst ** 2)))
        x = math.sqrt((2 * l + 1) / ((2 * l1 + 1) * (2 * l2 + 1)))
        steps.append((l, l1, l2, (x / norm) * cg))
    return tuple(steps)


class SphericalHarmonics(torch.nn.Module):
    """Y^0..Y^lmax of the (normalised) edge vectors, concatenated (n, (lmax+1)^2), e3nn-jax 'component' normalisation."""

    def __init__(self, lmax: int):
        super().__init__()
        self.lmax = int(lmax)
        self._steps = []
        for l, l1, l2, C in _sh_recursion(self.lmax):
            name = f"cg_sh_{l}"
            self.register_buffer(name, torch.as_tensor(C, dtype=torch.float64), persistent=False)
            self._steps.append((l, l1, l2, name))

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        r2 = torch.sum(vec ** 2, dim=-1, keepdim=True)
        r2 = torch.where(r2 == 0.0, torch.ones_like(r2), r2)
        x = vec / torch.sqrt(r2)
        ys = {0: torch.ones_like(x[..., :1])}
        if self.lmax >= 1:
            ys[1] = math.sqrt(3.0) * x
        for l, l1, l2, name in self._steps:
            C = getattr(self, name).to(x.dtype)
            ys[l] = torch.einsum("...i,...j,ijk->...k", ys[l1], ys[l2], C)
        return torch.cat([ys[l] for l in range(self.lmax + 1)], dim=-1)


def spherical_harmonics(lmax: int, vec: torch.Tensor) -> torch.Tensor:
    return SphericalHarmonics(lmax).to(vec.device)(vec)


# --------------------------------------------------------------------------------------------------- activations
def normalspace(n: int, dtype=torch.float64) -> torch.Tensor:
    """e3nn_jax.normalspace: x_i with Phi(x_i) = i / (n + 1)."""
    return math.sqrt(2) * torch.special.erfinv(torch.linspace(-1.0, 1.0, n + 2, dtype=dtype)[1:-1])


def silu(x: torch.Tensor) -> torch.Tensor:
    """jax.nn.silu = x * sigmoid(x)."""
    return x * torch.sigmoid(x)


@functools.lru_cache(maxsize=None)
def normalize_constant(name: str) -> float:
    """Normalisation constant c of e3nn_jax.normalize_function for 'silu' | 'tanh' | 'sigmoid' (float64 grid)."""
    fn = {'silu': silu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}[name]
    x = normalspace(1_000_001)
    c = torch.mean(fn(x) ** 2) ** 0.5
    return float(c.item())


class NormalizedSiLU(torch.nn.Module):
    """silu(x) / c (e3nn.scalar_activation / e3nn.gate with normalize_act=True on scalar inputs)."""

    def __init__(self):
        super().__init__()
        self.c = normalize_constant('silu')

    def forward(self, x):
        return silu(x) / self.c
