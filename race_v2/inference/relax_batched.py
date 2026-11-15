"""Batched structure relaxation for RACEv2: FIRE + FrechetCellFilter for many structures at once, one jitted model
call per optimizer step (2026-11-11).

The host side (SlotCalculator, _MemoFrechetCellFilter, StructureRelaxation, HostWorker, the host pools) and the
BatchedRelaxDriver are copied unchanged from BAM-GAMP-jax ``bam_jax/inference/batched.py`` (WP-S6 of that project,
same group): every structure keeps its own unmodified ``ase.optimize.FIRE`` on its own ``FrechetCellFilter``;
``StructureRelaxation.advance`` replays ``Dynamics.irun`` of ASE 3.29 with the force evaluations done by the driver
for all structures together.  New here: ``race_payload`` (host, JAX-free) and ``RaceEvaluator`` (device).

Import rule: this file is a TOP-LEVEL module (``relax_batched``, its directory on sys.path), not
``bam_omat24.relax_batched``: the host workers are spawned processes that unpickle its functions by module name,
and importing the package would load JAX in every worker (see stream_worker.py).  JAX is imported only inside
RaceEvaluator methods, which run in the main process.
"""
from __future__ import annotations

import dataclasses
import functools
import math
import os
import sys
import time
import traceback
from collections import deque
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase.filters import FrechetCellFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neighbors import neighbor_pairs  # noqa: E402

# StructureRelaxation.advance replays ase.optimize.optimize.Dynamics.irun of this ASE release.
_IRUN_ASE_VERSION = "3.29"


def _check_ase_version() -> None:
    import ase
    if not str(ase.__version__).startswith(_IRUN_ASE_VERSION):
        import warnings
        warnings.warn(
            f"batched relaxation replays Dynamics.irun of ASE {_IRUN_ASE_VERSION}.x but ASE "
            f"{ase.__version__} is installed: check StructureRelaxation.advance against that release", stacklevel=3)


def ase_payload(atoms):
    """Payload for :class:`AseEvaluator`: what an ASE calculator needs."""
    return (atoms.get_atomic_numbers().copy(), atoms.get_positions().copy(),
            atoms.cell.array.copy(), atoms.pbc.copy())


def estimate_edges(atoms, cutoff: float) -> int:
    """Expected directed-edge count for admission control: ``n * rho * 4/3 pi rc^3``."""
    n = len(atoms)
    vol = abs(float(np.linalg.det(atoms.cell.array)))
    if vol <= 0.0:
        return n * n
    return int(math.ceil(n * n / vol * (4.0 / 3.0) * math.pi * cutoff ** 3))


# ---------------------------------------------------------------------------
# Host side: one structure's optimizer
# ---------------------------------------------------------------------------
class SlotCalculator(Calculator):
    """Holds the batched model's results for exactly one geometry.

    Indistinguishable from ``BamJaxCalculator`` for ASE: same
    ``implemented_properties`` (so ``free_energy`` raises the same
    PropertyNotImplementedError FIRE's ``get_value`` probes for), results set
    with ``self.atoms = atoms.copy()`` exactly like ``Calculator.calculate``.
    A request for a geometry that was not evaluated is a driver bug and raises.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, record: bool = False):
        super().__init__()
        self.version = 0
        self.trace = [] if record else None

    def store(self, atoms, energy, forces, stress) -> None:
        self.atoms = atoms.copy()
        self.results = {"energy": energy, "forces": forces, "stress": stress}
        self.version += 1
        if self.trace is not None:
            self.trace.append((atoms.get_positions().copy(), atoms.cell.array.copy(),
                               float(energy)))

    def check_state(self, atoms, tol=1e-15):
        a = self.atoms
        # Fast path: everything ``compare_atoms`` looks at is exactly equal ->
        # no change (compare_atoms would say the same). Otherwise ASE decides.
        if (a is not None and len(a) == len(atoms)
                and np.array_equal(a.cell.array, atoms.cell.array)
                and np.array_equal(a.pbc, atoms.pbc)
                and a.arrays.keys() == atoms.arrays.keys()
                and all(np.array_equal(v, atoms.arrays[k]) for k, v in a.arrays.items())):
            return []
        return super().check_state(atoms, tol=tol)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        raise RuntimeError(
            "SlotCalculator: geometry was not evaluated by the batched model "
            f"(changes: {system_changes})"
        )


class _MemoFrechetCellFilter(FrechetCellFilter):
    """FrechetCellFilter whose ``get_forces`` is memoized on its exact inputs.

    ``Dynamics.irun`` calls it 3x per step at the same geometry. The key is
    (calculator result version, positions, cell); the cached value includes
    the side effect on ``self.stress``. Deterministic => bit-identical."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memo = None

    def get_forces(self, **kwargs):
        if kwargs:
            return super().get_forces(**kwargs)
        key = (getattr(self.atoms.calc, "version", None),
               self.atoms.positions.tobytes(), self.atoms.cell.array.tobytes())
        if self._memo is not None and self._memo[0] == key:
            self.stress = self._memo[2].copy()
            return self._memo[1].copy()
        forces = super().get_forces()
        self._memo = (key, forces.copy(), np.array(self.stress, copy=True))
        return forces


@dataclasses.dataclass
class RelaxSettings:
    fmax: float = 0.05
    max_steps: int = 500
    fire_logfile: str | None = None
    record_trajectory: bool = False


class StructureRelaxation:
    """``relax_wbm.relax_one`` split at its force evaluations.

    Life cycle: construct (the pending geometry is the initial one) ->
    ``advance(E, F, S)`` for the pending geometry -> either finished
    (``result()``), or a new pending geometry was set by ``FIRE.step()``.
    """

    def __init__(self, key: str, atoms, settings: RelaxSettings):
        self.key = key
        self.atoms = atoms
        self.settings = settings
        self.calc = SlotCalculator(record=settings.record_trajectory)
        atoms.calc = self.calc
        self.opt = None
        self.filt = None
        self.e_init = None
        self.t0 = None
        self.wall = None
        self.converged = False
        self.done = False

    def advance(self, energy, forces, stress) -> bool:
        """Consume the model output for the pending geometry; run until the
        next force evaluation is needed. Returns True when finished."""
        from ase.optimize import FIRE

        self.calc.store(self.atoms, energy, forces, stress)
        opt = self.opt
        if opt is None:
            # relax_one: single point on the initial geometry, then the filter
            # and FIRE (defaults), then run(fmax, steps=max_steps).
            self.e_init = float(self.atoms.get_potential_energy())
            self.filt = _MemoFrechetCellFilter(self.atoms)
            opt = self.opt = FIRE(self.filt, logfile=self.settings.fire_logfile)
            self.t0 = time.time()
            # Optimizer.run -> Dynamics.run -> Dynamics.irun (ase 3.29):
            opt.fmax = self.settings.fmax
            opt.max_steps = opt.nsteps + int(self.settings.max_steps)
            gradient = opt.optimizable.get_gradient()
            if opt.nsteps == 0:
                opt.log(gradient)
                if opt.trajectory is None:
                    opt.call_observers()
                elif opt._traj_is_empty():
                    opt.call_observers()
            gradient = opt.optimizable.get_gradient()
            is_converged = opt.gradient_converged(gradient)
        else:
            # ... loop body after ``self.step()`` (whose force call we served)
            opt.nsteps += 1
            gradient = opt.optimizable.get_gradient()
            opt.log(gradient)
            opt.call_observers()
            gradient = opt.optimizable.get_gradient()
            is_converged = opt.gradient_converged(gradient)
        self.converged = bool(is_converged)
        if is_converged or not (opt.nsteps < opt.max_steps):
            self.wall = time.time() - self.t0
            self.done = True
            return True
        opt.step()                           # sets the next pending geometry
        return False

    def result(self) -> dict:
        """The dict ``relax_wbm.relax_one`` returns (same keys and order)."""
        atoms = self.atoms
        e_final = float(atoms.get_potential_energy())
        f_final = np.asarray(atoms.get_forces(), dtype=np.float64)
        s_final = np.asarray(atoms.get_stress(voigt=True), dtype=np.float64)
        return {
            "n_atoms": int(len(atoms)),
            "n_steps": int(self.opt.nsteps),
            "converged": self.converged,
            "initial_energy": self.e_init,
            "final_energy": e_final,
            "numbers": atoms.get_atomic_numbers().tolist(),
            "final_positions": atoms.get_positions().tolist(),
            "final_cell": atoms.get_cell().array.tolist(),
            "final_forces": f_final.tolist(),
            "final_stress": s_final.tolist(),
            "wall_seconds": float(self.wall),
        }


class HostWorker:
    """Owns a set of :class:`StructureRelaxation` and does all host-side work
    for them: admitting, consuming model results, stepping FIRE, building the
    next payload. ``process`` takes ``[(slot_id, op, data)]`` with op "admit"
    (data = (key, atoms)), "result" (data = (E, F, S)) or "drop" (data = the
    reason; the model could not evaluate it) and returns
    ``{slot_id: (status, value)}`` with status "run" (value = payload),
    "done" (value = (result dict, trace or None)) or "error" (value = text)."""

    def __init__(self, prepare: Callable, settings: RelaxSettings):
        self.prepare = prepare
        self.settings = settings
        self.slots: dict[int, StructureRelaxation] = {}

    def process(self, msgs) -> dict:
        out = {}
        for slot_id, op, data in msgs:
            try:
                if op == "admit":
                    key, atoms = data
                    rel = StructureRelaxation(key, atoms, self.settings)
                    self.slots[slot_id] = rel
                    out[slot_id] = ("run", self.prepare(rel.atoms))
                elif op == "drop":                      # evaluation failed for it
                    self.slots.pop(slot_id, None)
                    out[slot_id] = ("error", str(data))
                    continue
                elif op == "result":
                    rel = self.slots[slot_id]
                    if rel.advance(*data):
                        del self.slots[slot_id]
                        out[slot_id] = ("done", (rel.result(), rel.calc.trace))
                    else:
                        out[slot_id] = ("run", self.prepare(rel.atoms))
                else:
                    raise ValueError(f"unknown op {op!r}")
            except Exception as exc:                       # per-structure failure
                self.slots.pop(slot_id, None)
                out[slot_id] = ("error", f"{type(exc).__name__}: {exc}\n"
                                         f"{traceback.format_exc(limit=3)}")
        return out


def _worker_main(conn, prepare, settings):          # pragma: no cover (subprocess)
    # Host workers never touch JAX; keep BLAS single-threaded per process.
    worker = HostWorker(prepare, settings)
    while True:
        msg = conn.recv()
        if msg is None:
            break
        conn.send(worker.process(msg))
    conn.close()


class InProcessHost:
    """Single-process host: ``HostWorker`` called directly."""

    def __init__(self, prepare: Callable, settings: RelaxSettings):
        self.worker = HostWorker(prepare, settings)
        self.n_workers = 1

    def assign(self, slot_id: int) -> int:
        return 0

    def release(self, slot_id: int) -> None:
        pass

    def process(self, msgs) -> dict:
        return self.worker.process(msgs)

    def close(self) -> None:
        pass


class ProcessHostPool:
    """``n`` worker processes (spawn context: the parent holds a CUDA context),
    each owning the structures assigned to it. ``process`` sends every worker
    its messages first and then collects, so workers run in parallel."""

    def __init__(self, prepare: Callable, settings: RelaxSettings, n: int):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.n_workers = int(n)
        self._conns = []
        self._procs = []
        env_backup = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        try:
            for _ in range(self.n_workers):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker_main, args=(child, prepare, settings), daemon=True)
                p.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(p)
        finally:
            for k, v in env_backup.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self._owner: dict[int, int] = {}
        self._load = [0] * self.n_workers

    def assign(self, slot_id: int) -> int:
        w = int(np.argmin(self._load))
        self._owner[slot_id] = w
        self._load[w] += 1
        return w

    def release(self, slot_id: int) -> None:
        w = self._owner.pop(slot_id, None)
        if w is not None:
            self._load[w] -= 1

    def process(self, msgs) -> dict:
        per = [[] for _ in range(self.n_workers)]
        for m in msgs:
            per[self._owner[m[0]]].append(m)
        busy = [w for w in range(self.n_workers) if per[w]]
        for w in busy:
            self._conns[w].send(per[w])
        out = {}
        for w in busy:
            out.update(self._conns[w].recv())
        return out

    def close(self) -> None:
        for c in self._conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()



class AseEvaluator:
    """Evaluator over any ASE calculator, one structure at a time (reference /
    tests). ``calc_factory()`` is called once per evaluation."""

    def __init__(self, calc_factory: Callable, cutoff: float | None = None):
        self.calc_factory = calc_factory
        self.cutoff = cutoff
        self.shapes: set = set()
        self.last_shape = None

    def prepare_fn(self) -> Callable:
        return ase_payload

    def cost(self, atoms):
        return len(atoms), (0 if self.cutoff is None else estimate_edges(atoms, self.cutoff))

    def payload_cost(self, payload):
        return len(payload[0]), 0

    def dispatch(self, payloads):
        from ase import Atoms
        out = []
        for numbers, positions, cell, pbc in payloads:
            atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
            atoms.calc = self.calc_factory()
            out.append((float(atoms.get_potential_energy()), atoms.get_forces(),
                        atoms.get_stress(voigt=True)))
        return out

    def collect(self, handle):
        if isinstance(handle, _FailedDispatch):
            raise handle.exc
        return handle

    def evaluate(self, payloads):
        return self.collect(self.dispatch(payloads))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class _FailedDispatch:
    """Placeholder handle for a dispatch that raised; collect re-raises."""

    def __init__(self, exc: Exception):
        self.exc = exc


@dataclasses.dataclass
class RelaxOutcome:
    key: str
    result: dict | None = None          # relax_one-style dict when finished
    error: str | None = None            # text when the structure failed
    trace: list | None = None           # [(positions, cell, energy)] if recorded


@dataclasses.dataclass
class _Slot:
    key: str
    payload: Any = None
    cost: tuple = (0, 0)


class BatchedRelaxDriver:
    """Relax a stream of ``(key, atoms)`` with up to ``max_structures``
    structures per group in flight.

    Each group is one model call per step. With ``groups=2`` the host work of
    one group (FIRE steps, neighbour lists) overlaps the device work of the
    other (JAX dispatch is asynchronous). A finished structure leaves its slot
    immediately and the slot is refilled from the queue (FIFO within a
    ``lookahead`` window; a structure that does not fit the group's
    ``atom_budget`` / ``edge_budget`` waits, unless the group is empty).
    ``host_workers > 0`` moves all host work into that many processes."""

    def __init__(self, evaluator, *, fmax: float = 0.05, max_steps: int = 500,
                 fire_logfile: str | None = None, max_structures: int = 32,
                 groups: int = 2, atom_budget: int | None = None,
                 edge_budget: int | None = None, host_workers: int = 0,
                 lookahead: int = 256, record_trajectory: bool = False,
                 log: Callable[[str], None] | None = None, log_every: float = 60.0):
        if max_structures < 1 or groups < 1:
            raise ValueError("max_structures and groups must be >= 1")
        self.ev = evaluator
        self.settings = RelaxSettings(fmax=float(fmax), max_steps=int(max_steps),
                                      fire_logfile=fire_logfile,
                                      record_trajectory=bool(record_trajectory))
        self.max_structures = int(max_structures)
        self.n_groups = int(groups)
        self.atom_budget = atom_budget
        self.edge_budget = edge_budget
        self.host_workers = int(host_workers)
        self.lookahead = max(int(lookahead), 1)
        _check_ase_version()
        self.log = log
        self.log_every = float(log_every)
        self.stats = {"calls": 0, "structure_steps": 0, "t_host": 0.0, "t_wait": 0.0,
                      "t_dispatch": 0.0, "finished": 0, "failed": 0}

    # ------------------------------------------------------------------
    def _fits(self, members: dict, cost) -> bool:
        if not members:
            return True
        if len(members) >= self.max_structures:
            return False
        n_sum = sum(s.cost[0] for s in members.values())
        e_sum = sum(s.cost[1] for s in members.values())
        if self.atom_budget is not None and n_sum + cost[0] > self.atom_budget:
            return False
        if self.edge_budget is not None and e_sum + cost[1] > self.edge_budget:
            return False
        return True

    def run(self, items: Iterable) -> Iterator[RelaxOutcome]:
        host = (ProcessHostPool(self.ev.prepare_fn(), self.settings, self.host_workers)
                if self.host_workers > 0 else InProcessHost(self.ev.prepare_fn(), self.settings))
        try:
            yield from self._run(iter(items), host)
        finally:
            host.close()

    def _run(self, it, host) -> Iterator[RelaxOutcome]:
        st = self.stats
        G = self.n_groups
        members: list[dict[int, _Slot]] = [dict() for _ in range(G)]
        handles: list = [None] * G
        order: list[list[int]] = [[] for _ in range(G)]
        buffer: deque = deque()
        exhausted = False
        next_id = 0
        t_last_log = time.time()
        t_start = time.time()

        def fill_buffer():
            nonlocal exhausted
            while not exhausted and len(buffer) < self.lookahead:
                try:
                    key, atoms = next(it)
                except StopIteration:
                    exhausted = True
                    break
                buffer.append((key, atoms, self.ev.cost(atoms)))

        def handle_outputs(g, outputs) -> list[RelaxOutcome]:
            done = []
            for sid, (status, value) in outputs.items():
                slot = members[g][sid]
                if status == "run":
                    slot.payload = value
                    slot.cost = self.ev.payload_cost(value)
                    continue
                del members[g][sid]
                host.release(sid)
                if status == "done":
                    res, trace = value
                    st["finished"] += 1
                    done.append(RelaxOutcome(slot.key, result=res, trace=trace))
                else:
                    st["failed"] += 1
                    done.append(RelaxOutcome(slot.key, error=value))
            return done

        def refill(g) -> tuple[list[RelaxOutcome], int]:
            nonlocal next_id
            fill_buffer()
            admits = []
            while buffer and len(members[g]) < self.max_structures:
                pick = None
                for i, (_key, _atoms, cost) in enumerate(buffer):
                    if self._fits(members[g], cost):
                        pick = i
                        break
                if pick is None:
                    break
                key, atoms, cost = buffer[pick]
                del buffer[pick]
                sid = next_id
                next_id += 1
                host.assign(sid)
                members[g][sid] = _Slot(key=key, cost=cost)
                admits.append((sid, "admit", (key, atoms)))
                fill_buffer()
            if not admits:
                return [], 0
            t0 = time.time()
            outputs = host.process(admits)
            st["t_host"] += time.time() - t0
            return handle_outputs(g, outputs), len(admits)

        def dispatch(g):
            if not members[g]:
                handles[g] = None
                return
            t0 = time.time()
            order[g] = list(members[g])
            payloads = [members[g][s].payload for s in order[g]]
            try:
                handles[g] = self.ev.dispatch(payloads)
            except Exception as exc:          # e.g. compile/OOM: resolved at collect
                handles[g] = _FailedDispatch(exc)
            st["t_dispatch"] += time.time() - t0
            st["calls"] += 1
            st["structure_steps"] += len(order[g])

        # Group g's host work (collect -> FIRE -> refill -> neighbour lists)
        # runs while the other groups' dispatched calls execute on the device.
        while True:
            progressed = False
            for g in range(G):
                if handles[g] is not None:
                    t0 = time.time()
                    try:
                        results = self.ev.collect(handles[g])
                    except Exception as exc:
                        results = self._evaluate_one_by_one(g, members, order, exc)
                    st["t_wait"] += time.time() - t0
                    handles[g] = None
                    msgs = [(sid, "result", r) if not isinstance(r, str) else (sid, "drop", r)
                            for sid, r in zip(order[g], results)]
                    t0 = time.time()
                    outputs = host.process(msgs)
                    st["t_host"] += time.time() - t0
                    yield from handle_outputs(g, outputs)
                    progressed = True
                outs, n_admitted = refill(g)
                yield from outs
                progressed = progressed or n_admitted > 0
                dispatch(g)
            if self.log is not None and time.time() - t_last_log >= self.log_every:
                t_last_log = time.time()
                self.log(self.progress_line(time.time() - t_start,
                                            sum(len(m) for m in members)))
            if all(h is None for h in handles):
                fill_buffer()
                if not buffer and exhausted:
                    break
                if not progressed:      # an empty group always admits: a bug
                    raise RuntimeError("batched driver stalled with a non-empty queue")
        if self.log is not None:
            self.log(self.progress_line(time.time() - t_start, 0, final=True))

    def _evaluate_one_by_one(self, g, members, order, exc) -> list:
        """The group's model call failed (e.g. device OOM on one huge
        structure): evaluate its structures one at a time. A structure that
        fails alone gets its error text instead of results (-> "drop"), so one
        bad structure fails alone, as in relax_wbm.py."""
        if self.log is not None:
            self.log(f"[batched] group call with {len(order[g])} structures failed "
                     f"({type(exc).__name__}: {str(exc)[:200]}); evaluating one by one")
        self.stats["fallbacks"] = self.stats.get("fallbacks", 0) + 1
        out = []
        for sid in order[g]:
            try:
                out.append(self.ev.evaluate([members[g][sid].payload])[0])
            except Exception as e1:
                out.append(f"model evaluation failed: {type(e1).__name__}: {e1}")
        return out

    def progress_line(self, elapsed: float, in_flight: int, final: bool = False) -> str:
        st = self.stats
        calls = max(st["calls"], 1)
        rate = st["finished"] / max(elapsed, 1e-9)
        shape = getattr(self.ev, "last_shape", None)
        n_shapes = len(getattr(self.ev, "shapes", ()))
        tag = "[batched] FINAL" if final else "[batched]"
        # bench_summary.py (WP-S6 benchmark) parses t / in_flight / structure_steps
        # to get the steady-state rate while the queue is not drained.
        return (f"{tag} t={elapsed:.1f}s finished={st['finished']} failed={st['failed']} "
                f"in_flight={in_flight} calls={st['calls']} "
                f"structure_steps={st['structure_steps']} "
                f"structs/call={st['structure_steps'] / calls:.1f} "
                f"rate={rate * 3600:.0f}/h ms/structure-step="
                f"{1e3 * elapsed / max(st['structure_steps'], 1):.2f} "
                f"host={st['t_host']:.0f}s wait={st['t_wait']:.0f}s "
                f"dispatch={st['t_dispatch']:.0f}s shapes={n_shapes} last_shape={shape}")


# ---------------------------------------------------------------------------
# RACEv2: host payload and device evaluator
# ---------------------------------------------------------------------------
def race_payload(atoms, *, cutoff: float, z_to_index: dict) -> dict:
    """Atoms -> the per-graph arrays of stream_io.graph_arrays (same edge convention: receivers = i, senders = j,
    Sij = -S), without targets and in float64 (positions are not rounded to float32 for relaxations)."""
    src, dst, S = neighbor_pairs(atoms, cutoff)
    n = len(atoms)
    cell = np.asarray(atoms.cell.array, np.float64)
    return {'n_node': np.array([n], np.int32), 'n_edge': np.array([len(src)], np.int32),
            'species': np.array([z_to_index[int(z)] for z in atoms.numbers], np.int32),
            'positions': np.asarray(atoms.positions, np.float64), 'forces': np.zeros((n, 3), np.float64),
            'energy': np.zeros(1, np.float64),
            'senders': np.asarray(dst, np.int32), 'receivers': np.asarray(src, np.int32),
            'Sij': (-np.asarray(S)).astype(np.float64), 'cell': cell[None],
            'cutoff': np.array([cutoff], np.float64), 'volume': np.array([abs(np.linalg.det(cell))], np.float64),
            'stress': np.zeros((1, 6), np.float64),
            'numbers': np.asarray(atoms.numbers, np.int64)}


def _rung(n: int, base: int) -> int:
    """Smallest value >= n on the ladder base * 2^k * {1, 1.5} (few padded shapes, at most 33 % padding)."""
    r = base
    while True:
        if n <= r:
            return r
        if n <= r + r // 2:
            return r + r // 2
        r *= 2


class RaceEvaluator:
    """payloads (race_payload) -> one padded batch -> one jitted RACEv2 call -> [(E_total, forces, stress6)].

    E_total = model energy + sum of the config's atom_energies (E0), i.e. the scale of the training label
    (MPtrj: uncorrected DFT total energy).  Stress in eV/A^3, ASE sign and Voigt order (the model's output)."""

    def __init__(self, ckpt_path, *, max_graphs: int = 64, node_rung: int = 256, edge_rung: int = 8192,
                 weights: str = 'ema'):
        import jax
        from bam_omat24.predict import Predictor
        p = Predictor(ckpt_path, species_refit=False)
        if weights != 'ema':
            raise ValueError("only the EMA weights of the checkpoint are supported")
        self.cfg = p.cfg
        self.cutoff = float(p.cfg.get('cutoff', 5.0))
        self.atomic_numbers = list(p.atomic_numbers)
        self.e0 = np.zeros(119)
        self.e0[np.asarray(self.atomic_numbers)] = p.atom_energies
        self.dtype = np.float64 if p.cfg.get('x64') else np.float32
        self.max_graphs, self.node_rung, self.edge_rung = int(max_graphs), int(node_rung), int(edge_rung)
        self._step = jax.jit(lambda b: p.model(b))
        self.shapes: set = set()
        self.last_shape = None

    def prepare_fn(self) -> Callable:
        return functools.partial(race_payload, cutoff=self.cutoff,
                                 z_to_index={int(z): i for i, z in enumerate(self.atomic_numbers)})

    def cost(self, atoms):
        return len(atoms), estimate_edges(atoms, self.cutoff)

    def payload_cost(self, payload):
        return int(payload['n_node'][0]), int(payload['n_edge'][0])

    def dispatch(self, payloads):
        from bam_omat24.data_nnx import fixed_collate_fn
        from bam_omat24.stream_data import _to_graph, _cast
        if len(payloads) > self.max_graphs:
            raise ValueError(f"{len(payloads)} structures in one call, max_graphs is {self.max_graphs}")
        n = sum(int(p['n_node'][0]) for p in payloads)
        e = sum(int(p['n_edge'][0]) for p in payloads)
        shape = (_rung(n + 1, self.node_rung), _rung(max(e, 1), self.edge_rung))
        self.shapes.add(shape); self.last_shape = shape
        graphs = [_to_graph(_cast({k: v for k, v in p.items() if k != 'numbers'}, self.dtype)) for p in payloads]
        batch = fixed_collate_fn(graphs, self.max_graphs, shape[0], shape[1])
        return self._step(batch), [p['numbers'] for p in payloads]

    def collect(self, handle):
        if isinstance(handle, _FailedDispatch):
            raise handle.exc
        out, numbers = handle
        E = np.asarray(out['energy'], np.float64).reshape(-1)
        F = np.asarray(out['forces'], np.float64)
        S = np.asarray(out['stress'], np.float64).reshape(-1, 6)
        res, off = [], 0
        for g, z in enumerate(numbers):
            f = F[off:off + len(z)].copy(); off += len(z)
            if not (np.isfinite(E[g]) and np.all(np.isfinite(f)) and np.all(np.isfinite(S[g]))):
                raise FloatingPointError(f"non-finite model output for structure {g} of the call")
            res.append((float(E[g] + self.e0[z].sum()), f, S[g].copy()))
        return res

    def evaluate(self, payloads):
        return self.collect(self.dispatch(payloads))
