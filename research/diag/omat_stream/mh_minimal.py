"""Smallest multi-process JAX program (2026-11-11): one process per node, all GPUs in one mesh, a jitted reduction
over the 'dp' axis (the collective the trainer's gradient sum needs).  No model, no data loader: if this hangs with
4 processes the problem is in the JAX / NCCL setup, not in the trainer.  Stages are printed with times.

    srun --ntasks-per-node=1 --cpus-per-task=288 python -X faulthandler runs/diag/omat_stream/mh_minimal.py
(BAM_DISTRIBUTED=1, MASTER_ADDR, MASTER_PORT as for the trainer.)
"""
import faulthandler
import os
import sys
import time

faulthandler.dump_traceback_later(float(os.environ.get('BAM_STACK_DUMP_SECS', 120)), repeat=True)
here = os.path.dirname(os.path.abspath(__file__))
for base in (os.getcwd(), os.path.join(here, '..', '..', '..')):
    if os.path.exists(os.path.join(base, 'bam_omat24', 'distributed.py')):
        sys.path.insert(0, os.path.join(base, 'bam_omat24'))
        break
t0 = time.time()


def say(msg):
    print(f'[p{os.environ.get("SLURM_PROCID", "0")} {time.time() - t0:6.1f} s] {msg}', flush=True)


import jax  # noqa: E402
import distributed as D  # noqa: E402  (bam_omat24/distributed.py as a top-level module: no model import)

D.init_distributed()
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

say(f'process {jax.process_index()}/{jax.process_count()}, local {jax.local_device_count()}, global {jax.device_count()}')
mesh = Mesh(np.array(jax.devices()), ('dp',))
sh = NamedSharding(mesh, P('dp'))
n_local = jax.local_device_count()
x = jax.make_array_from_process_local_data(sh, np.full((n_local, 4096), jax.process_index() + 1.0, np.float32))
say(f'global array {x.shape}')


@jax.jit
def total(x):
    return jnp.sum(x)


@jax.jit
def step(w, x):                                   # a small "training step": per-device work, then a global mean
    g = jnp.mean(jnp.tanh(x @ w), axis=0)
    return w - 0.01 * jnp.outer(jnp.ones(w.shape[0]), g)[:, :w.shape[1]]


s = total(x)
s.block_until_ready()
say(f'sum over all devices = {float(np.asarray(s.addressable_data(0)))} (expected '
    f'{4096 * n_local * sum(range(1, jax.process_count() + 1))})')
w = jax.device_put(jnp.zeros((4096, 64), jnp.float32), NamedSharding(mesh, P()))
t1 = time.time()
w = step(w, x); w.block_until_ready()
say(f'first step (compile) {time.time() - t1:.1f} s')
t1 = time.time()
for _ in range(50):
    w = step(w, x)
w.block_until_ready()
say(f'50 steps {time.time() - t1:.2f} s; done')
faulthandler.cancel_dump_traceback_later()
