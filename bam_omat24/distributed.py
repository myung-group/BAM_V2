"""Multi-process (multi-node) start-up for data parallel training (review §13).

Imports only os and jax, so that bam_omat24/__init__.py can call init_distributed() before the
model modules, whose import initialises the XLA backend.
"""
import os

import jax


def _initialized() -> bool:
    if hasattr(jax.distributed, "is_initialized"):
        return jax.distributed.is_initialized()
    from jax._src import distributed as _d
    return _d.global_state.client is not None


def _tasks_on_this_node(env) -> int:
    """Tasks of this srun step on the current node, from SLURM_STEP_TASKS_PER_NODE (or
    SLURM_TASKS_PER_NODE), e.g. "4(x2),2", indexed by SLURM_NODEID.  SLURM_NTASKS_PER_NODE is set
    only with an explicit --ntasks-per-node and is not used."""
    spec = env.get("SLURM_STEP_TASKS_PER_NODE") or env.get("SLURM_TASKS_PER_NODE")
    if not spec:
        return 1
    counts = []
    for tok in spec.split(","):
        if "(x" in tok:
            n, k = tok.split("(x")
            counts += [int(n)] * int(k.rstrip(")"))
        else:
            counts.append(int(tok))
    node = int(env.get("SLURM_NODEID", "0"))
    return counts[node] if node < len(counts) else counts[0]


def _visible_gpus(env) -> int:
    vis = env.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return len([v for v in vis.split(",") if v.strip()])
    return int(env.get("SLURM_GPUS_ON_NODE", "1"))


def init_distributed() -> None:
    """Start the JAX multi-process runtime for multi-node data parallel training.  Opt-in: does
    nothing unless BAM_DISTRIBUTED=1 (so a single-process run inside a multi-task allocation is
    unaffected).  Must run before the first JAX computation.

    Launch with srun, one process per node (all GPUs of the node, recommended) or one per GPU, and
    export BAM_DISTRIBUTED=1: bam_omat24/__init__.py then calls this before the model modules are
    imported (importing them initialises the XLA backend, after which initialize() is refused).
    Read from the environment (BAM_* overrides SLURM_*):
      BAM_NUM_PROCESSES / SLURM_NTASKS, BAM_PROCESS_ID / SLURM_PROCID,
      BAM_COORDINATOR (host:port) or MASTER_ADDR + MASTER_PORT,
      BAM_LOCAL_DEVICE_IDS ("0,1,2,3"), otherwise derived from the tasks on this node and the GPUs
      this process sees: one task on the node -> every visible GPU; one task per GPU with all GPUs
      visible (--gpu-bind=none) -> [SLURM_LOCALID]; a task bound to its own GPU(s) (--gpus-per-task)
      -> every visible GPU.
    The local device ids are always passed explicitly: JAX's Slurm auto-detection would otherwise
    set [SLURM_LOCALID] and a one-process-per-node job would drive only GPU 0 of each node.
    """
    env = os.environ
    if env.get("BAM_DISTRIBUTED") != "1" or _initialized():
        return
    n_proc = int(env.get("BAM_NUM_PROCESSES", env.get("SLURM_NTASKS", "1")))
    if n_proc <= 1:
        return
    pid = int(env.get("BAM_PROCESS_ID", env.get("SLURM_PROCID", "0")))
    coord = env.get("BAM_COORDINATOR")
    if coord is None:
        if "MASTER_ADDR" not in env:
            raise RuntimeError("BAM_DISTRIBUTED=1 needs BAM_COORDINATOR or MASTER_ADDR (+ MASTER_PORT)")
        coord = f"{env['MASTER_ADDR']}:{env.get('MASTER_PORT', '29500')}"
    if "BAM_LOCAL_DEVICE_IDS" in env:
        local_ids = [int(i) for i in env["BAM_LOCAL_DEVICE_IDS"].split(",")]
    else:
        per_node, n_vis = _tasks_on_this_node(env), _visible_gpus(env)
        lid = int(env.get("SLURM_LOCALID", "0"))
        if per_node == 1 or n_vis < per_node:
            local_ids = list(range(n_vis))           # whole node, or GPUs bound to this task
        elif lid < n_vis:
            local_ids = [lid]                         # one task per GPU, all GPUs visible
        else:
            raise RuntimeError(f"cannot map local task {lid} to one of {n_vis} visible GPUs; set BAM_LOCAL_DEVICE_IDS")
    jax.distributed.initialize(coordinator_address=coord, num_processes=n_proc, process_id=pid,
                               local_device_ids=local_ids)
    print(f"jax.distributed: process {jax.process_index()}/{jax.process_count()}, "
          f"local devices {jax.local_devices()}", flush=True)
