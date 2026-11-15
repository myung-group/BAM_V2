"""Worker side of the streaming loader (stream_data.StreamLoader): JAX-free, and it must stay so.

This file is imported as the TOP-LEVEL module ``stream_worker`` (its directory is on sys.path), never as
``race_v2.data.stream_worker``: a forkserver worker unpickles the functions by module name, and importing the package
``race_v2`` would load JAX and initialise the GPU backend in every worker (2026-11-11: that killed the workers
on HANGANG).  ``clean_main()`` stops the workers from re-importing the parent's ``__main__`` (training.train) as well.
"""
import os
import sys

_W = {}
MAX_OPEN = 32            # open sources (lmdb environments / xyz indices) kept per worker


def clean_main():
    """Make multiprocessing start children without re-running the parent's __main__ module (which imports JAX).
    Safe when no pickled function lives in __main__; call once in the parent before the pool starts."""
    from multiprocessing import spawn
    if getattr(spawn.get_preparation_data, '_bam_clean', False):
        return
    orig = spawn.get_preparation_data

    def get_preparation_data(name):
        d = orig(name)
        d.pop('init_main_from_name', None)
        d.pop('init_main_from_path', None)
        return d
    get_preparation_data._bam_clean = True
    spawn.get_preparation_data = get_preparation_data


def _worker_init(kinds, files, atomic_numbers, atom_energies, cutoff, fields=None, opts=None, e0_tables=None):
    """opts: source options of every file (stream_io.file_opts); e0_tables: {task: E0 per species index}."""
    if 'jax' in sys.modules and os.environ.get('BAM_STREAM_ALLOW_JAX_WORKERS') != '1':
        raise RuntimeError("stream worker: jax is imported in a loader worker (workers must be JAX-free; see "
                           "stream_worker.py)")
    import numpy as np
    import stream_io as S
    from graph_fields import field_keys
    _W['S'] = S
    _W['src'] = {}
    _W['fields'] = dict(fields or {})
    _W['keys'] = field_keys(**{k: v for k, v in _W['fields'].items()
                               if k not in ('mixed', 'dipole', 'opt_charge_labels', 'groups', 'slab', 'quadrupole',
                                            'spin_labels', 'charge_nbo_labels')})
    _W['meta'] = {}
    _W['opts'] = list(opts) if opts else [{} for _ in files]
    _W['e0_tables'] = {int(k): np.asarray(v, np.float64) for k, v in (e0_tables or {}).items()}
    _W['kinds'], _W['files'] = kinds, files
    _W['idx'] = {int(z): i for i, z in enumerate(sorted(atomic_numbers))}
    _W['e0'], _W['cutoff'] = list(atom_energies), float(cutoff)


def _source(fi):
    # cached by (kind, path), not by file index: stream.train.parts gives every repeated copy its own file indices
    # for the same paths, and an lmdb environment may be opened only once per process
    S, src = _W['S'], _W['src']
    kind, path = _W['kinds'][fi], _W['files'][fi]
    k = (kind, path)
    if k in src:
        src[k] = src.pop(k)                        # most recently used last
        return src[k]
    while len(src) >= MAX_OPEN:
        old = src.pop(next(iter(src)))
        for env in getattr(old, '_envs', {}).values():
            env.close()
    src[k] = (S.ExtxyzSource([path], info_keys=_W['keys'][0], array_keys=_W['keys'][1])
              if kind == 'extxyz' else S.AselmdbSource([path]))
    return src[k]


def _worker_block(task):
    """task = (seq, file, keys) -> (seq, list of graph-array dicts)."""
    S = _W['S']
    seq, fi, keys = task
    sts = _source(fi).read_block(0, keys)
    if _W['fields'].get('groups') and _W['kinds'][fi] == 'aselmdb':
        # charge-constraint groups of the rows from the shard's meta-data (memory-mapped)
        if fi not in _W['meta']:
            mp_ = S.meta_path(_W['opts'][fi].get('meta_dir'), _W['files'][fi])
            _W['meta'][fi] = S.ShardMeta(mp_) if mp_ is not None else None
        if _W['meta'][fi] is not None:
            for st, k in zip(sts, keys):
                st['groups'] = _W['meta'][fi].get(int(k))
    return seq, [S.graph_arrays(st, _W['idx'], _W['e0'], _W['cutoff'], _W['fields'], _W['opts'][fi], _W['e0_tables'])
                 for st in sts]


def _worker_state(names=('jax', 'race_v2', 'race_v2', 'data')):
    """Diagnostics: (pid, which of the modules `names` this worker has loaded, its number of threads)."""
    threads = -1
    try:
        with open('/proc/self/status') as fh:
            threads = next(int(line.split()[1]) for line in fh if line.startswith('Threads'))
    except OSError:
        pass
    return os.getpid(), [n for n in names if n in sys.modules], threads
