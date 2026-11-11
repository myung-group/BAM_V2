"""Worker side of the streaming loader (stream_data.StreamLoader): JAX-free, and it must stay so.

This file is imported as the TOP-LEVEL module ``stream_worker`` (its directory is on sys.path), never as
``bam_omat24.stream_worker``: a forkserver worker unpickles the functions by module name, and importing the package
``bam_omat24`` would load JAX and initialise the GPU backend in every worker (2026-11-11: that killed the workers
on HANGANG).  ``clean_main()`` stops the workers from re-importing the parent's ``__main__`` (train_3bpa) as well.
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


def _worker_init(kinds, files, atomic_numbers, atom_energies, cutoff, fields=None):
    if 'jax' in sys.modules and os.environ.get('BAM_STREAM_ALLOW_JAX_WORKERS') != '1':
        raise RuntimeError("stream worker: jax is imported in a loader worker (workers must be JAX-free; see "
                           "stream_worker.py)")
    import stream_io as S
    from graph_fields import field_keys
    _W['S'] = S
    _W['src'] = {}
    _W['fields'] = dict(fields or {})
    _W['keys'] = field_keys(**_W['fields'])
    _W['kinds'], _W['files'] = kinds, files
    _W['idx'] = {int(z): i for i, z in enumerate(sorted(atomic_numbers))}
    _W['e0'], _W['cutoff'] = list(atom_energies), float(cutoff)


def _source(fi):
    S, src = _W['S'], _W['src']
    if fi in src:
        src[fi] = src.pop(fi)                      # most recently used last
        return src[fi]
    while len(src) >= MAX_OPEN:
        old = src.pop(next(iter(src)))
        for env in getattr(old, '_envs', {}).values():
            env.close()
    kind, path = _W['kinds'][fi], _W['files'][fi]
    src[fi] = (S.ExtxyzSource([path], info_keys=_W['keys'][0], array_keys=_W['keys'][1])
               if kind == 'extxyz' else S.AselmdbSource([path]))
    return src[fi]


def _worker_block(task):
    """task = (seq, file, keys) -> (seq, list of graph-array dicts)."""
    S = _W['S']
    seq, fi, keys = task
    sts = _source(fi).read_block(0, keys)
    return seq, [S.graph_arrays(st, _W['idx'], _W['e0'], _W['cutoff'], _W['fields']) for st in sts]
