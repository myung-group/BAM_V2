"""JAX RACEv2 checkpoint -> race_v2_torch model file, without JAX.

Sources:
  * a pickled training checkpoint (race_v2 ``save_checkpoint``: dict with 'ema_params', 'params', 'config', ...),
    read with a stub unpickler: flax / jax classes become stubs, jax arrays are rebuilt as numpy arrays;
  * or a ``params.npz`` (keys = ``jax.tree_util.keystr`` of the nnx.Param paths, written by
    tools/dump_jax_reference.py) + ``config.json``.

    python -m race_v2_torch.convert --ckpt runs/<run>/ckpt_<run>.pkl --out model.pt [--which ema_params]
    python -m race_v2_torch.convert --npz ref_dir/params.npz --config ref_dir/config.json --out model.pt

The variance heads (frc_logvar, energy_logvar, stress_logvar) and the unused per-species x projection (x_linear, x_mode
basis) are not converted.  Every other parameter must match a torch parameter of the same path and shape.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .model import RACEv2, resolve_config

IGNORED_TOP = ('frc_logvar', 'energy_logvar', 'stress_logvar', 'x_linear')


# ----------------------------------------------------------------------------------------------- pickle without jax
class _Stub:
    def __new__(cls, *a, **k):
        o = object.__new__(cls)
        o.args = a
        o.state = None
        return o

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.state = s


def _rebuild_jax_array(fun, args, arr_state, aval_state=None):
    a = fun(*args)
    a.__setstate__(arr_state)
    return a


class _NoJaxUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core') and not hasattr(np, '_core'):
            module = 'numpy.core' + module[len('numpy._core'):]          # pickled with numpy 2, read with numpy 1
        if module.split('.')[0] in ('numpy', 'builtins', 'collections', '_codecs', 'copyreg'):
            return super().find_class(module, name)
        if name == '_reconstruct_array':                                # jax.Array.__reduce__
            return _rebuild_jax_array
        return type(name, (_Stub,), {'_module': module})


def _flatten_state(o, path, out):
    if isinstance(o, _Stub):
        cls = type(o).__name__
        st = o.state if isinstance(o.state, dict) else {}
        if '_mapping' in st:                                            # nnx.State
            _flatten_state(st['_mapping'], path, out)
        elif '_raw_value' in st or 'raw_value' in st or 'value' in st:   # nnx.Param (Variable)
            v = st.get('_raw_value', st.get('raw_value', st.get('value')))
            out[path] = np.asarray(v)
        else:
            raise ValueError(f"unexpected object {o._module}.{cls} at {'/'.join(map(str, path))}")
    elif isinstance(o, dict):
        for k, v in o.items():
            _flatten_state(v, path + (k,), out)
    elif isinstance(o, np.ndarray):
        out[path] = o
    else:
        raise ValueError(f"unexpected leaf {type(o)} at {'/'.join(map(str, path))}")


def load_pickled_checkpoint(path, which: str = 'ema_params') -> Tuple[Dict[tuple, np.ndarray], Dict]:
    with open(path, 'rb') as f:
        ck = _NoJaxUnpickler(f).load()
    if which not in ck:
        raise KeyError(f"{path}: no {which!r} (keys {list(ck)})")
    params: Dict[tuple, np.ndarray] = {}
    _flatten_state(ck[which], (), params)
    return params, ck.get('config')


# ----------------------------------------------------------------------------------------------- params.npz
_TOKEN = re.compile(r"\[\s*'([^']*)'\s*\]|\[\s*\"([^\"]*)\"\s*\]|\[\s*(-?\d+)\s*\]|\.([A-Za-z_][A-Za-z0-9_]*)")
_VALUE_ATTRS = {'value', 'raw_value', '_raw_value'}


def parse_keystr(key: str) -> tuple:
    """"['layers'][0]['skip']['weight_w_0_0_128x0e_128x0e'].value" -> ('layers', 0, 'skip', 'weight_...')."""
    toks, pos = [], 0
    for m in _TOKEN.finditer(key):
        if m.start() != pos:
            raise ValueError(f"cannot parse parameter key {key!r}")
        pos = m.end()
        s1, s2, i, attr = m.groups()
        if i is not None:
            toks.append(int(i))
        elif attr is not None:
            toks.append(attr)
        else:
            toks.append(s1 if s1 is not None else s2)
    if pos != len(key):
        raise ValueError(f"cannot parse parameter key {key!r}")
    while toks and isinstance(toks[-1], str) and toks[-1] in _VALUE_ATTRS:
        toks.pop()
    if '_mapping' in toks:
        toks = [t for t in toks if t != '_mapping']
    return tuple(toks)


def load_params_npz(path) -> Dict[tuple, np.ndarray]:
    z = np.load(path)
    return {parse_keystr(k): np.asarray(z[k]) for k in z.files}


# ----------------------------------------------------------------------------------------------- conversion
def build_from_params(params: Dict[tuple, np.ndarray], cfg: Dict, strict: bool = True) -> Tuple[RACEv2, Dict]:
    """Torch model with the JAX parameters; returns (model, report)."""
    model = RACEv2(resolve_config(cfg))
    sd = model.state_dict()
    targets = {k for k, _ in model.named_parameters()}
    new, used, ignored, dtypes = {}, set(), [], set()
    for path, arr in params.items():
        if path and path[0] in IGNORED_TOP:
            ignored.append(path)
            continue
        key = '.'.join(str(p) for p in path)
        if key not in targets:
            raise KeyError(f"JAX parameter {'/'.join(map(str, path))} {arr.shape} has no torch counterpart {key!r}")
        if tuple(sd[key].shape) != tuple(arr.shape):
            raise ValueError(f"shape mismatch for {key}: JAX {arr.shape}, torch {tuple(sd[key].shape)}")
        dtypes.add(str(arr.dtype))
        new[key] = torch.from_numpy(np.array(arr, dtype=np.float64))
        used.add(key)
    missing = sorted(targets - used)
    if missing and strict:
        raise KeyError(f"torch parameters without a JAX value: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    sd.update(new)
    model.load_state_dict(sd, strict=True)
    n = sum(int(np.prod(params[p].shape)) for p in params if not (p and p[0] in IGNORED_TOP))
    report = dict(n_converted=len(used), n_values=n, n_ignored=len(ignored),
                  ignored=sorted({str(p[0]) for p in ignored}), source_dtypes=sorted(dtypes), missing=missing)
    return model.eval(), report


def convert(out, ckpt=None, npz=None, config=None, which='ema_params', core_exclusion=None) -> Tuple[RACEv2, Dict]:
    if ckpt is not None:
        params, cfg = load_pickled_checkpoint(ckpt, which)
        source = dict(kind='pickle', path=str(ckpt), which=which)
    else:
        params, cfg = load_params_npz(npz), None
        source = dict(kind='npz', path=str(npz))
    if config is not None:
        cfg = json.load(open(config)) if not isinstance(config, dict) else dict(config)
    if cfg is None:
        raise ValueError("no config: the checkpoint has none, pass --config")
    model, report = build_from_params(params, cfg)
    if core_exclusion is not None:
        model.set_core_exclusion(core_exclusion)
        source['core_exclusion'] = {k: v for k, v in core_exclusion.items()}
    if out is not None:
        model.save(out, config=cfg, extra=dict(source=source, report=report))
    return model, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--ckpt', help='pickled race_v2 checkpoint (.pkl)')
    g.add_argument('--npz', help='params.npz of tools/dump_jax_reference.py')
    ap.add_argument('--config', help='config.json (required with --npz; overrides the checkpoint config)')
    ap.add_argument('--which', default='ema_params', help="checkpoint entry: ema_params (default, as predict.py) or params")
    ap.add_argument('--core-exclusion', nargs='+', default=None, metavar='MINDIST_JSON',
                    help="store the core exclusion in the model file: mindist JSON table(s) (cap 0.5, width 0.1), or "
                         "'default' for the table of the RACE v2.1 training data.  Without it the model file carries "
                         "the checkpoint config's own core_exclusion, if any; the calculator switches the default "
                         "table on for a file without one")
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ce = None
    if a.core_exclusion:
        from .calculator import DEFAULT_CORE_EXCLUSION
        ce = (dict(DEFAULT_CORE_EXCLUSION) if a.core_exclusion == ['default']
              else {'table': list(a.core_exclusion), 'cap': 0.5, 'width': 0.1})
    model, report = convert(a.out, ckpt=a.ckpt, npz=a.npz, config=a.config, which=a.which, core_exclusion=ce)
    n_t = sum(p.numel() for p in model.parameters())
    print(f"converted {report['n_converted']} arrays ({report['n_values']:,} values; torch model {n_t:,} parameters), "
          f"ignored {report['n_ignored']} arrays of {report['ignored']}, source dtypes {report['source_dtypes']} "
          f"-> {a.out}")


if __name__ == '__main__':
    main()
