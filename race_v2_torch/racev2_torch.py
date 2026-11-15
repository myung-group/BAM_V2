"""Compatibility alias (2026-11-16): ``race_v2_torch`` -> ``race_v2_torch`` (same folder on the path)."""
import importlib as _importlib
import sys as _sys

_new = _importlib.import_module('race_v2_torch')
for _m in ['calculator', 'constants', 'convert', 'model', 'o3']:
    _sys.modules[f'race_v2_torch.{_m}'] = _importlib.import_module(f'race_v2_torch.{_m}')
_sys.modules[__name__] = _new
