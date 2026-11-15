"""Compatibility alias (2026-11-16): the package ``racev2_torch`` is now ``race_v2_torch`` (folder race_v2_torch/).

``import racev2_torch`` and ``python -m racev2_torch.convert`` keep working when this folder (the old
``<repo>/racev2_torch``) or the new ``<repo>/race_v2_torch`` is on the path.  Saved model files keep the format tag
``racev2_torch``.
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.realpath(__file__))))
_PATH = _os.path.join(_ROOT, 'race_v2_torch')
if _PATH not in _sys.path:
    _sys.path.insert(0, _PATH)

import race_v2_torch as _new  # noqa: E402
from race_v2_torch import *  # noqa: E402,F401,F403
from race_v2_torch import RACEv2, resolve_config, FORMAT, FORMAT_VERSION  # noqa: E402,F401
