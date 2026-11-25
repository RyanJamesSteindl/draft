"""The hexapod locomotion task (milestone M1 of the hexapod-sim project).

Same recipe as the flat ``quadruped`` task, for six legs named
``{left,right}_{front,mid,rear}``; see ``base_env.py`` for what differs.
Re-exports are lazy, as in ``quadruped/__init__.py``.
"""

import importlib

_LAZY = {
    "make_env_cfg": ".env",
    "make_hexapod_entity_cfg": ".hexapod_entity",
    "make_runner_cfg": ".runner_cfg",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
