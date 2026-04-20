"""Harold model package.

Public API:

- :class:`Harold` — top-level model
- :func:`build_model` — factory from :class:`ModelConfig`
- :class:`FlowMatchingSchedule` — noise schedule (useful for samplers)

Internal modules (``blocks``, ``attention``, ``ssm``, ``moe``, ``norm``,
``quantization``) are importable explicitly via their submodule path but are
not part of the stable API.
"""

from .harold import Harold, build_model
from .schedule import FlowMatchingSchedule

__all__ = ["Harold", "build_model", "FlowMatchingSchedule"]