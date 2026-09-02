"""Detector registry.

Detectors are registered by name so that ``--only`` / ``--skip`` on the CLI and
third-party plugins both work against the same list.  Order here is the order
findings appear in the report: cheapest-to-act-on first.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Type

from .advantage_collapse import AdvantageCollapse
from .base import Detector, Finding, Severity
from .clip_saturation import ClipSaturation
from .entropy_collapse import EntropyCollapse
from .grad_pathology import GradientPathology
from .kl_drift import KLDrift
from .length_hacking import LengthPathology
from .plateau import Plateau
from .reward_composition import RewardComposition
from .reward_hacking import RewardHacking

BUILTIN_DETECTORS: List[Type[Detector]] = [
    AdvantageCollapse,
    EntropyCollapse,
    RewardHacking,
    LengthPathology,
    ClipSaturation,
    KLDrift,
    GradientPathology,
    RewardComposition,
    Plateau,
]

_REGISTRY: Dict[str, Type[Detector]] = {cls.name: cls for cls in BUILTIN_DETECTORS}


def register(cls: Type[Detector]) -> Type[Detector]:
    """Register a custom detector. Usable as a decorator."""
    if not issubclass(cls, Detector):
        raise TypeError(f"{cls!r} is not a Detector subclass")
    _REGISTRY[cls.name] = cls
    return cls


def available() -> List[str]:
    return list(_REGISTRY)


def get(name: str) -> Type[Detector]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown detector {name!r}; available: {', '.join(sorted(_REGISTRY))}") from None


def build(only: Iterable[str] = (), skip: Iterable[str] = ()) -> List[Detector]:
    """Instantiate the selected detectors, preserving registry order."""
    only_set, skip_set = set(only), set(skip)
    for name in only_set | skip_set:
        get(name)  # raises with a helpful message on typos
    names = [n for n in _REGISTRY if (not only_set or n in only_set) and n not in skip_set]
    return [_REGISTRY[n]() for n in names]


__all__ = [
    "BUILTIN_DETECTORS",
    "Detector",
    "Finding",
    "Severity",
    "available",
    "build",
    "get",
    "register",
]
