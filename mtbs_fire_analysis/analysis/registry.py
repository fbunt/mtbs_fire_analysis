"""Central distribution registry.

Provides a single source of truth for available lifetime distributions and
simple helpers to construct or enumerate them. Keeps SciPy-backed models and
custom models (HLH) decoupled from each other.
"""
from __future__ import annotations

from typing import Callable, Dict

from .hlh_dist import HalfLifeHazardDistribution
from .lifetime_base import BaseLifetime
from .scipy_dist import InverseGauss, Weibull


# Public registry mapping short names to constructors
REGISTRY: Dict[str, Callable[..., BaseLifetime]] = {
    "hlh": HalfLifeHazardDistribution,
    "weibull": Weibull,
    "invgauss": InverseGauss,
}


def make(name: str, /, **kwargs) -> BaseLifetime:
    """Construct a distribution by registry name.

    Raises KeyError if name is unknown (message lists valid options).
    """
    try:
        ctor = REGISTRY[name]
    except KeyError as e:  # provide helpful message
        opts = ", ".join(sorted(REGISTRY))
        raise KeyError(
            f"Unknown distribution '{name}'. Options: {opts}"
        ) from e
    return ctor(**kwargs)


def list_distributions() -> list[str]:
    """Return sorted list of available distribution names."""
    return sorted(REGISTRY)
