"""
Central default parameters for lifetime models and simulations.

Single source of truth for class constructor defaults and study/test defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# Model constructor defaults (used by classes directly)
HLH_DEFAULTS: Dict[str, float] = {"hazard_inf": 0.03, "half_life": 50.0}
WEIBULL_DEFAULTS: Dict[str, float] = {"shape": 1.5, "scale": 85.0}
IG_DEFAULTS: Dict[str, float] = {"mu": 75.0, "lam": 200.0}


# Simulation defaults (shared by statistical tests and tools)
@dataclass(frozen=True)
class SimulationDefaults:
    num_pixels: int = 5000
    time_interval: int = 39
    iterations: int = 100
    pre_window: int = 500
    random_seed: int = 1989


@dataclass(frozen=True)
class HLHDefaults:
    hazard_inf: float = HLH_DEFAULTS["hazard_inf"]
    half_life: float = HLH_DEFAULTS["half_life"]


@dataclass(frozen=True)
class WeibullDefaults:
    shape: float = WEIBULL_DEFAULTS["shape"]
    scale: float = WEIBULL_DEFAULTS["scale"]


@dataclass(frozen=True)
class InverseGaussDefaults:
    mu: float = IG_DEFAULTS["mu"]
    lam: float = IG_DEFAULTS["lam"]
