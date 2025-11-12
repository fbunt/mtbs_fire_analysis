from __future__ import annotations

from dataclasses import dataclass

from mtbs_fire_analysis.analysis.hlh_dist import HalfLifeHazardDistribution
from mtbs_fire_analysis.analysis.scipy_dist import InverseGauss, Weibull


@dataclass(frozen=True)
class SimulationDefaults:
    num_pixels: int = 5000
    time_interval: int = 39
    iterations: int = 100
    pre_window: int = 500
    random_seed: int = 1989


@dataclass(frozen=True)
class HLHDefaults:
    hazard_inf: float = 0.03
    half_life: float = 50.0

    def build(self) -> HalfLifeHazardDistribution:
        return HalfLifeHazardDistribution(self.hazard_inf, self.half_life)


@dataclass(frozen=True)
class WeibullDefaults:
    shape: float = 1.5
    scale: float = 85.0

    def build(self) -> Weibull:
        return Weibull(shape=self.shape, scale=self.scale)


@dataclass(frozen=True)
class InverseGaussDefaults:
    mu: float = 75.0
    lam: float = 200.0

    def build(self) -> InverseGauss:
        return InverseGauss(mu=self.mu, lam=self.lam)


SIM = SimulationDefaults()
HLH_DEF = HLHDefaults()
WEIBULL_DEF = WeibullDefaults()
IG_DEF = InverseGaussDefaults()
