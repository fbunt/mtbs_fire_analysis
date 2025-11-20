from __future__ import annotations

from dataclasses import dataclass

from mtbs_fire_analysis.analysis.default_params import (
    HLHDefaults as _HLHDefaults,
    InverseGaussDefaults as _InverseGaussDefaults,
    SimulationDefaults as _SimulationDefaults,
    WeibullDefaults as _WeibullDefaults,
)
from mtbs_fire_analysis.analysis.hlh_dist import HalfLifeHazardDistribution
from mtbs_fire_analysis.analysis.scipy_dist import InverseGauss, Weibull


# Re-export dataclasses at this location for backward compatibility and
# provide build() helpers used by statistical tests


@dataclass(frozen=True)
class SimulationDefaults(_SimulationDefaults):
    pass


@dataclass(frozen=True)
class HLHDefaults(_HLHDefaults):
    def build(self) -> HalfLifeHazardDistribution:
        return HalfLifeHazardDistribution(self.hazard_inf, self.half_life)


@dataclass(frozen=True)
class WeibullDefaults(_WeibullDefaults):
    def build(self) -> Weibull:
        return Weibull(shape=self.shape, scale=self.scale)


@dataclass(frozen=True)
class InverseGaussDefaults(_InverseGaussDefaults):
    def build(self) -> InverseGauss:
        return InverseGauss(mu=self.mu, lam=self.lam)


SIM = SimulationDefaults()
HLH_DEF = HLHDefaults()
WEIBULL_DEF = WeibullDefaults()
IG_DEF = InverseGaussDefaults()
