"""
Generic SciPy-backed lifetime distributions for renewal-style likelihoods
=========================================================================

This module provides a small interface compatible with your existing
`HalfLifeHazardDistribution` (HLH) so you can plug in **any** SciPy
continuous distribution that exposes `pdf`, `sf`, and `mean`.

It implements the extra likelihood pieces you use:
  • forward‑recurrence initial gaps  →  log S(g) − log μ
  • empty censor windows of width W  →  log ∫_W^∞ S(z)dz − log μ
and an `expected_hazard_ge(W)` helper for empty‑window hazard calculation:

    E[ h(T) | T ≥ W ] = ∫_W^∞ h(t) S(t) dt / ∫_W^∞ S(t) dt
                       = S(W) / ∫_W^∞ S(t) dt   (since h·S = f)

A tiny **registry** lets you select distributions by name, mixing HLH and
SciPy-backed ones behind the same API.

Usage (see __main__ at bottom for a runnable example):

    model = REGISTRY["weibull"](shape=1.7, scale=12.0)
    res = model.fit(data=dts, survival_data=rights,
                    initial_gaps=gaps, empty_windows=Ws)
    print(model.params)

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate, optimize, stats
from scipy.special import gamma, gammaincc
from .lifetime_base import BaseLifetime

_EPS = 1e-300  # numeric safety for logs/divisions
_QUAD_OPTS = {"epsrel": 1e-9, "epsabs": 0.0, "limit": 200}


# BaseLifetime moved to lifetime_base.py and imported above.


# -----------------------------------------------------------------------------
# A concrete SciPy-backed Parametric base: map unconstrained θ → frozen rv.
# We implement caching so pdf/sf/mean call the underlying SciPy object.
# -----------------------------------------------------------------------------
class SciPyParametric(BaseLifetime):
    """Base class to wrap a SciPy rv_continuous via a θ → frozen‑rv mapping."""

    def __init__(self) -> None:
        self._rv: Optional[stats.rv_continuous] = None  # frozen instance

    # ---- BaseLifetime primitives use the frozen rv ------------------------
    def pdf(self, x):
        return self._rv.pdf(x)  # type: ignore[arg-type]

    def sf(self, x):
        return self._rv.sf(x)  # type: ignore[arg-type]

    def mean(self) -> float:
        m = float(self._rv.mean())
        if np.isfinite(m):
            return m
        # Fallback numeric mean via ∫_0^∞ S(u) du when SciPy returns inf
        m_num = integrate.quad(
            lambda z: float(max(self.sf(z), 0.0)), 0.0, np.inf, **_QUAD_OPTS
        )[0]
        return float(m_num)

    # ---- subclasses must call this after changing θ -----------------------
    def _set_frozen(self, rv: stats.rv_continuous) -> None:
        self._rv = rv

    # --------------------------- random variates --------------------------
    def rvs(self, size: int = 1, rng=None):
        """Draw random lifetimes using the frozen SciPy distribution.
        Accepts an np.random.Generator (preferred) or int seed.
        """
        return self._rv.rvs(size=size, random_state=rng)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Example: Weibull (weibull_min).  Parametrised with positive (shape, scale).
# We optimise in log‑space for stability: θ = (log k, log λ)
# -----------------------------------------------------------------------------
class Weibull(SciPyParametric):
    def __init__(self, shape: float = 1.0, scale: float = 1.0) -> None:
        super().__init__()
        self._theta = np.array([np.log(shape), np.log(scale)], float)
        self._refresh()

    # parametrisation hooks
    def _theta_get(self) -> np.ndarray:
        return self._theta.copy()

    def _theta_set(self, theta: np.ndarray) -> None:
        self._theta = np.asarray(theta, float)
        self._refresh()

    def _refresh(self) -> None:
        k, lam = np.exp(self._theta)  # k>0, lam>0
        # scipy: weibull_min(c=k, loc=0, scale=lam)
        self._set_frozen(stats.weibull_min(c=k, scale=lam))
        self.shape, self.scale = float(k), float(lam)

    @property
    def params(self) -> Dict[str, float]:
        return {"shape": self.shape, "scale": self.scale}

    # Closed-form tail survival integral: ∫_W^∞ S(t) dt = (λ / k) Γ(1/k, (W/λ)^k)
    def tail_survival(self, w: np.ndarray | float) -> np.ndarray | float:  # type: ignore[override]
        w_arr = np.atleast_1d(np.asarray(w, float))
        k, lam = float(self.shape), float(self.scale)
        # Guard invalid parameters to avoid crashes inside optimizer explorations
        if (not np.isfinite(k)) or (not np.isfinite(lam)) or (k <= 0.0) or (lam <= 0.0):
            bad = np.full_like(w_arr, np.inf, dtype=float)
            return bad if np.ndim(w) else float(bad[0])
        a = 1.0 / k
        u = np.maximum(w_arr / lam, 0.0) ** k
        tail = (lam / k) * gamma(a) * gammaincc(a, u)
        return tail if np.ndim(w) else float(tail[0])

    def mean(self) -> float:
        """Closed-form Weibull mean: μ = λ Γ(1 + 1/k)."""
        k, lam = float(self.shape), float(self.scale)
        if (not np.isfinite(k)) or (not np.isfinite(lam)) or (k <= 0.0) or (lam <= 0.0):
            return np.inf
        val = lam * gamma(1.0 + 1.0 / k)
        return float(val) if np.isfinite(val) else np.inf

    # Reasonable, wide bounds in log-space to keep optimizer from k→0 or extreme λ
    def default_bounds(self) -> Sequence[Tuple[float, float]]:
        k_min, k_max = 1e-3, 1e3
        lam_min, lam_max = 1e-3, 1e5
        return [(np.log(k_min), np.log(k_max)), (np.log(lam_min), np.log(lam_max))]

    # Soft penalty to keep search in a practical region; smooth quadratic outside box
    def _soft_penalty(self) -> float:
        k, lam = float(self.shape), float(self.scale)
        # Heavy penalty if invalid to steer optimizer away quickly
        if (not np.isfinite(k)) or (not np.isfinite(lam)) or (k <= 0.0) or (lam <= 0.0):
            return 1e9
        # Preferred box (domain-specific; adjust if needed)
        k_lo, k_hi = 0.3, 10.0
        lam_lo, lam_hi = 1.0, 200.0
        strength = 1e3
        pen = 0.0
        if k < k_lo:
            d = (k_lo - k) / k_lo
            pen += d * d
        elif k > k_hi:
            d = (k - k_hi) / k_hi
            pen += d * d
        if lam < lam_lo:
            d = (lam_lo - lam) / lam_lo
            pen += d * d
        elif lam > lam_hi:
            d = (lam - lam_hi) / lam_hi
            pen += d * d
        return float(strength * pen)


# -----------------------------------------------------------------------------
# Inverse Gaussian (Wald) using SciPy's invgauss; expose IG(mu, lam) params
# Mapping to SciPy: if Y ~ invgauss(mu_s), X = scale * Y has
#   E[X] = mu_s * scale, Var[X] = mu_s^3 * scale^2.
# To match IG(mean=mu, shape=lam), set mu_s = mu/lam, scale = lam.
# -----------------------------------------------------------------------------
class InverseGauss(SciPyParametric):
    def __init__(self, mu: float = 1.0, lam: float = 1.0) -> None:
        super().__init__()
        if mu <= 0 or lam <= 0:
            raise ValueError("mu and lam must be positive.")
        self._theta = np.array([np.log(mu), np.log(lam)], float)
        self._refresh()

    def _theta_get(self) -> np.ndarray:
        return self._theta.copy()

    def _theta_set(self, theta: np.ndarray) -> None:
        self._theta = np.asarray(theta, float)
        self._refresh()

    def _refresh(self) -> None:
        mu, lam = np.exp(self._theta)
        mu_s = mu / lam
        scale = lam
        self._set_frozen(stats.invgauss(mu=mu_s, scale=scale))
        self.mu, self.lam = float(mu), float(lam)

    @property
    def params(self) -> Dict[str, float]:
        return {"mu": self.mu, "lam": self.lam}

    def mean(self) -> float:  # closed form
        return float(self.mu)

    def default_bounds(self) -> Sequence[Tuple[float, float]]:
        mu_min, mu_max = 1e-3, 1e4
        lam_min, lam_max = 1e-3, 1e5
        return [
            (np.log(mu_min), np.log(mu_max)),
            (np.log(lam_min), np.log(lam_max)),
        ]

    def _soft_penalty(self) -> float:
        mu, lam = float(self.mu), float(self.lam)
        if (
            (not np.isfinite(mu))
            or (not np.isfinite(lam))
            or (mu <= 0.0)
            or (lam <= 0.0)
        ):
            return 1e9
        mu_lo, mu_hi = 1e-2, 1e3
        lam_lo, lam_hi = 1e-2, 1e4
        strength = 1e3
        pen = 0.0
        if mu < mu_lo:
            d = (mu_lo - mu) / mu_lo
            pen += d * d
        elif mu > mu_hi:
            d = (mu - mu_hi) / mu_hi
            pen += d * d
        if lam < lam_lo:
            d = (lam_lo - lam) / lam_lo
            pen += d * d
        elif lam > lam_hi:
            d = (lam - lam_hi) / lam_hi
            pen += d * d
        return float(strength * pen)


# -----------------------------------------------------------------------------
# Optional: hook in your bespoke HLH into the same registry if available
# -----------------------------------------------------------------------------
try:
    # Your local file may be `hlh_dist.py` with class
    # `HalfLifeHazardDistribution`
    from hlh_dist import HalfLifeHazardDistribution
except Exception:  # pragma: no cover — optional
    HalfLifeHazardDistribution = None  # type: ignore


# -----------------------------------------------------------------------------
# Distribution registry
# -----------------------------------------------------------------------------
REGISTRY: Dict[str, Callable[..., BaseLifetime]] = {
    "weibull": lambda **kw: Weibull(**kw),
    "invgauss": lambda **kw: InverseGauss(**kw),
}
if HalfLifeHazardDistribution is not None:
    REGISTRY["hlh"] = lambda **kw: HalfLifeHazardDistribution(**kw)


# -----------------------------------------------------------------------------
# Example / smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # True model: Weibull(k=1.6, λ=10).  Generate a mix of: events,
    # right‑censor,
    # initial forward gaps, and empty windows.
    true = Weibull(shape=1.6, scale=10.0)

    # events
    dts = stats.weibull_min(c=true.shape, scale=true.scale).rvs(
        size=4000, random_state=rng
    )

    # right‑censor at t=8 (keep only those above 8 as survival observations)
    cens_thresh = 8.0
    survival = dts[dts > cens_thresh]

    # initial gaps (forward‑recurrence): sample gaps as independent draws too
    gaps = stats.weibull_min(c=true.shape, scale=true.scale).rvs(
        size=1000, random_state=rng
    )

    # empty windows: lengths between 8 and 15
    Ws = rng.uniform(8.0, 15.0, size=1000)

    # Fit model with small random initialisation
    init = Weibull(shape=np.exp(rng.normal(np.log(1.0), 0.2)),
                   scale=np.exp(rng.normal(np.log(9.0), 0.2)))

    init.fit(
        data=dts,
        survival_data=survival,
        initial_gaps=gaps,
        empty_windows=Ws,
        method="L-BFGS-B",
    )

    print("Fitted Weibull:", init.params)
    print("Mean (μ):", init.mean())
    # Example of the extra helpers
    print("tail_survival(W=12):", init.tail_survival(12.0))
    print("E[h|T≥12]:", init.expected_hazard_ge(12.0))
