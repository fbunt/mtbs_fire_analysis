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

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate, stats

from .default_params import IG_DEFAULTS, WEIBULL_DEFAULTS
from .fit_constraints import (
    get_bounds_log_for,
    get_penalty_strength_for,
    get_soft_box_for,
    penalty_scale_factor,
)
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
        # Guard SciPy warnings from extreme interim parameter proposals
        with np.errstate(
            over="ignore",
            under="ignore",
            invalid="ignore",
            divide="ignore",
        ):
            return self._rv.pdf(x)  # type: ignore[arg-type]

    def sf(self, x):
        with np.errstate(
            over="ignore",
            under="ignore",
            invalid="ignore",
            divide="ignore",
        ):
            return self._rv.sf(x)  # type: ignore[arg-type]

    def mean(self) -> float:
        with np.errstate(
            over="ignore",
            under="ignore",
            invalid="ignore",
            divide="ignore",
        ):
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
    def __init__(
        self,
        shape: float = WEIBULL_DEFAULTS["shape"],
        scale: float = WEIBULL_DEFAULTS["scale"],
    ) -> None:
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

    # Closed-form tail survival integral:
    #   ∫_W^∞ S(t) dt = (λ / k) Γ(1/k, (W/λ)^k)
    def tail_survival(self, w: np.ndarray | float) -> np.ndarray | float:  # type: ignore[override]
        w_arr = np.atleast_1d(np.asarray(w, float))
        k, lam = float(self.shape), float(self.scale)
        # Guard invalid parameters to avoid crashes inside optimizer
        # explorations
        if (
            (not np.isfinite(k))
            or (not np.isfinite(lam))
            or (k <= 0.0)
            or (lam <= 0.0)
        ):
            bad = np.full_like(w_arr, np.inf, dtype=float)
            return bad if np.ndim(w) else float(bad[0])
        a = 1.0 / k
        x = np.maximum(w_arr / lam, 0.0)
        # Compute u = x**k safely: if k*log(x) > log(max_float), set u=inf
        log_max = np.log(np.finfo(float).max)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            logu = k * np.log(np.maximum(x, _EPS))
            u = np.where(logu > log_max, np.inf, np.exp(logu))
            tail = (lam / k) * gamma(a) * gammaincc(a, u)
        return tail if np.ndim(w) else float(tail[0])

    def mean(self) -> float:
        """Closed-form Weibull mean: μ = λ Γ(1 + 1/k)."""
        k, lam = float(self.shape), float(self.scale)
        if (
            (not np.isfinite(k))
            or (not np.isfinite(lam))
            or (k <= 0.0)
            or (lam <= 0.0)
        ):
            return np.inf
        val = lam * gamma(1.0 + 1.0 / k)
        return float(val) if np.isfinite(val) else np.inf

    # Reasonable, wide bounds in log-space to keep optimizer from k→0 or
    # extreme λ
    def default_bounds(self) -> Sequence[Tuple[float, float]]:
        bounds = get_bounds_log_for(type(self).__name__)
        return list(bounds)

    # Soft penalty to keep search in a practical region; smooth quadratic
    # outside box
    def _soft_penalty(self) -> float:
        k, lam = float(self.shape), float(self.scale)
        # Heavy penalty if invalid to steer optimizer away quickly
        if (
            (not np.isfinite(k))
            or (not np.isfinite(lam))
            or (k <= 0.0)
            or (lam <= 0.0)
        ):
            return 1e9
        sb = get_soft_box_for(type(self).__name__)
        k_lo, k_hi = sb.get("shape", (0.3, 10.0))
        lam_lo, lam_hi = sb.get("scale", (1.0, 200.0))
        strength = get_penalty_strength_for(type(self).__name__) or 1e3
        # Penalty in LOG space: use distances in s = ln(param) normalised by
        # the soft-box log-width so multiplicative deviations are symmetric.
        pen = 0.0
        s_k, s_lam = np.log(k), np.log(lam)
        s_k_lo, s_k_hi = np.log(k_lo), np.log(k_hi)
        s_lam_lo, s_lam_hi = np.log(lam_lo), np.log(lam_hi)
        w_k = max(s_k_hi - s_k_lo, 1e-12)
        w_lam = max(s_lam_hi - s_lam_lo, 1e-12)
        if s_k < s_k_lo:
            d = (s_k_lo - s_k) / w_k
            pen += d * d
        elif s_k > s_k_hi:
            d = (s_k - s_k_hi) / w_k
            pen += d * d
        if s_lam < s_lam_lo:
            d = (s_lam_lo - s_lam) / w_lam
            pen += d * d
        elif s_lam > s_lam_hi:
            d = (s_lam - s_lam_hi) / w_lam
            pen += d * d
        n_eff = getattr(self, "_fit_n_eff", 0.0)
        return float(strength * penalty_scale_factor(n_eff) * pen)


# -----------------------------------------------------------------------------
# Inverse Gaussian (Wald) using SciPy's invgauss; expose IG(mu, lam) params
# Mapping to SciPy: if Y ~ invgauss(mu_s), X = scale * Y has
#   E[X] = mu_s * scale, Var[X] = mu_s^3 * scale^2.
# To match IG(mean=mu, shape=lam), set mu_s = mu/lam, scale = lam.
# -----------------------------------------------------------------------------
class InverseGauss(SciPyParametric):
    def __init__(
        self, mu: float = IG_DEFAULTS["mu"], lam: float = IG_DEFAULTS["lam"]
    ) -> None:
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

    def tail_survival(self, w: np.ndarray | float) -> np.ndarray | float:  # type: ignore[override]
        """Numeric tail integral with guards for tiny survival values.

        Avoids quad warnings for extremely small S(W) by short-circuiting
        to 0.0 when S(W) is below a tiny threshold. Uses a slightly
        looser tolerance than the BaseLifetime default to improve
        robustness for this distribution.
        """
        w_arr = np.atleast_1d(np.asarray(w, float))
        out = np.empty_like(w_arr, dtype=float)
        for i, w0 in enumerate(w_arr):
            with np.errstate(
                over="ignore",
                under="ignore",
                invalid="ignore",
                divide="ignore",
            ):
                s_w = float(self.sf(w0))
            if (not np.isfinite(s_w)) or (s_w < 1e-14):
                out[i] = 0.0
                continue
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", category=integrate.IntegrationWarning
                )
                val = integrate.quad(
                    lambda z: float(max(self.sf(z), 0.0)),
                    float(max(w0, 0.0)),
                    np.inf,
                    epsrel=1e-6,
                    epsabs=0.0,
                    limit=200,
                )[0]
            out[i] = float(val)
        return out if np.ndim(w) else float(out[0])

    def default_bounds(self) -> Sequence[Tuple[float, float]]:
        bounds = get_bounds_log_for(type(self).__name__)
        return list(bounds)

    def _soft_penalty(self) -> float:
        mu, lam = float(self.mu), float(self.lam)
        if (
            (not np.isfinite(mu))
            or (not np.isfinite(lam))
            or (mu <= 0.0)
            or (lam <= 0.0)
        ):
            return 1e9
        sb = get_soft_box_for(type(self).__name__)
        mu_lo, mu_hi = sb.get("mu", (1e-2, 1e3))
        lam_lo, lam_hi = sb.get("lam", (1e-2, 1e4))
        strength = get_penalty_strength_for(type(self).__name__) or 1e3
        # Log-space penalty symmetrical in multiplicative deviations
        pen = 0.0
        s_mu, s_lam = np.log(mu), np.log(lam)
        s_mu_lo, s_mu_hi = np.log(mu_lo), np.log(mu_hi)
        s_lam_lo, s_lam_hi = np.log(lam_lo), np.log(lam_hi)
        w_mu = max(s_mu_hi - s_mu_lo, 1e-12)
        w_lam = max(s_lam_hi - s_lam_lo, 1e-12)
        if s_mu < s_mu_lo:
            d = (s_mu_lo - s_mu) / w_mu
            pen += d * d
        elif s_mu > s_mu_hi:
            d = (s_mu - s_mu_hi) / w_mu
            pen += d * d
        if s_lam < s_lam_lo:
            d = (s_lam_lo - s_lam) / w_lam
            pen += d * d
        elif s_lam > s_lam_hi:
            d = (s_lam - s_lam_hi) / w_lam
            pen += d * d
        n_eff = getattr(self, "_fit_n_eff", 0.0)
        return float(strength * penalty_scale_factor(n_eff) * pen)


# Note: Global registry was moved to mtbs_fire_analysis.analysis.registry


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
