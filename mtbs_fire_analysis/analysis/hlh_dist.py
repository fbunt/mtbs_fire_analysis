from __future__ import annotations

"""Half‑Life Hazard Distribution
===============================
Lifetime model whose hazard rate starts at 0 and rises monotonically toward an
asymptote *h_inf* with exponential half‑life *τ₁⁄₂*.

Add‑ons compared with the original prototype
--------------------------------------------
* Closed‑form **tail survival integral** → fast empty‑window likelihood.
* Forward‑recurrence and empty‑window likelihood terms (optional).
* Convenience methods/properties from the first iteration brought back:
  ``dist_type``, ``half_life`` property, ``params`` dict, ``log_pdf``/``log_cdf``/
  ``log_survival``, ``sf`` alias, ``rvs`` generator, and ``reset_params``.

Gradient code has been dropped – optimisation relies on numerical derivatives.
"""

from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammainc, gammaln, lambertw

LN2 = np.log(2.0)
_EPS = 1e-12  # numeric safety guard for logs / divisions


# ---------------------------------------------------------------------------
#  Distribution definition
# ---------------------------------------------------------------------------
class HalfLifeHazardDistribution:
    """Lifetime model with asymptoting hazard (see module docstring)."""

    # ------------------------- construction --------------------------------
    def __init__(self, hazard_inf: float = 0.1, half_life: float = 10.0):
        if hazard_inf <= 0 or half_life <= 0:
            raise ValueError("hazard_inf and half_life must be positive.")
        self.hazard_inf = float(hazard_inf)
        self.lam = LN2 / float(half_life)  # λ = ln2 / τ½

    # ----------------------- metadata / convenience ------------------------
    @property
    def dist_type(self) -> str:  # for generic plotting / registry code
        return "HalfLifeHazard"

    @property
    def half_life(self) -> float:
        """Return *τ₁⁄₂* = ln 2 / λ."""
        return LN2 / self.lam

    @property
    def params(self):
        """Dictionary of the current parameters."""
        return {"hazard_inf": self.hazard_inf, "half_life": self.half_life}

    # --------------------- elementary functions ----------------------------
    def hazard(self, t):
        """Instantaneous hazard *h(t)*."""
        t = np.asarray(t, float)
        return self.hazard_inf * (-np.expm1(-self.lam * t))  # 1 − e^{−λt}

    def cum_hazard(self, t):
        """Cumulative hazard *H(t)* = ∫₀ᵗ h(z) dz."""
        t = np.asarray(t, float)
        e = np.exp(-self.lam * t)
        return self.hazard_inf * (t - (1.0 - e) / self.lam)

    def survival(self, t):
        """Survival function *S(t)* = exp[−H(t)]."""
        return np.exp(-self.cum_hazard(t))

    # aliases / logs --------------------------------------------------------
    sf = survival  # common name in scipy.stats

    def pdf(self, t):
        """Density *f(t)*."""
        return self.hazard(t) * self.survival(t)

    def log_pdf(self, t):
        return np.log(self.pdf(t))

    def cdf(self, t):
        return 1.0 - self.survival(t)

    def log_cdf(self, t):
        return np.log(self.cdf(t))

    def log_survival(self, t):
        return np.log(self.survival(t))

    # --------------------- first moment ------------------------------------
    def mean(self):
        """μ = E[T] (finite for all positive parameters, evaluated stably)."""
        k = self.hazard_inf / self.lam
        p = gammainc(k, k)  # regularised lower incomplete gamma P(k,k)
        log_mu = (
            k - k * np.log(k) + gammaln(k) + np.log(p) - np.log(self.lam)
        )
        return np.exp(log_mu)

    # --------------------- tail survival integral --------------------------
    def tail_survival(self, W):
        """∫_W^∞ S(z) dz (vectorised closed form)."""
        W = np.asarray(W, float)
        k = self.hazard_inf / self.lam
        x = k * np.exp(-self.lam * W)
        log_prefac = k - np.log(self.lam) - k * np.log(k)
        log_gamma_lower = np.log(np.maximum(gammainc(k, x), _EPS)) + gammaln(k)
        return np.exp(log_prefac + log_gamma_lower)

    # ----------------------- random variates ------------------------------
    def rvs(self, size=1, rng=None):
        """Generate random variates via inverse‑transform (vectorised)."""
        rng = np.random.default_rng() if rng is None else rng
        u = rng.random(size)
        y = -np.log1p(-u) / self.hazard_inf  # exponential with rate h_inf
        c = 1.0 + self.lam * y
        z = c + lambertw(-np.exp(-c), k=0).real  # principal branch
        return z / self.lam

    # ------------------------ parameter reset ------------------------------
    def reset_params(self, *, hazard_inf: float | None = None, half_life: float | None = None):
        """Reset parameters in‑place (handy for manual tuning / warm starts)."""
        if hazard_inf is not None:
            if hazard_inf <= 0:
                raise ValueError("hazard_inf must be positive.")
            self.hazard_inf = float(hazard_inf)
        if half_life is not None:
            if half_life <= 0:
                raise ValueError("half_life must be positive.")
            self.lam = LN2 / float(half_life)

    # -----------------------------------------------------------------------
    #  Negative log‑likelihood (log‑parametrised)
    # -----------------------------------------------------------------------
    @staticmethod
    def _neg_log_likelihood_for_params(
        theta_log: Sequence[float],
        data: np.ndarray,
        data_counts: np.ndarray | None,
        survival_data: np.ndarray | None,
        survival_counts: np.ndarray | None,
        initial_gaps: np.ndarray | None,
        initial_counts: np.ndarray | None,
        empty_windows: np.ndarray | None,
        empty_counts: np.ndarray | None,
    ) -> float:
        """Helper that builds a distribution with exp(theta_log) and returns −log L."""
        h_inf, tau = np.exp(theta_log)
        dist = HalfLifeHazardDistribution(h_inf, tau)
        mu = dist.mean()

        nll = 0.0
        # -- exact events ---------------------------------------------------
        if data.size:
            log_pdf = dist.log_pdf(data)
            nll -= np.dot(data_counts, log_pdf) if data_counts is not None else log_pdf.sum()
        # -- right‑censored --------------------------------------------------
        if survival_data is not None and survival_data.size:
            log_surv = dist.log_survival(survival_data)
            nll -= np.dot(survival_counts, log_surv) if survival_counts is not None else log_surv.sum()
        # -- forward‑recurrence gaps ---------------------------------------
        if initial_gaps is not None and initial_gaps.size:
            log_term = dist.log_survival(initial_gaps) - np.log(mu)
            nll -= np.dot(initial_counts, log_term) if initial_counts is not None else log_term.sum()
        # -- empty windows --------------------------------------------------
        if empty_windows is not None and empty_windows.size:
            tail = dist.tail_survival(empty_windows)
            log_term = np.log(tail) - np.log(mu)
            nll -= np.dot(empty_counts, log_term) if empty_counts is not None else log_term.sum()

        return float(nll)

    # -----------------------------------------------------------------------
    #  Public NLL wrapper (instance uses its own parameters) -----------------
    # -----------------------------------------------------------------------
    def neg_log_likelihood(
        self,
        data,
        data_counts=None,
        survival_data=None,
        survival_counts=None,
        initial_gaps=None,
        initial_counts=None,
        empty_windows=None,
        empty_counts=None,
    ) -> float:
        """Compute −log L for the provided observations."""
        data = np.asarray(data, float)
        survival_data = None if survival_data is None else np.asarray(survival_data, float)
        initial_gaps = None if initial_gaps is None else np.asarray(initial_gaps, float)
        empty_windows = None if empty_windows is None else np.asarray(empty_windows, float)

        return self._neg_log_likelihood_for_params(
            np.log([self.hazard_inf, self.half_life]),
            data,
            data_counts,
            survival_data,
            survival_counts,
            initial_gaps,
            initial_counts,
            empty_windows,
            empty_counts,
        )

    # -----------------------------------------------------------------------
    #  Maximum‑likelihood fit ----------------------------------------------
    # -----------------------------------------------------------------------
    def fit(
        self,
        data,
        data_counts=None,
        survival_data=None,
        survival_counts=None,
        initial_gaps=None,
        initial_counts=None,
        empty_windows=None,
        empty_counts=None,
        method: str = "L-BFGS-B",
        options: dict | None = None,
    ):
        """Maximum‑likelihood fit to any mix of observation types.

        All *_counts* vectors are optional weights.  The optimiser uses finite‑
        differences; switch *method* to a gradient‑free routine (e.g. ‘Nelder‑
        Mead’) if preferred.
        """
        data = np.asarray(data, float)
        survival_data = None if survival_data is None else np.asarray(survival_data, float)
        initial_gaps = None if initial_gaps is None else np.asarray(initial_gaps, float)
        empty_windows = None if empty_windows is None else np.asarray(empty_windows, float)

        theta0 = np.log([self.hazard_inf, self.half_life])
        bounds_log = ((np.log(1e-12), np.log(4.0)), (np.log(1e-3), np.log(1e3)))

        res = minimize(
            self._neg_log_likelihood_for_params,
            theta0,
            args=(
                data,
                data_counts,
                survival_data,
                survival_counts,
                initial_gaps,
                initial_counts,
                empty_windows,
                empty_counts,
            ),
            method=method,
            bounds=bounds_log,
            options={} if options is None else options,
        )

        if not res.success:
            raise RuntimeError("HLH fit failed: " + res.message)

        self.hazard_inf, tau = np.exp(res.x)
        self.lam = LN2 / tau
        return self
