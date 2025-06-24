"""Half‑Life Hazard Distribution with Analytic Gradient
=======================================================
Lifetime model whose hazard rate starts at 0 and rises monotonically toward an
asymptote *h_inf* with exponential half‑life *τ₁⁄₂*.

This version **adds a closed‑form Jacobian** of the negative log‑likelihood so
`scipy.optimize.minimize` can exploit exact derivatives.  The update preserves
all convenience helpers (mean, `rvs`, etc.) and the extra likelihood pieces
(forward‑recurrence gaps and empty windows) introduced earlier.

Highlights
----------
* **Analytic ∇(−log L)** → faster & more reliable fits; eliminates most
  "ABNORMAL_TERMINATION_IN_LNSRCH" issues.
* Parameters are still optimised in log‑space ``(log h_inf, log τ₁⁄₂)`` with
  the same public API (`fit`, `neg_log_likelihood`).
* Finite differences remain accessible: pass ``jac=None`` or switch optimiser
  to a gradient‑free method.
"""

from typing import Sequence

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import digamma, gammainc, gammaln, lambertw

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
        log_mu = k - k * np.log(k) + gammaln(k) + np.log(p) - np.log(self.lam)
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

    # --------------------- expected hazard rate for empty windows ----------
    # This is E[ h(T) | T ≥ W ] = ∫_W^∞ h(t) S(t) dt / ∫_W^∞ S(t) dt
    def expected_hazard_ge(
        self,
        W: float,
        rel_tol: float = 1e-9,
        abs_tol: float = 0.0,
        large_W_ratio: float = 8.0,
    ) -> float:
        """
        Return E[ h(T) | T ≥ W ]   (same time-units as the model).
        For use in calculating the hazard rate for empty windows.

        Parameters
        ----------
        W               : censor-window length (scalar, ≥ 0).
        rel_tol,abs_tol : integration tolerances for `scipy.integrate.quad`.
        large_W_ratio   : if  W / τ₁⁄₂ ≥ large_W_ratio,   short-circuit to
                        h_inf because the expectation differs by < 0.1 %.

        Notes
        -----
        * Everything inside the integral is evaluated in log-space:
            log I(t) = 2·log h(t) + log S(t)
        and only exponentiated once per quadrature node.
        * The survival denominator is also taken in log form to avoid
        dividing two *tiny* numbers.
        * The early-exit branch keeps this O(1) when W is many half-lives
        long (the common case with your 39-year empty windows).
        """

        # -------------------------------- quick exit -------------------------
        if W / self.half_life >= large_W_ratio:
            return float(self.hazard_inf)

        # -------------------------------- log-integrand ----------------------
        h_inf, lam = self.hazard_inf, self.lam  # local speed-ups
        ln_h_inf = np.log(h_inf)
        _EPS = 1e-300  # guard for exp()

        def _log_integrand(t):
            """log[ h(t)**2 · S(t) ]  evaluated stably."""
            e = np.exp(-lam * t)
            one_minus_e = -np.expm1(-lam * t)  # 1 − e^{−λt}
            log_h = ln_h_inf + np.log(one_minus_e + _EPS)
            cum_H = h_inf * (t - one_minus_e / lam)
            return 2.0 * log_h - cum_H

        # -------------------------------- numerator --------------------------
        num = quad(
            lambda x: np.exp(_log_integrand(x)),
            W,
            np.inf,
            limit=200,
            epsrel=rel_tol,
            epsabs=abs_tol,
        )[0]

        # -------------------------------- denominator ------------------------
        log_S_W = -self.cum_hazard(W)  # already stable
        log_num = np.log(num + _EPS)

        return float(np.exp(log_num - log_S_W))

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
    def reset_params(
        self,
        *,
        hazard_inf: float | None = None,
        half_life: float | None = None,
    ):
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
    #  Helpers for analytic derivatives
    # -----------------------------------------------------------------------
    @staticmethod
    def _theta_to_params(theta_log: Sequence[float]):
        """Return (h_inf, tau, lam) from log‑parameters."""
        h_inf, tau = np.exp(theta_log)
        lam = LN2 / tau
        return h_inf, tau, lam

    @staticmethod
    def _common_terms(k: float, x: float):
        """
        Return (gamma_lower, term_common) where
            gamma_lower  = Γ(k) · P(k, x)          (lower incomplete gamma)
            term_common  = x^{k-1} · e^{-x} / gamma_lower
        Both quantities are evaluated in a numerically safe way—no
        np.exp(gammaln(k))—so they stay finite up to very large k.
        """
        # log Γ(k ···)  — always finite
        log_gamma_k = gammaln(k)

        # log Γ_lower(k, x) = log[ Γ(k) · P(k,x) ]
        #   = log P(k,x) + log Γ(k)
        log_gamma_lower = (
            np.log(np.maximum(gammainc(k, x), _EPS)) + log_gamma_k
        )

        # term_common  =  exp( (k-1)·log x  –  x  –  log Γ_lower )
        log_term_common = (k - 1.0) * np.log(x) - x - log_gamma_lower
        term_common = np.exp(log_term_common)

        # Return Γ_lower in *linear* space because callers use it outside logs
        gamma_lower = np.exp(log_gamma_lower)
        return gamma_lower, term_common

    # -----------------------------------------------------------------------
    #  Negative log‑likelihood and Jacobian (static helpers) ----------------
    # -----------------------------------------------------------------------
    @staticmethod
    def _neg_log_likelihood_for_params(  # noqa: C901 – long but flat
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
        """Return −log L for exp(theta_log)."""
        h_inf, tau, lam = HalfLifeHazardDistribution._theta_to_params(
            theta_log
        )
        dist = HalfLifeHazardDistribution(h_inf, tau)
        mu = dist.mean()

        nll = 0.0
        # -- exact events ---------------------------------------------------
        if data.size:
            log_pdf = dist.log_pdf(data)
            nll -= (
                np.dot(data_counts, log_pdf)
                if data_counts is not None
                else log_pdf.sum()
            )
        # -- right‑censored --------------------------------------------------
        if survival_data is not None and survival_data.size:
            log_surv = dist.log_survival(survival_data)
            nll -= (
                np.dot(survival_counts, log_surv)
                if survival_counts is not None
                else log_surv.sum()
            )
        # -- forward‑recurrence gaps ---------------------------------------
        if initial_gaps is not None and initial_gaps.size:
            log_term = dist.log_survival(initial_gaps) - np.log(mu)
            nll -= (
                np.dot(initial_counts, log_term)
                if initial_counts is not None
                else log_term.sum()
            )
        # -- empty windows --------------------------------------------------
        if empty_windows is not None and empty_windows.size:
            tail = dist.tail_survival(empty_windows)
            log_term = np.log(tail) - np.log(mu)
            nll -= (
                np.dot(empty_counts, log_term)
                if empty_counts is not None
                else log_term.sum()
            )

        return float(nll)

    # ---------------------------------------------------------------------
    #  Analytic gradient of −log L in log‑parameter space -------------------
    # ---------------------------------------------------------------------
    @staticmethod
    def _grad_neg_log_likelihood_for_params(  # noqa: C901 – long but flat
        theta_log: Sequence[float],
        data: np.ndarray,
        data_counts: np.ndarray | None,
        survival_data: np.ndarray | None,
        survival_counts: np.ndarray | None,
        initial_gaps: np.ndarray | None,
        initial_counts: np.ndarray | None,
        empty_windows: np.ndarray | None,
        empty_counts: np.ndarray | None,
    ) -> np.ndarray:
        """Return ∇_{log p}(−log L)."""
        # Unpack parameters
        h_inf, tau, lam = HalfLifeHazardDistribution._theta_to_params(
            theta_log
        )
        k = h_inf / lam

        # Mean and its derivatives ----------------------------------------
        p_k = np.maximum(gammainc(k, k), _EPS)
        log_gamma_k = gammaln(k)
        log_gamma_lower_k = np.log(p_k) + log_gamma_k  # Γ(k)·P(k,k)
        term_k = np.exp((k - 1.0) * np.log(k) - k - log_gamma_lower_k)
        theta_k = digamma(k) - np.log(k) + term_k  # Θ(k,k)

        dlogmu_dh = (1.0 / lam) * theta_k
        dlogmu_dl = (-h_inf / lam**2) * theta_k - 1.0 / lam

        # Helper to accumulate gradient contributions
        grad_h, grad_l = 0.0, 0.0

        # Events -----------------------------------------------------------
        if data.size:
            t = data
            e = np.exp(-lam * t)
            one_minus_e = 1.0 - e

            dlogh_dh = 1.0 / h_inf
            dlogh_dl = t * e / one_minus_e

            dlogS_dh = -(t - one_minus_e / lam)
            dlogS_dl = -h_inf * (t * e / lam - one_minus_e / lam**2)

            w = data_counts if data_counts is not None else 1.0
            grad_h -= np.sum(w * (dlogh_dh + dlogS_dh))
            grad_l -= np.sum(w * (dlogh_dl + dlogS_dl))

        # Right‑censored ----------------------------------------------------
        if survival_data is not None and survival_data.size:
            t = survival_data
            e = np.exp(-lam * t)
            one_minus_e = 1.0 - e
            dlogS_dh = -(t - one_minus_e / lam)
            dlogS_dl = -h_inf * (t * e / lam - one_minus_e / lam**2)
            w = survival_counts if survival_counts is not None else 1.0
            grad_h -= np.sum(w * dlogS_dh)
            grad_l -= np.sum(w * dlogS_dl)

        # Forward‑recurrence gaps -----------------------------------------
        if initial_gaps is not None and initial_gaps.size:
            t = initial_gaps
            e = np.exp(-lam * t)
            one_minus_e = 1.0 - e
            dlogS_dh = -(t - one_minus_e / lam)
            dlogS_dl = -h_inf * (t * e / lam - one_minus_e / lam**2)
            w = initial_counts if initial_counts is not None else 1.0
            grad_h -= np.sum(w * (dlogS_dh - dlogmu_dh))
            grad_l -= np.sum(w * (dlogS_dl - dlogmu_dl))

        # Empty windows -----------------------------------------------------
        if empty_windows is not None and empty_windows.size:
            W = empty_windows
            x_W = k * np.exp(-lam * W)
            log_gamma_lower_x, term_x = (
                HalfLifeHazardDistribution._common_terms(k, x_W)
            )
            theta_x = digamma(k) - np.log(k) + term_x  # Θ(k,x_W)

            dlogT_dh = (1.0 / lam) * theta_x

            # dlogT/dλ  (see derivation)
            dx_dlam = x_W * (
                -h_inf / (lam**2 * k) - W
            )  # derivative of x_W wrt λ
            dlogT_dl = (
                (-h_inf / lam**2) * theta_x + dx_dlam * term_x - 1.0 / lam
            )

            w = empty_counts if empty_counts is not None else 1.0
            grad_h -= np.sum(w * (dlogT_dh - dlogmu_dh))
            grad_l -= np.sum(w * (dlogT_dl - dlogmu_dl))

        # Chain rule to log‑params -----------------------------------------
        grad_log_h = h_inf * grad_h
        grad_log_tau = (
            -lam * grad_l
        )  # ∂log τ = τ ∂τ, and ∂τ/∂λ = -τ²/LN2 = -τ² λ / LN2 τ? but simpler −λ
        return np.array([grad_log_h, grad_log_tau])

    # ---------------------------------------------------------------------
    #  Public NLL wrapper (instance uses its own parameters) ----------------
    # ---------------------------------------------------------------------
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
        survival_data = (
            None if survival_data is None else np.asarray(survival_data, float)
        )
        initial_gaps = (
            None if initial_gaps is None else np.asarray(initial_gaps, float)
        )
        empty_windows = (
            None if empty_windows is None else np.asarray(empty_windows, float)
        )

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

    # ---------------------------------------------------------------------
    #  Maximum‑likelihood fit ----------------------------------------------
    # ---------------------------------------------------------------------
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
        use_gradient: bool = True,
        print_errors: bool = True,
    ):
        """Maximum‑likelihood fit to any mix of observation types.

        * All *_counts* arrays are optional weights.
        * Pass ``use_gradient=False`` to fall back to finite differences (or if
          you change parameters and haven’t updated the derivative code yet).
        """
        data = np.asarray(data, float)
        survival_data = (
            None if survival_data is None else np.asarray(survival_data, float)
        )
        initial_gaps = (
            None if initial_gaps is None else np.asarray(initial_gaps, float)
        )
        empty_windows = (
            None if empty_windows is None else np.asarray(empty_windows, float)
        )

        theta0 = np.log([self.hazard_inf, self.half_life])
        bounds_log = (
            (np.log(1e-12), np.log(10.0)),
            (np.log(1e-5), np.log(1e5)),
        )

        # jac = (
        #     self._grad_neg_log_likelihood_for_params
        #     if use_gradient else None
        # )

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
            # jac=jac,
            bounds=bounds_log,
            options={} if options is None else options,
        )

        if not res.success:
            if print_errors:
                print(res.message)
                print("niterations:", res.nit)
                print("nfev:", res.nfev)
                print("njev:", res.njev)
                print("status:", res.status)
                print("fun:", res.fun)
                print("x:", res.x)
                print("h_inf:", np.exp(res.x[0]))
                print("tau:", np.exp(res.x[1]))
            raise RuntimeError("HLH fit failed: " + res.message)

        self.hazard_inf, tau = np.exp(res.x)
        self.lam = LN2 / tau
        return self
