"""
Common lifetime distribution base class used by HLH and SciPy-backed models.

Provides:
- pdf/sf/mean (abstract)
- hazard, tail_survival (numeric default), expected_hazard_ge
- generic renewal-style neg_log_likelihood
- robust fit() with optional subclass bounds and soft penalties
- copy() via params reconstruction
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate, optimize

_EPS = 1e-300
_QUAD_OPTS = {"epsrel": 1e-9, "epsabs": 0.0, "limit": 200}


class BaseLifetime:
    """Abstract lifetime model interface (shared by HLH and SciPy-backed)."""

    # --- primitives to implement ----------------------------------------
    def pdf(self, x):  # shape like numpy ufuncs
        raise NotImplementedError

    def sf(self, x):  # survival function
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    # --- provided helpers -------------------------------------------------
    def hazard(self, x):
        s = np.maximum(self.sf(x), _EPS)
        return self.pdf(x) / s

    def tail_survival(self, w):
        """Numeric default: ∫_w^∞ S(z) dz."""
        w_arr = np.atleast_1d(np.asarray(w, float))
        out = np.empty_like(w_arr)
        for i, w0 in enumerate(w_arr):
            out[i] = integrate.quad(
                lambda z: float(max(self.sf(z), 0.0)),
                float(max(w0, 0.0)),
                np.inf,
                **_QUAD_OPTS,
            )[0]
        return out if np.ndim(w) else float(out[0])

    def expected_hazard_ge(self, w: float) -> float:
        """E[h(T) | T ≥ w] using S(w)/∫_w^∞ S."""
        tail = self.tail_survival(w)
        num = np.log(max(float(self.sf(w)), _EPS))
        den = np.log(max(float(tail), _EPS))
        return float(np.exp(num - den))

    # --- generic negative log-likelihood ---------------------------------
    def neg_log_likelihood(
        self,
        data: Optional[np.ndarray] = None,
        data_counts: Optional[np.ndarray] = None,
        survival_data: Optional[np.ndarray] = None,
        survival_counts: Optional[np.ndarray] = None,
        initial_gaps: Optional[np.ndarray] = None,
        initial_counts: Optional[np.ndarray] = None,
        empty_windows: Optional[np.ndarray] = None,
        empty_counts: Optional[np.ndarray] = None,
    ) -> float:
        nll = 0.0
        mu = float(self.mean())
        log_mu = np.log(max(mu, _EPS))

        if data is not None and len(data):
            x = np.asarray(data, float)
            wts = (
                np.ones_like(x) if data_counts is None else np.asarray(data_counts, float)
            )
            p = self.pdf(x)
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            nll -= np.sum(wts * np.log(np.maximum(p, _EPS)))

        if survival_data is not None and len(survival_data):
            x = np.asarray(survival_data, float)
            wts = (
                np.ones_like(x)
                if survival_counts is None
                else np.asarray(survival_counts, float)
            )
            s = self.sf(x)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            nll -= np.sum(wts * np.log(np.maximum(s, _EPS)))

        if initial_gaps is not None and len(initial_gaps):
            g = np.asarray(initial_gaps, float)
            wts = (
                np.ones_like(g)
                if initial_counts is None
                else np.asarray(initial_counts, float)
            )
            s = self.sf(g)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            nll -= np.sum(wts * (np.log(np.maximum(s, _EPS)) - log_mu))

        if empty_windows is not None and len(empty_windows):
            wlen = np.asarray(empty_windows, float)
            wts = (
                np.ones_like(wlen)
                if empty_counts is None
                else np.asarray(empty_counts, float)
            )
            tails = self.tail_survival(wlen)
            tails = np.nan_to_num(tails, nan=0.0, posinf=0.0, neginf=0.0)
            nll -= np.sum(wts * (np.log(np.maximum(tails, _EPS)) - log_mu))

        return float(nll)

    # --- generic fitter with bounds/penalty hooks ------------------------
    def fit(
        self,
        data: Optional[np.ndarray] = None,
        data_counts: Optional[np.ndarray] = None,
        survival_data: Optional[np.ndarray] = None,
        survival_counts: Optional[np.ndarray] = None,
        initial_gaps: Optional[np.ndarray] = None,
        initial_counts: Optional[np.ndarray] = None,
        empty_windows: Optional[np.ndarray] = None,
        empty_counts: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        method: str = "L-BFGS-B",
        verbose: bool = False,
        **kwargs,
    ) -> "BaseLifetime":
        _ = kwargs
        theta0 = self._theta_get() if x0 is None else np.asarray(x0, float)
        if bounds is None and hasattr(self, "default_bounds"):
            try:
                bounds = self.default_bounds()  # type: ignore[attr-defined]
            except Exception:
                bounds = None

        def objective(theta: np.ndarray) -> float:
            self._theta_set(theta)
            nll = self.neg_log_likelihood(
                data=data,
                data_counts=data_counts,
                survival_data=survival_data,
                survival_counts=survival_counts,
                initial_gaps=initial_gaps,
                initial_counts=initial_counts,
                empty_windows=empty_windows,
                empty_counts=empty_counts,
            )
            penalty = 0.0
            if hasattr(self, "_soft_penalty"):
                try:
                    penalty = float(self._soft_penalty())  # type: ignore[attr-defined]
                except Exception:
                    penalty = 0.0
            if not np.isfinite(nll):
                nll = 1e12
            total = nll + penalty
            if not np.isfinite(total):
                total = 1e12
            return float(total)

        res = optimize.minimize(
            objective,
            x0=np.asarray(theta0, float),
            method=method,
            bounds=bounds,
        )
        if verbose:
            print(res)
        if not res.success:
            self._theta_set(res.x)
            raise RuntimeError("Fit failed: " + res.message)
        self._theta_set(res.x)
        return self

    # --- parametrisation hooks ------------------------------------------
    def _theta_get(self) -> np.ndarray:
        raise NotImplementedError

    def _theta_set(self, theta: np.ndarray) -> None:
        raise NotImplementedError

    @property
    def params(self) -> Dict[str, float]:
        raise NotImplementedError

    # --- cloning ----------------------------------------------------------
    def copy(self) -> "BaseLifetime":
        try:
            return type(self)(**self.params)
        except Exception:
            import copy as _copy
            return _copy.deepcopy(self)
