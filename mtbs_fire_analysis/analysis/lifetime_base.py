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

from .fit_constraints import (
    POLICY,
    get_bounds_log_for,
    get_soft_box_for,
)

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

    # --- lightweight health check (cheap) -------------------------------
    def quick_health(self) -> dict:
        """Cheap post-fit health signal.

        Criteria (no data re-evaluation):
        - params dict exists and all numeric, finite
        - mean finite and positive
        - theta not sitting exactly on provided/default bounds
        Returns dict with 'ok' bool and 'issues' list.
        """
        issues: list[str] = []
        params = getattr(self, "params", None)
        if not isinstance(params, dict) or not params:
            issues.append("no_params")
        else:
            for k, v in params.items():
                try:
                    val = float(v)
                except Exception:
                    issues.append(f"non_numeric:{k}")
                    continue
                if not np.isfinite(val):
                    issues.append(f"non_finite:{k}")
                if val == 0.0:
                    issues.append(f"zero_value:{k}")
        # mean
        try:
            m = float(self.mean())
            if (not np.isfinite(m)) or (m <= 0.0):
                issues.append("mean_invalid")
        except Exception:
            issues.append("mean_error")
        # bounds proximity (exactly at lo or hi) + near-bounds policy
        theta = np.asarray(self._theta_get(), float)
        bounds = get_bounds_log_for(type(self).__name__)
        if len(bounds) != len(theta):
            raise RuntimeError("Bounds dimensionality does not match theta")
        frac_margin = float(POLICY.get("hard_bounds_margin_fraction", 0.02))
        for t, (lo, hi) in zip(theta, bounds, strict=True):
            if np.isfinite(lo) and t <= lo:
                issues.append("at_lower_bound")
            if np.isfinite(hi) and t >= hi:
                issues.append("at_upper_bound")
            if POLICY.get("fail_if_near_hard_bounds", True):
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    width = hi - lo
                    dist_lo = (t - lo) / width
                    dist_hi = (hi - t) / width
                    if dist_lo <= frac_margin:
                        issues.append("near_lower_bound")
                    if dist_hi <= frac_margin:
                        issues.append("near_upper_bound")

        # soft-box proximity via central constraints
        soft_box = get_soft_box_for(type(self).__name__)
        # Gather params by name from either self.params or attributes
        params: Dict[str, float] = {}
        param_src: Dict[str, float] = dict(getattr(self, "params"))  # type: ignore[arg-type]
        for name in soft_box.keys():
            if name in param_src:
                params[name] = float(param_src[name])
            elif hasattr(self, name):
                params[name] = float(getattr(self, name))
            else:
                raise KeyError(f"Missing parameter '{name}' on model")
        from .fit_constraints import soft_box_distance  # local import to avoid cycles

        inside, min_margin = soft_box_distance(params, soft_box)
        if POLICY.get("fail_if_outside_soft_box", True) and not inside:
            issues.append("outside_soft_box")
        if POLICY.get("fail_if_near_soft_edge", True) and inside:
            edge_frac = float(POLICY.get("soft_edge_margin_fraction", 0.05))
            if min_margin <= edge_frac:
                issues.append("near_soft_edge")
        return {"ok": len(issues) == 0, "issues": issues}

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
                np.ones_like(x)
                if data_counts is None
                else np.asarray(data_counts, float)
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
        # Always use centralised bounds; if missing, raise KeyError
        bounds = (
            get_bounds_log_for(type(self).__name__)
            if bounds is None
            else bounds
        )

        # Compute an effective sample size for optional penalty scaling
        def _sum_counts(w: Optional[np.ndarray]) -> float:
            if w is None:
                return 0.0
            return float(np.sum(np.asarray(w, float)))

        # Effective N is the SUM OF COUNTS only (ignore raw lengths)
        n_eff = 0.0
        n_eff += _sum_counts(data_counts)
        n_eff += _sum_counts(survival_counts)
        n_eff += _sum_counts(initial_counts)
        n_eff += _sum_counts(empty_counts)

        def objective(theta: np.ndarray) -> float:
            self._theta_set(theta)
            # Expose effective N to subclass penalty if needed
            try:
                setattr(self, "_fit_n_eff", float(n_eff))
            except Exception:
                pass
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
        # Post-fit health gate (optional failure)
        fail_on_unhealthy = bool(
            kwargs.get(
                "fail_on_unhealthy",
                POLICY.get("fail_on_unhealthy_default", False),
            )
        )
        if fail_on_unhealthy:
            health = self.quick_health()
            if not health.get("ok", False):
                raise RuntimeError(
                    "Unhealthy fit: " + ",".join(health.get("issues", []))
                )
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
