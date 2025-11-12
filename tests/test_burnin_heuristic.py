from __future__ import annotations

import math
import unittest

from mtbs_fire_analysis.analysis.lifetime_base import BaseLifetime
from mtbs_fire_analysis.analysis.statistical_tests.defaults import (
    HLH_DEF,
    IG_DEF,
    SIM,
    WEIBULL_DEF,
)


def _p999_generic(model: BaseLifetime, max_iter: int = 80) -> float:
    """Return an estimate of the 99.9th percentile for any model.

    Uses SciPy ppf when available (SciPy-backed models). Otherwise falls back
    to a monotone bisection on the survival function S(t) until S(t) ≈ 0.001.
    """
    # SciPy-backed models expose a frozen rv via _rv; prefer that path.
    rv = getattr(model, "_rv", None)
    if rv is not None and hasattr(rv, "ppf"):
        try:
            val = float(rv.ppf(0.999))  # type: ignore[attr-defined]
            if math.isfinite(val) and val > 0:
                return val
        except Exception:
            pass

    # Fallback: numeric invert S(t) = 0.001 via bisection
    target_surv = 1.0 - 0.999  # 0.001
    # Bracket: start around mean, expand multiplicatively
    m = float(model.mean())
    lo, hi = 0.0, max(10.0, 10.0 * m)
    # expand hi until S(hi) <= target or cap
    for _ in range(50):
        s_hi = float(model.sf(hi))
        if s_hi <= target_surv:
            break
        hi *= 2.0
        if hi > 1e9:  # guard
            break
    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s_mid = float(model.sf(mid))
        if s_mid > target_surv:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


class BurnInHeuristicTest(unittest.TestCase):
    """Check that default burn-in is within 1–2× max(5·mean, p99.9).

    This is a cautious heuristic; if it fails, we should revisit defaults
    or per-distribution burn-in suggestions.
    """

    def _check_model(self, model: BaseLifetime):
        mean = float(model.mean())
        p999 = _p999_generic(model)
        target = max(5.0 * mean, p999)
        burn_in = float(SIM.pre_window)

        # Must be between 1× and 2× target
        self.assertGreaterEqual(
            burn_in, target, msg=f"burn-in {burn_in} < target {target}"
        )
        self.assertLessEqual(
            burn_in,
            2.0 * target,
            msg=f"burn-in {burn_in} > 2× target {target}",
        )

    def test_hlh_default_burnin(self):
        self._check_model(HLH_DEF.build())

    def test_weibull_default_burnin(self):
        self._check_model(WEIBULL_DEF.build())

    def test_inverse_gauss_default_burnin(self):
        self._check_model(IG_DEF.build())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
