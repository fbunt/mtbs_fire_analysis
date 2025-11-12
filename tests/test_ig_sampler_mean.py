from __future__ import annotations

import math
import unittest

import numpy as np

from mtbs_fire_analysis.analysis.statistical_tests.defaults import IG_DEF


class IGSamplerMeanTest(unittest.TestCase):
    """Sanity-check: IG sample mean is statistically consistent with μ.

    We allow variability by comparing the observed mean against μ using
    the estimated standard error of the mean and require |z| < 4.
    """

    def test_inverse_gauss_mean_matches_mu(self):
        rng = np.random.default_rng(123)
        model = IG_DEF.build()
        # Large but finite sample; IG variance can be large for small lam
        n = 300_000
        x = model.rvs(size=n, rng=rng)
        x = np.asarray(x, float)
        m = float(np.mean(x))
        s2 = float(np.var(x, ddof=1))
        se = math.sqrt(s2 / n)
        z = abs(m - model.mu) / max(se, 1e-12)
        self.assertLess(
            z,
            4.0,
            msg=f"IG mean far from μ (z={z:.2f}, mean={m:.3f}, mu={model.mu})",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
