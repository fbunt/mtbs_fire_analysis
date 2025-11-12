from __future__ import annotations

import math
import unittest

import numpy as np

from mtbs_fire_analysis.analysis.statistical_tests.defaults import IG_DEF


class TestInverseGaussRVMean(unittest.TestCase):
    """Sanity check that IG.rvs produces the expected mean.

    To keep variance manageable, use a large shape (lam) so that
    Var[X] = mu^3 / lam is small enough for a tight tolerance with
    moderate sample sizes.
    """

    def test_mean_close_to_mu(self):
        # Use project defaults for IG parameters
        model = IG_DEF.build()
        mu = float(model.mu)
        n = 100_000
        rng = np.random.default_rng(12345)

        samples = model.rvs(size=n, rng=rng)
        samples = np.asarray(samples, dtype=float)
        sample_mean = float(np.mean(samples))
        sample_var = float(np.var(samples, ddof=1))
        se = math.sqrt(sample_var / n)

        # Z-score based check: allow typical statistical fluctuation
        z = abs(sample_mean - mu) / max(se, 1e-12)
        self.assertLess(
            z,
            4.0,
            msg=(
                f"IG rvs mean far from mu: z={z:.2f}, "
                f"mean={sample_mean:.6f}, mu={mu:.6f}"
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
