from __future__ import annotations

import unittest

from mtbs_fire_analysis.analysis.hlh_dist import (
    HalfLifeHazardDistribution,
)
from mtbs_fire_analysis.analysis.scipy_dist import InverseGauss, Weibull
from mtbs_fire_analysis.analysis.statistical_tests.repeat_helpers import (
    run_repeat_simulation,
)


def _rel_close(a: float, b: float, tol: float = 1e-3) -> bool:
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom <= tol


class RepeatMeanVsNaiveTest(unittest.TestCase):
    """Ensure dtctutet mean stats align with naive within 0.1%.

    We use moderately large samples to reduce Monte Carlo noise while
    keeping runtime reasonable.
    """

    NUM_PIXELS = 5000
    TIME_INTERVAL = 39
    ITERATIONS = 100
    PRE_WINDOW = 500
    SEED = 1989

    def _check_dist(self, truth_model):
        out, _ = run_repeat_simulation(
            model_ctor=type(truth_model),
            truth=truth_model,
            num_pixels=self.NUM_PIXELS,
            time_interval=self.TIME_INTERVAL,
            iterations=self.ITERATIONS,
            properties=("mean",),
            prop_args={},
            pre_window=self.PRE_WINDOW,
            random_seed=self.SEED,
            modes=("dtctutet", "naive"),
        )

        # Pull the three stats for dtctutet and naive
        row_dt = out.loc[out["name"] == "dtctutet"].iloc[0]
        row_nv = out.loc[out["name"] == "naive"].iloc[0]

        for col in ("mean_10pc", "mean_mean", "mean_90pc"):
            self.assertTrue(
                _rel_close(float(row_dt[col]), float(row_nv[col]), 1e-3),
                msg=f"{type(truth_model).__name__}: {col} differs:"
                f" dtctutet={row_dt[col]} vs naive={row_nv[col]}",
            )

    def test_hlh(self):
        truth = HalfLifeHazardDistribution(0.03, 50)
        self._check_dist(truth)

    def test_weibull(self):
        # Snapshot of current defaults: shape=1.5, scale=85.0
        truth = Weibull(shape=1.5, scale=85.0)
        self._check_dist(truth)

    def test_inverse_gauss(self):
        # Snapshot of current defaults: mu=75.0, lam=200.0
        truth = InverseGauss(mu=75.0, lam=200.0)
        self._check_dist(truth)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
