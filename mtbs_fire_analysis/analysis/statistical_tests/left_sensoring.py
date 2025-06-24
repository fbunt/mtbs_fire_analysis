"""
This script generates synthetic data to test the effect of left censoring on
fitting a Half-Life Hazard Distribution (HLHD) model. It compares the fitted
parameters and negative log-likelihood values of the model when left censoring
is applied versus when it is not. The script also generates plots to visualize
the fitted distributions against the true distribution and the data.

At the time of writing, the results show that left censoring has no effect on
the fitted parameters or negative log-likelihood values, as the fitted
parameters are close to the truth in both cases.

At least for data similar to that generated, it seems we don't need to account
for left censoring in the fitting process.
"""

from pathlib import Path

import mtbs_fire_analysis.analysis.statistical_tests.test_helpers as th
from mtbs_fire_analysis.analysis.distributions import (
    HalfLifeHazardDistribution as HLHD,
)

num_pixels = 100_000
time_interval = 38

pre_window = 102

truth = HLHD(hazard_inf=0.09, half_life=13)

nlc_dts, nlc_sts = th.create_sample_data(
    num_pixels=num_pixels,
    time_interval=time_interval,
    truth=truth,
    left_censoring_time=0,
)

lc_dts, lc_sts = th.create_sample_data(
    num_pixels=num_pixels,
    time_interval=time_interval,
    truth=truth,
    left_censoring_time=pre_window,
)

nlc_fitter = HLHD(hazard_inf=0.15, half_life=20)
nlc_fitter.fit(nlc_dts, survival_data=nlc_sts)

lc_fitter = HLHD(hazard_inf=0.15, half_life=20)
lc_fitter.fit(lc_dts, survival_data=lc_sts)

th.evaluate_fits(
    lc_dts,
    lc_sts,
    [truth, nlc_fitter, lc_fitter],
    names=["Truth", "No Left Censoring", "Left Censoring"],
)

out_dir = (
    Path("mtbs_fire_analysis")
    / "analysis"
    / "statistical_tests"
    / "test_outputs"
    / "left_sensoring"
)
out_dir.mkdir(parents=False, exist_ok=True)

th.output_plots(
    lc_dts,
    lc_sts,
    [truth, nlc_fitter, lc_fitter],
    folder=out_dir,
    names=["Truth", "No Left Censoring", "Left Censoring"],
)
