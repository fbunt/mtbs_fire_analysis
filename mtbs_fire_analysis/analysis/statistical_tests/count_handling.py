"""
To do description
"""

from pathlib import Path

import numpy as np

import mtbs_fire_analysis.analysis.statistical_tests.test_helpers as th
from mtbs_fire_analysis.analysis.distributions import (
    HalfLifeHazardDistribution as HLHD,
)

num_pixels = 100_000
time_interval = 38

truth = HLHD(hazard_inf=0.09, half_life=13)

dts, sts = th.create_sample_data(
    num_pixels=num_pixels,
    time_interval=time_interval,
    truth=truth,
    left_censoring_time=0,
)

# Round the dts and sts to 1 decimal places for binning counts
dts = np.round(dts, 1)
sts = np.round(sts, 1)

dt_bins = np.arange(0, 38.1, 0.1)
st_bins = np.arange(0, 38.1, 0.1)
# Bin the dts and sts
dts_binned = np.histogram(dts, bins=dt_bins)[0]
sts_binned = np.histogram(sts, bins=st_bins)[0]
dt_locs = dt_bins[:-1] + 0.05
st_locs = st_bins[:-1] + 0.05

fitter = HLHD(hazard_inf=0.15, half_life=20)
fitter.fit(dts, survival_data=sts)
fitter_counts = HLHD(hazard_inf=0.15, half_life=20)
fitter_counts.fit(
    dt_locs, dts_binned, survival_data=st_locs, survival_counts=sts_binned
)

fits = [
    truth,
    fitter,
    fitter_counts,
]

names = [
    "Truth",
    "Normal Fit",
    "Binned Fit",
]

th.evaluate_fits(dts, sts, fits, names)

out_dir = (
    Path("mtbs_fire_analysis")
    / "analysis"
    / "statistical_tests"
    / "test_outputs"
    / "count_handling"
)
out_dir.mkdir(parents=False, exist_ok=True)

th.output_plots(dts, sts, fits, out_dir, names)
