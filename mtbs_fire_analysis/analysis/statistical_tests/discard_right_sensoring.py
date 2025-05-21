"""
This script compares the true distribution that generated the
data with fitted distributions:
1. ignoring right sensoring
2. accounting for right sensoring in full
3. accounting for right sensoring and discarding sensoring
    that occurs for the entire time interval

Results at the time of writing are that at least for the HLH
distribution, both ignoring right sensoring and accounting for it
with discarding cause the parameters to be far from the truth,
but sensoring with the full survival data gives parameters that are
close to the truth.

This is the basis for requiring the sensoring data for all pixels
even when they don't have any fire events in the interval.
"""

import mtbs_fire_analysis.analysis.statistical_tests.test_helpers as th
from mtbs_fire_analysis.analysis.distributions import (
    HalfLifeHazardDistribution as HLHD,
)
from pathlib import Path

num_pixels = 100_000
time_interval = 38

truth = HLHD(hazard_inf=0.09, half_life=13)

dts, sts = th.create_sample_data(
    num_pixels=num_pixels,
    time_interval=time_interval,
    truth=truth,
    left_censoring_time=0,
)

fitter = HLHD(hazard_inf=0.15, half_life=20)
fitter.fit(dts)
fitter_sensor = HLHD(hazard_inf=0.15, half_life=20)
fitter_sensor.fit(dts, survival_data=sts)
fitter_sensor_discard = HLHD(hazard_inf=0.15, half_life=20)
fitter_sensor_discard.fit(dts, survival_data=sts[sts < 37.5])

fits = [
    truth,
    fitter,
    fitter_sensor,
    fitter_sensor_discard,
]

names = [
    "Truth",
    "No Sensoring",
    "Right Sensoring",
    "Discarded Right Sensoring",
]

th.evaluate_fits(
    dts,
    sts,
    fits,
    names
)

out_dir = (Path("mtbs_fire_analysis")
           / "analysis"
           / "statistical_tests"
           / "test_outputs"
           / "discarded_right_sensoring"
)
out_dir.mkdir(parents=False, exist_ok=True)

th.output_plots(
    dts,
    sts,
    fits,
    out_dir,
    names
)
