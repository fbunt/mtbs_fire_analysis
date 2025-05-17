import mtbs_fire_analysis.analysis.test_helpers as th
from mtbs_fire_analysis.analysis.distributions import (
    HalfLifeHazardDistribution as HLHD,
)

num_pixels = 1_000_000
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
nlc_fitter.fit(nlc_dts, nlc_sts)

lc_fitter = HLHD(hazard_inf=0.15, half_life=20)
lc_fitter.fit(lc_dts, lc_sts)

th.evaluate_fits(
    lc_dts,
    lc_sts,
    [truth, nlc_fitter, lc_fitter],
    names=["Truth","No Left Censoring", "Left Censoring"],
)
th.output_plots(
    lc_dts,
    lc_sts,
    [truth, nlc_fitter, lc_fitter],
    prefix="NoLeftCensoring_",
    names=["Truth", "No Left Censoring", "Left Censoring"],
)
