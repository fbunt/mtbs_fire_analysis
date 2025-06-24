import pandas as pd

import mtbs_fire_analysis.analysis.statistical_tests.test_helpers as th
from mtbs_fire_analysis.analysis.hlh_dist import (
    HalfLifeHazardDistribution as HLHD,
)

num_pixels = 200
time_interval = 38

pre_window = 1000

truth = HLHD(hazard_inf=0.04, half_life=25)

dts, dt_counts, cts, ct_counts, uts, ut_counts, ets, et_counts = (
    th.create_sample_data_all(
        num_pixels=num_pixels,
        time_interval=time_interval,
        truth=truth,
        start_up_time=pre_window,
        round_to_nearest=0.1,
    )
)
print("sim data created")
dt_only_fitter = HLHD(hazard_inf=0.15, half_life=20)
dt_only_fitter.fit(dts, dt_counts)

dtct_fitter = HLHD(hazard_inf=0.15, half_life=20)
dtct_fitter.fit(dts, dt_counts, cts, ct_counts)

dtctut_fitter = HLHD(hazard_inf=0.15, half_life=20)
dtctut_fitter.fit(dts, dt_counts, cts, ct_counts, uts, ut_counts)

dtctutet_fitter = HLHD(hazard_inf=0.15, half_life=20)
dtctutet_fitter.fit(
    dts, dt_counts, cts, ct_counts, uts, ut_counts, ets, et_counts
)

fit_names = [
    "truth",
    "dt_only",
    "dtct",
    "dtctut",
    "dtctutet",
]
# create a dataframe with the fitted parameters, means, and names

fits = [
    truth,
    dt_only_fitter,
    dtct_fitter,
    dtctut_fitter,
    dtctutet_fitter,
]  # , dumb_survival_fitter]

output_df = pd.DataFrame(
    {
        "name": fit_names,
        "hazard_inf": [fit.hazard_inf for fit in fits],
        "half_life": [fit.half_life for fit in fits],
        "mean_dt": [fit.mean() for fit in fits],
    }
)
print(output_df)
