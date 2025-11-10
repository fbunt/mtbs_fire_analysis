import pandas as pd

import mtbs_fire_analysis.analysis.statistical_tests.test_helpers as th
from mtbs_fire_analysis.analysis.scipy_dist import (
    Weibull as WBD,
)

num_pixels = 200
time_interval = 38

pre_window = 1000

truth = WBD(shape=1.5, scale=25)

init = WBD(shape=1.8, scale=10)

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
dt_only_fitter = init.copy()
dt_only_fitter.fit(dts, dt_counts)

dtct_fitter = init.copy()
dtct_fitter.fit(dts, dt_counts, cts, ct_counts)

dtctut_fitter = init.copy()
dtctut_fitter.fit(dts, dt_counts, cts, ct_counts, uts, ut_counts)

dtctutet_fitter = init.copy()
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



# Generic param extraction (works for any BaseLifetime subclass exposing .params)
params_df = pd.DataFrame([f.params for f in fits])
params_df.insert(0, "name", fit_names)
params_df["mean_dt"] = [f.mean() for f in fits]

print(params_df)
