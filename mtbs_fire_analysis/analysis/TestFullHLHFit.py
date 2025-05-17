#%%
from pathlib import Path

import distributions as cd
import polars as pl

from mtbs_fire_analysis.analysis.distributions import (
    HalfLifeHazardDistribution as HLHD,
)
from mtbs_fire_analysis.analysis.mining import (
    build_dts_df,
    build_survival_times,
)

refresh_dts = False
refresh_sts = True

cache_path = (
    Path("/fastdata") / "FredData" /  "cache"
)

if refresh_dts or refresh_sts:
    data_path = Path("/fastdata") / "FredData" / "mtbs_CONUS_1984_2022"
    lf = pl.scan_parquet(data_path)
    cache_path.mkdir(parents=True, exist_ok=True)

if refresh_dts:
    ldts = build_dts_df(lf, extra_cols=["nlcd"])
    dts = ldts.collect()
    dts.write_parquet(cache_path / "dts.parquet")
else:
    dts = pl.scan_parquet(cache_path / "dts.parquet").collect()

if refresh_sts:
    sts = build_survival_times(lf, extra_cols=["nlcd"]).collect()
    sts.write_parquet(cache_path / "sts.parquet")
else:
    sts = pl.scan_parquet(cache_path / "sts.parquet").collect()

#%%
"""Rough task list:
- Set up description of the groups of data we care about
    (configurable, for what eco regions grouping what LUT)
- Pull down dts and survival times for each subset
- Fit the data to the HLH distribution
- Plot the results, and the parameters of the fit
- Save the results to files
"""
test_config = {"name": "NW Mountains Forest", "eco": [6], "nlcd": [41, 42, 43]}

configs = [test_config]

outputs = {}

out_dir = Path("Outputs")

for config in configs:
    prefix = "TestHLHFull"
    # Get the data for this config
    sub_dts = (
        dts.filter(
            (pl.col("eco").is_in(config["eco"]))
            & (pl.col("nlcd1").is_in(config["nlcd"]))
            & (pl.col("dt") > 1.0)
        )
        .select("dt")
        .to_numpy()
        .flatten()
    )
    # Get the survival times for this config
    sub_sts = (
        sts.filter(
            (pl.col("eco").is_in(config["eco"]))
            & (pl.col("nlcd").is_in(config["nlcd"]))
        )
        .select("survival_time")
        .to_numpy()
        .flatten()
    )

    # Fit the data to the HLH distribution
    fitter = HLHD(hazard_inf=0.1, half_life=3)
    fitter.fit(sub_dts, sub_sts)
    #fitter.fit(sub_dts)#, sub_sts)
    # Plot the results
    out_file = out_dir / (prefix + config["name"] + ".png")
    cd.plot_fit(fitter,sub_dts, sub_sts, out_file)
    outputs[config["name"]] = {
        "fitter": fitter,
        "dist": fitter.dist_type,
        "params": fitter.params,
        "eco": config["eco"],
        "nlcd": config["nlcd"],
    }

print(outputs)
#%%
