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

cache_path = Path("/fastdata") / "FredData" / "cache"

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
configs = [{"name": "Northern Wetland", "eco": [5], "nlcd": [90, 95]}, # 4M
           {"name": "NW Mountain Forest", "eco": [6], "nlcd": [41, 42, 43]}, # 100M
           {"name": "NW Mountain Schrub", "eco": [6], "nlcd": [52]}, # 41M
           {"name": "NW Mountain Grassland", "eco": [6], "nlcd": [71]}, # 52M
           {"name": "Eastern Temperate Forest", "eco": [8], "nlcd": [41, 42, 43]}, # 62M
           {"name": "Eastern Temperate Wetlands", "eco": [8], "nlcd": [90,95]}, # 31M
           {"name": "Great Plains Forest", "eco": [9], "nlcd": [41, 42, 43]}, # 5M
           {"name": "Great Plains Schrub", "eco": [9], "nlcd": [52]}, # 33M
           {"name": "Great Plains Grassland", "eco": [9], "nlcd": [71]}, # 57M 
           {"name": "Great Plains Wetland", "eco": [9], "nlcd": [90, 95]}, # 8M 
           {"name": "NA Desert Forest", "eco": [10], "nlcd": [41, 42, 43]}, # 7M 
           {"name": "NA Desert Schrub", "eco": [10], "nlcd": [52]}, # 85M
           {"name": "NA Desert Grassland", "eco": [10], "nlcd": [71]}, # 94M
           {"name": "Medit California Forest", "eco": [11], "nlcd": [41, 42, 43]}, # 8M
           {"name": "Medit California Schrub", "eco": [11], "nlcd": [52]}, # 25M
           {"name": "Medit California Grassland", "eco": [11], "nlcd": [71]}, # 18M
           {"name": "Southern Semiarid Schrub", "eco": [12], "nlcd": [52]}, # 10M
           {"name": "Temperate Sierra Forest", "eco": [13], "nlcd": [41, 42, 43]}, # 20M
           {"name": "Temperate Sierra Schrub", "eco": [13], "nlcd": [52]}, # 11M
           {"name": "Temperate Sierra Grassland", "eco": [13], "nlcd": [71]}, # 6M
           {"name": "Tropical Wetland", "eco": [15], "nlcd": [90, 95]}, # 11M
           ]

eco_to_str = {
    5: "Northern Forest",
    6: "NW Forested Mountains",
    7: "Marine West Coast Forest",
    8: "Eastern Temperate Forests",
    9: "Great Plains",
    10: "North American Deserts",
    11: "Mediterranean California",
    12: "Southern Semiarid Highlands",
    13: "Temperate Sierras",
}

outputs = {}

out_dir = Path("Outputs")

for config in configs[1]:
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
    cd.plot_fit(fitter, sub_dts, sub_sts, out_file)
    outputs[config["name"]] = {
        "fitter": fitter,
        "dist": fitter.dist_type,
        "params": fitter.params,
        "eco": config["eco"],
        "nlcd": config["nlcd"],
    }

print(outputs)
#%%
