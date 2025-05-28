#%%
from pathlib import Path

import numpy as np
import polars as pl
import pickle

import mtbs_fire_analysis.analysis.distributions as cd
from mtbs_fire_analysis.analysis.distributions import (
    HalfLifeHazardDistribution as HLHD,
)
from mtbs_fire_analysis.analysis.mining import (
    build_dts_df,
    build_survival_times,
)

refresh_dts = False
refresh_sts = False
refresh_polygons = False

bootstrap = False

cache_path = Path("/fastdata") / "FredData" / "cache"

if refresh_dts or refresh_sts:
    data_path = Path("/fastdata") / "FredData" / "mtbs_CONUS_1984_2022"
    lf = pl.scan_parquet(data_path)
    cache_path.mkdir(parents=True, exist_ok=True)

if refresh_dts:
    ldts = build_dts_df(lf, extra_cols=["nlcd", "Event_ID"])
    dts = ldts.collect()
    dts.write_parquet(cache_path / "dts.parquet")
else:
    dts = pl.scan_parquet(cache_path / "dts.parquet").collect()

if refresh_sts:
    sts = build_survival_times(lf, extra_cols=["nlcd", "Event_ID"]).collect()
    sts.write_parquet(cache_path / "sts.parquet")
else:
    sts = pl.scan_parquet(cache_path / "sts.parquet").collect()

if refresh_polygons:
    dt_polygons = dts.group_by(["eco","nlcd2","Event_ID1","Event_ID2","dt"]).agg(
        pl.len().alias("Pixel Count")
    ).filter(pl.col("dt") > 0.5)

    st_polygons = sts.filter(pl.col("n")>1).group_by(["eco","nlcd","Event_ID", "st"]).agg(
        pl.len().alias("Pixel Count")
    )
    dt_polygons.write_parquet(cache_path / "dt_polygons.parquet")
    st_polygons.write_parquet(cache_path / "st_polygons.parquet")
else:
    dt_polygons = pl.scan_parquet(cache_path / "dt_polygons.parquet").collect()
    st_polygons = pl.scan_parquet(cache_path / "st_polygons.parquet").collect()


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

outputs = {}

out_dir = Path("mtbs_fire_analysis") / "outputs" / "HLH_Fits"
out_dir.mkdir(parents=False, exist_ok=True)

def do_fit(fitter, dt_polygons, st_polygons, num_pixels, def_st):
    dts_and_counts = dt_polygons.group_by(["dt"]).agg(
        pl.sum("Pixel Count").alias("Count")
    )
    _dts = dts_and_counts.get_column("dt").to_numpy().flatten()
    _dt_counts = dts_and_counts.get_column("Count").to_numpy().flatten()

    sts_and_counts = st_polygons.group_by(["st"]).agg(
        pl.sum("Pixel Count").alias("Count")
    )
    _sts = sts_and_counts.get_column("st").to_numpy().flatten()
    _st_counts = sts_and_counts.get_column("Count").to_numpy().flatten()

    extra_pixels = num_pixels - _st_counts.sum()
    if extra_pixels < 0:
        raise ValueError(
            f"num survival times > max: {num_pixels} - {_st_counts.sum()}"
        )
    _sts = np.append(_sts, def_st)
    _st_counts = np.append(_st_counts, extra_pixels)

    fitter.fit(_dts, _dt_counts, _sts, _st_counts)

    return fitter, _dts, _dt_counts, _sts, _st_counts

for config in configs:
    sub_dt_polygons = (
        dt_polygons.filter(
            (pl.col("eco").is_in(config["eco"]))
            & (pl.col("nlcd2").is_in(config["nlcd"]))
        )
    )

    sub_st_polygons = (
        st_polygons.filter(
            (pl.col("eco").is_in(config["eco"]))
            & (pl.col("nlcd").is_in(config["nlcd"]))
        )
    )

    num_pixels = sub_st_polygons.get_column("Pixel Count").sum()
    #num_pixels = 10_000_000
    def_st = 38.0

    fitter, sub_dts, sub_dt_counts, sub_sts, sub_st_counts  = do_fit(
        HLHD(hazard_inf=0.1, half_life=3),
        sub_dt_polygons,
        sub_st_polygons,
        num_pixels,
        def_st
    )

    out_file = out_dir / (config["name"] + ".png")
    #cd.plot_fit(fitter, sub_dts, sub_sts, out_file)
    cd.plot_fit(
        fitter,
        sub_dts,
        sub_dt_counts,
        sub_sts,
        sub_st_counts,
        out_dir / (config["name"] + ".png"),
        max_dt=60,
    )
    outputs[config["name"]] = {
        "fitter": fitter,
        "dist": fitter.dist_type,
        "params": fitter.params,
        "eco": config["eco"],
        "nlcd": config["nlcd"],
    }

    if bootstrap:
        bootstrap_cache = {}
        for param in fitter.params:
            bootstrap_cache[param] = []
        bootstrap_cache["FRI"] = []
        num_bootstraps = 1000
        dt_samples = sub_dt_polygons.shape[1]
        st_samples = sub_st_polygons.shape[1]
        for i in range(num_bootstraps):
            # Sample the data with replacement
            sample_dt_polygons = sub_dt_polygons.sample(
                n=dt_samples, replace=True)
            sample_st_polygons = sub_st_polygons.sample(
                n=st_samples, replace=True)
            # Get the dts and counts for this sample
            boot_fitter = do_fit(
                HLHD(hazard_inf=0.1, half_life=3),
                sample_dt_polygons,
                sample_st_polygons,
                num_pixels,
                def_st
            )
            for param in fitter.params:
                bootstrap_cache[param].append(fitter.params[param])
            bootstrap_cache["FRI"].append(fitter.mean())

#print(outputs)

with open(out_dir / "HLH_fits.pkl", "wb") as f:
    pickle.dump(outputs, f)
#%%
