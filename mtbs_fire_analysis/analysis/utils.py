from pathlib import Path

import polars as pl

from mtbs_fire_analysis.analysis.mining import (
    build_dts_df,
    build_survival_times,
)

cache_path = Path("/fastdata") / "FredData" / "cache"
data_path = Path("/fastdata") / "FredData" / "mtbs_CONUS_1984_2022"
lf = None

def get_lf():
    """Get the lazy frame."""
    global lf
    if lf is None:
        lf = pl.scan_parquet(data_path)
        return lf
    else:
        return lf

def get_dts(refresh=False):
    """Get the dts dataframe."""

    if refresh:
        ldts = build_dts_df(get_lf(), extra_cols=["nlcd", "Event_ID"])
        dts = ldts.collect()
        dts.write_parquet(cache_path / "dts.parquet")
    else:
        dts = pl.scan_parquet(cache_path / "dts.parquet").collect()
    return dts

def get_sts(refresh=False):
    """Get the sts dataframe."""

    if refresh:
        sts = build_survival_times(get_lf(), extra_cols=["nlcd", "Event_ID"]).collect()
        sts.write_parquet(cache_path / "sts.parquet")
    else:
        sts = pl.scan_parquet(cache_path / "sts.parquet").collect()
    return sts

def get_dt_polygons(refresh=False):
    """Get the dt polygons dataframe."""

    if refresh:
        dts = get_dts()
        dt_polygons = dts.group_by(["eco","nlcd2","Event_ID1","Event_ID2","dt"]).agg(
            pl.len().alias("Pixel Count")
        ).filter(pl.col("dt") > 0.5)
        dt_polygons.write_parquet(cache_path / "dt_polygons.parquet")
    else:
        dt_polygons = pl.scan_parquet(cache_path / "dt_polygons.parquet").collect()
    return dt_polygons

def get_st_polygons(refresh=False):
    """Get the st polygons dataframe."""

    if refresh:
        sts = get_sts()
        st_polygons = sts.group_by(["eco","nlcd","Event_ID", "st"]).agg(
            pl.len().alias("Pixel Count")
        )
        st_polygons.write_parquet(cache_path / "st_polygons.parquet")
    else:
        st_polygons = pl.scan_parquet(cache_path / "st_polygons.parquet").collect()
    return st_polygons
