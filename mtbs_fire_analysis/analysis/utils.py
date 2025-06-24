from datetime import datetime as dt
from pathlib import Path

import polars as pl

from mtbs_fire_analysis.analysis.defaults import (
    MTBS_END,
)
from mtbs_fire_analysis.analysis.mining import (
    build_dts_df,
    build_event_histories,
    build_survival_times,
)
from mtbs_fire_analysis.pipeline.paths import CACHE_DIR, RESULTS_DIR

data_path = Path(RESULTS_DIR) / "mtbs_CONUS_1984_2022"
# data_path = Path("/fastdata") / "data" / "results" / "mtbs_CONUS_1984_2022"
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
        dts.write_parquet(CACHE_DIR / "dts.parquet")
    else:
        dts = pl.scan_parquet(CACHE_DIR / "dts.parquet").collect()
    return dts


def get_sts(max_date: dt | str = MTBS_END, refresh: bool = False):
    """Get the sts dataframe."""
    if type(max_date) is str:
        max_date = dt.fromisoformat(max_date)
    if refresh:
        sts = build_survival_times(
            get_lf(), max_date, extra_cols=["nlcd", "Event_ID"]
        ).collect()
        sts.write_parquet(
            CACHE_DIR / f"sts_{max_date.strftime('%Y-%m-%d')}.parquet"
        )
    else:
        sts = pl.scan_parquet(
            CACHE_DIR / f"sts_{max_date.strftime('%Y-%m-%d')}.parquet"
        ).collect()
    return sts


def get_dt_polygons(refresh=False):
    """Get the dt polygons dataframe."""

    if refresh:
        dts = get_dts()
        dt_polygons = (
            dts.group_by(["eco3", "nlcd2", "Event_ID1", "Event_ID2", "dt"])
            .agg(pl.len().alias("Pixel Count"))
            .filter(pl.col("dt") > 0.5)
        )
        dt_polygons.write_parquet(CACHE_DIR / "dt_polygons.parquet")
    else:
        dt_polygons = pl.scan_parquet(
            CACHE_DIR / "dt_polygons.parquet"
        ).collect()
    return dt_polygons


def get_st_polygons(refresh=False):
    """Get the st polygons dataframe."""

    if refresh:
        sts = get_sts()
        st_polygons = sts.group_by(["eco3", "nlcd", "Event_ID", "st"]).agg(
            pl.len().alias("Pixel Count")
        )
        st_polygons.write_parquet(CACHE_DIR / "st_polygons.parquet")
    else:
        st_polygons = pl.scan_parquet(
            CACHE_DIR / "st_polygons.parquet"
        ).collect()
    return st_polygons


def get_events(refresh=False):
    """Get the events dataframe."""
    if refresh:
        lf = get_lf()
        events = build_event_histories(
            lf,
            max_date=None,
            fixed_pivots=["eco_lvl_3"],
            varied_pivots=["nlcd"],
        )

        events.write_parquet(CACHE_DIR / "events.parquet")
    else:
        events = pl.scan_parquet(CACHE_DIR / "events.parquet").collect()
    return events
