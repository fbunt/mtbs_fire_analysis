from datetime import datetime as dt

import polars as pl

from mtbs_fire_analysis.analysis.mining import (
    build_event_histories,
    event_hist_to_blank_pixels_events,
    event_hist_to_dts,
    event_hist_to_sts,
)

data = pl.DataFrame(
    {
        "geohash": [
            "a",
            "a",
            "b",
            "b",
            "b",
            "c",
            "c",
            "d",
            "d",
            "e",
            "e",
            "f",
        ],
        "Ig_Date": [
            dt(1990, 1, 1),
            dt(1992, 1, 1),
            dt(1990, 1, 1),
            dt(1992, 1, 1),
            dt(1994, 1, 1),
            dt(1990, 1, 1),
            dt(1992, 1, 1),
            dt(1992, 1, 1),
            dt(1994, 1, 1),
            dt(1991, 1, 1),
            dt(1993, 1, 1),
            dt(1991, 1, 1),
        ],
        "eco_lvl_1": ["A"] * 12,
        "eco_lvl_2": ["A"] * 12,
        "eco_lvl_3": ["A"] * 12,
        "nlcd": [1, 1, 1, 1, 4, 1, 1, 2, 2, 2, 2, 2],
        "perim_index": [1, 3, 1, 3, 5, 1, 3, 3, 5, 2, 4, 2],
    }
)

event_hist = build_event_histories(
    data,
    max_date=dt(1995, 1, 1),
    fixed_pivots=["eco_lvl_3"],
    varied_pivots=["nlcd"],
)

dts = event_hist_to_dts(
    event_hist, fixed_pivots=["eco_lvl_3"], varied_pivots=["nlcd"]
)
sts = event_hist_to_sts(
    event_hist,
    min_date=dt(1989, 1, 1),
    max_date=dt(1995, 1, 1),
    fixed_pivots=["eco_lvl_3"],
    varied_pivots=["nlcd"],
)
no_events = event_hist_to_blank_pixels_events(event_hist, total_pixels=20)
