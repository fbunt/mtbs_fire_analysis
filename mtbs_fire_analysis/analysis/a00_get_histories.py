import argparse
from pathlib import Path
import polars as pl
from datetime import datetime as dt

from mtbs_fire_analysis.analysis.mining import (
    build_event_histories
)

from mtbs_fire_analysis.pipeline.paths import (
    RESULTS_DIR,
    CACHE_DIR
)

def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--max_date",
        help="Maximum date for event histories",
        default="2023-01-01"
    )
    p.add_argument(
        "--fixed_pivots", help="Pivots that are fixed per location",
        nargs="+",
        default=["eco_lvl_3"]
    )
    p.add_argument(
        "--varied_pivots",
        help="Pivots that are varying over time per location",
        nargs="+",
        default=["nlcd"]
    )
    return p

if __name__ == "__main__":
    args = _get_parser().parse_args()
    lf = pl.scan_parquet(Path(RESULTS_DIR) / "mtbs_CONUS_1984_2022")
    out_df = build_event_histories(lf,
        max_date=dt.fromisoformat(args.max_date),
        fixed_pivots=args.fixed_pivots,
        varied_pivots=args.varied_pivots
    )
    out_df.sink_parquet(CACHE_DIR / f"event_histories{args.max_date}.parquet")
