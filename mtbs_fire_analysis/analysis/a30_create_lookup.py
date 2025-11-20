# %%
import argparse
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

import mtbs_fire_analysis.analysis.utils as cache_utils
from mtbs_fire_analysis.pipeline.paths import CACHE_DIR, RESULTS_DIR
from mtbs_fire_analysis.analysis.defaults import (
    FIXED_LABELS,
    MTBS_END,
    MTBS_START,
    VARIED_LABELS,
)
from mtbs_fire_analysis.analysis.hlh_dist import (
    HalfLifeHazardDistribution as HLHD,
)

def hazard_to_burn_prob(hazard: float) -> float:
    """Convert hazard to burn probability."""
    if hazard <= 0:
        return 0.0
    return 1 - np.exp(-hazard)

def create_lookup_table(
    fits, max_date: dt | str = MTBS_END, refresh: bool = False
):
    """Create a lookup table for hazard values."""
    if type(max_date) is str:
        max_date = dt.fromisoformat(max_date)

    sts = cache_utils.get_sts(max_date=max_date, refresh=refresh)

    full_interval = max_date.year - MTBS_START.year

    records = []

    common_keys = list(
        filter(lambda x: x in FIXED_LABELS + VARIED_LABELS, fits.columns)
    )

    for row in fits.iter_rows(named=True):
        # Get the unique survival times for this row
        filters = {
            k: v for k, v in row.items() if k in FIXED_LABELS + VARIED_LABELS
        }
        pl_filters = [pl.col(col).is_in(vals) for col, vals in filters.items()]
        combined_filter = pl_filters[0]
        for f in pl_filters[1:]:
            combined_filter = combined_filter & f
        unique_sts = np.concatenate(
            [
                sts.filter(combined_filter)
                .select(pl.col("st"))
                .unique()
                .sort(pl.col("st"))
                .sort("st")
                .to_numpy()
                .flatten(),
            ]
        )

        fitter = HLHD(**row["params"])

        hazards = fitter.hazard(unique_sts)
        records.extend(
            [
                {
                    **{key: row[key] for key in common_keys},
                    **{
                        "st": unique_sts[i],
                        "burn_prob": hazard_to_burn_prob(hazards[i]),
                    },
                }
                for i in range(len(hazards))
            ]
            + [
                {
                    **{key: row[key] for key in common_keys},
                    **{
                        "st": full_interval,
                        "burn_prob": hazard_to_burn_prob(fitter.expected_hazard_ge(full_interval)),
                    },
                }
            ]
        )

    look_up_table = pl.from_dicts(records).with_columns(
        pl.col("st").cast(pl.Float32), pl.col("burn_prob").cast(pl.Float32)
    )

    for key in common_keys:
        look_up_table = look_up_table.explode(key)

    return look_up_table


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--max_date",
        help="Maximum date for event survival times",
        default="2022-01-01",
    )
    p.add_argument(
        "--fits",
        help="Directory containing fit data, in folders by year",
        default=RESULTS_DIR / "HLH_Fits",
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()

    year_file = Path(args.fits) / args.max_date
    fit_data_file = year_file / "fits.parquet"

    fits = pl.read_parquet(fit_data_file)

    look_up_table = create_lookup_table(
        fits, max_date=args.max_date, refresh=False
    )

    out_file = year_file / "hazard_lookup.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    look_up_table.write_parquet(out_file)
