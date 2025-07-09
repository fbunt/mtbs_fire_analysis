# %%
import argparse
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import yaml

import mtbs_fire_analysis.analysis.distributions as cd
from mtbs_fire_analysis.analysis.defaults import FIXED_LABELS, VARIED_LABELS
from mtbs_fire_analysis.analysis.hlh_dist import (
    HalfLifeHazardDistribution as HLHD,
)
from mtbs_fire_analysis.analysis.mining import (
    event_hist_to_cts,
    event_hist_to_dts,
    event_hist_to_uts,
)
from mtbs_fire_analysis.pipeline.paths import CACHE_DIR

pix_counts = pl.read_parquet(
    "mtbs_fire_analysis/data/eco_nlcd_mode_pixel_counts_eco3.pqt"
).rename(
    {
        "count": "Total num pixels",
    }
)


out_dir = Path("mtbs_fire_analysis") / "outputs" / "HLH_Fits_Eco3"
out_dir.mkdir(parents=False, exist_ok=True)


def do_fit(
    fitter, event_history, varied_filters, min_date, max_date, total_pixels
):
    # (fitter, dt_polygons, st_polygons, num_pixels, def_st):
    _dts_and_counts = event_hist_to_dts(event_history, varied_filters)

    _dts = _dts_and_counts["dt"].to_numpy()
    _dt_counts = _dts_and_counts["Pixel_Count"].to_numpy()

    # _sts_and_counts = event_hist_to_sts(event_history,
    #     min_date = None,
    #     max_date = max_date,
    #     fixed_pivots=fixed_pivots,
    #     varied_pivots=varied_pivots
    # )
    _cts_and_counts = event_hist_to_cts(
        event_history, max_date=max_date, varied_filters=varied_filters
    )

    _cts = _cts_and_counts["ct"].to_numpy()
    _ct_counts = _cts_and_counts["Pixel_Count"].to_numpy()

    _uts_and_counts = event_hist_to_uts(
        event_history, min_date=min_date, varied_filters=varied_filters
    )

    _uts = _uts_and_counts["ut"].to_numpy()
    _ut_counts = _uts_and_counts["Pixel_Count"].to_numpy()

    window_length = (max_date - min_date).days / 365.0

    empty_pixels = total_pixels - event_history["Pixel_Count"].sum()

    fitter.fit(
        _dts,
        _dt_counts,
        _cts,
        _ct_counts,
        _uts,
        _ut_counts,
        np.array([window_length]),
        np.array([empty_pixels]),
    )

    return fitter, _dts, _dt_counts, _cts, _ct_counts


def run_fit_over_configs(event_histories, configs, min_date, max_date):
    outputs = []

    for config in configs:
        # if config["name"] != "Southern Coastal Plain : Schrub":
        #      continue
        print(f"Fitting for config: {config['name']}")
        fixed_filters = {
            k: v
            for k, v in config.items()
            if k in FIXED_LABELS and k != "name"
        }
        varied_filters = {
            k: v
            for k, v in config.items()
            if k in VARIED_LABELS and k != "name"
        }
        fixed_pl_filters = [
            pl.col(col).is_in(vals) for col, vals in fixed_filters.items()
        ]
        varied_pl_filters = [
            pl.col(col).list.set_intersection(pl.lit(vals)).list.len() > 0
            for col, vals in varied_filters.items()
        ]
        pl_filters = fixed_pl_filters + varied_pl_filters
        combined_filter = pl_filters[0]
        for f in pl_filters[1:]:
            combined_filter = combined_filter & f
        config_histories = event_histories.filter(combined_filter)

        total_pixels_filters = {k: v for k, v in config.items() if k != "name"}
        pl_filters = [
            pl.col(col).is_in(vals)
            for col, vals in total_pixels_filters.items()
        ]
        combined_filter = pl_filters[0]
        for f in pl_filters[1:]:
            combined_filter = combined_filter & f
        total_pixels = (
            pix_counts.filter(combined_filter)
            .get_column("Total num pixels")
            .sum()
        )
        print(f"Total pixels: {total_pixels}")

        try:
            fitter, sub_dts, sub_dt_counts, sub_sts, sub_st_counts = do_fit(
                fitter=HLHD(hazard_inf=0.05, half_life=10),
                event_history=config_histories,
                varied_filters=varied_filters,
                min_date=min_date,
                max_date=max_date,
                total_pixels=total_pixels,
            )
        except RuntimeError as e:
            print(f"Failed to fit for config {config['name']}: {e}")
            continue

        cd.plot_fit(
            fitter,
            sub_dts,
            sub_dt_counts,
            sub_sts,
            sub_st_counts,
            out_dir / "plots" / (config["name"] + ".png"),
            max_dt=60,
        )

        output = config.copy()
        output["dist"] = fitter.dist_type
        output["params"] = fitter.params

        outputs.append(output)

        # create data frame from outputs
    out_df = pl.DataFrame(outputs)
    out_df.write_parquet(out_dir / "fits.parquet")


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--max_date",
        help="Maximum date for event histories",
        default="2023-01-01",
    )
    p.add_argument(
        "--min_date",
        help="Maximum date for event histories",
        default="1984-01-01",
    )
    p.add_argument(
        "--config_name",
        help="""Configuration name to use for fitting,
            pulls that config from configs.yaml""",
        default="eco3_config",
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()

    configs = Path("mtbs_fire_analysis") / "analysis" / "configs.yaml"

    with open(configs) as f:
        config_data = yaml.safe_load(f)

    # get event histories
    event_histories = pl.read_parquet(
        CACHE_DIR / "event_histories_2023-01-01.parquet"
    )

    run_fit_over_configs(
        event_histories,
        config_data[args.config_name],
        dt.strptime(args.min_date, "%Y-%m-%d"),
        dt.strptime(args.max_date, "%Y-%m-%d"),
    )
