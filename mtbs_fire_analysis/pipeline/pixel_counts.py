# DEPRECATED 2026-05-01: canonical implementation lives downstream at
# `packages/fire-interval/fire_interval/etl/pixel_counts.py` (Phase 4a
# Q1=A migration; Phase 5b downstream-thick boundary close). The
# downstream copy is the canonical home for future regenerations of
# `eco_nlcd_mode_pixel_counts_eco3.pqt`. This upstream copy is
# retained as a runnable script for historical reproducibility; new
# invocations should target the downstream module via
# `python -m fire_interval.etl.pixel_counts`.
#
# This file will be removed when `feat/spatial-covariates` (both this
# upstream branch and the downstream branch) merges to upstream main.
# See downstream `docs/plans/PHASE_4A_BOUNDARY_DECISION.md`
# §"Migration sequence" step 4 for the long-term plan.
import argparse
import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import raster_tools as rts
from dask.diagnostics import ProgressBar

from mtbs_fire_analysis.pipeline.paths import (
    ECO_REGIONS_PATH,
    NLCD_MODE_RASTER_PATH,
    PIXEL_COUNT_OUT_FMT,
    RESULTS_DIR,
    ROOT_TMP_DIR,
)
from mtbs_fire_analysis.utils import protected_raster_save_with_cleanup


def clip_nlcd(eco_level):
    eco_name = ECO_LEVEL_TO_NAME[eco_level]
    eco_regions = (
        gpd.read_file(ECO_REGIONS_PATH)[[eco_name, "geometry"]]
        .dissolve(eco_name)
        .reset_index()
    )
    ecos = eco_regions[eco_name].unique()
    n = len(ecos)
    for i, eco in enumerate(ecos):
        geom = eco_regions[eco_regions[eco_name] == eco]
        print(f"Clipping NLCD to eco: {eco} ({i + 1}/{n})")
        try:
            out_path = (
                ROOT_TMP_DIR
                / f"nlcd_mode_clipped_to_eco_{eco_level}_{eco:05}.tif"
            )
            if out_path.exists():
                print(f"{out_path} already exists. Skipping")
                continue
            print(f"Saving to {out_path}")
            nlcd = rts.clipping.clip(geom, NLCD_MODE_RASTER_PATH)
            protected_raster_save_with_cleanup(nlcd, out_path, progress=False)
            nlcd = None
        except RuntimeError:
            print("Clip resulted in empty raster. Skipping")


ECO_LEVEL_TO_NAME = {1: "eco_lvl_1", 2: "eco_lvl_2", 3: "eco_lvl_3"}


def compute_counts(eco_level):
    eco_name = ECO_LEVEL_TO_NAME[eco_level]
    paths = sorted(
        glob.glob(
            str(ROOT_TMP_DIR / f"nlcd_mode_clipped_to_eco_{eco_level}_*.tif")
        )
    )
    ecos = [int(p[-9:-4]) for p in paths]
    results = []
    for eco, path in zip(ecos, paths, strict=True):
        print(f"Computing eco: {eco}")
        nlcd = rts.Raster(path)
        nv = nlcd.null_value
        values, counts = np.unique(nlcd, return_counts=True)
        mask = values == nv
        values = values[~mask]
        counts = counts[~mask]
        for vv, cc in zip(values, counts, strict=True):
            results.append((eco, vv, cc))
    results = np.array(results).T
    return pd.DataFrame(
        {
            "nlcd_src": ["mode"] * len(results[0]),
            eco_name: results[0],
            "nlcd": results[1],
            "count": results[2],
        }
    ).sort_values([eco_name, "nlcd"])


def main(eco_level):
    with ProgressBar():
        clip_nlcd(eco_level)
        print("---------")
        counts_df = compute_counts(eco_level)
    out_file = RESULTS_DIR / PIXEL_COUNT_OUT_FMT.format(eco_level=eco_level)
    print(f"Saving final counts to {out_file}")
    counts_df.to_parquet(out_file)


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-e",
        "--eco-level",
        type=int,
        default=1,
        help=(
            "Eco-region level to use when splitting up NLCD. Valid values "
            "are 1, 2, or 3. Default is 1."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _get_args()
    assert args.eco_level in {1, 2, 3}
    main(args.eco_level)
