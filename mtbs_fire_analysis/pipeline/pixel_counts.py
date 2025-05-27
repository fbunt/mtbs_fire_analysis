import argparse
import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import raster_tools as rts
from dask.diagnostics import ProgressBar

from mtbs_fire_analysis.pipeline.paths import (
    ECO_REGIONS_PATH,
    NLCD_PATH,
    NLCD_TIF_FMT,
    PIXEL_COUNT_OUT_FMT,
    RESULTS_DIR,
    ROOT_TMP_DIR,
)


def get_nlcd(year):
    return NLCD_PATH / NLCD_TIF_FMT.format(year=year)


def clip_nlcd(eco_level, year):
    eco_name = ECO_LEVEL_TO_NAME[eco_level]
    eco_regions = (
        gpd.read_file(ECO_REGIONS_PATH)[[eco_name, "geometry"]]
        .dissolve(eco_name)
        .reset_index()
    )
    ecos = eco_regions[eco_name].unique()
    for eco in ecos:
        geom = eco_regions[eco_regions[eco_name] == eco]
        nlcd = get_nlcd(year)
        print(f"Clipping NLCD to eco: {eco}")
        try:
            out_path = (
                ROOT_TMP_DIR
                / f"nlcd_clipped_to_eco_{year}_{eco_level}_{eco:05}.tif"
            )
            if out_path.exists():
                print(f"{out_path} already exists. Skipping")
                continue
            print(f"Saving to {out_path}")
            nlcd = rts.clipping.clip(geom, nlcd)
            nlcd.save(out_path)
        except RuntimeError:
            print("Clip resulted in empty raster. Skipping")


ECO_LEVEL_TO_NAME = {1: "eco_lvl_1", 2: "eco_lvl_2", 3: "eco_lvl_3"}


def compute_counts(eco_level, year):
    eco_name = ECO_LEVEL_TO_NAME[eco_level]
    paths = sorted(
        glob.glob(
            str(ROOT_TMP_DIR / f"nlcd_clipped_to_eco_{year}_{eco_level}_*.tif")
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
            "year": [year] * len(results[0]),
            eco_name: results[0],
            "nlcd": results[1],
            "count": results[2],
        }
    ).sort_values([eco_name, "nlcd"])


def main(eco_level, nlcd_year):
    with ProgressBar():
        clip_nlcd(eco_level, nlcd_year)
        print("---------")
        counts_df = compute_counts(eco_level, nlcd_year)
    out_file = RESULTS_DIR / PIXEL_COUNT_OUT_FMT.format(
        year=nlcd_year, eco_level=eco_level
    )
    print(f"Saving final counts to {out_file}")
    counts_df.to_parquet(out_file)


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument("nlcd_year", type=int, help="The target NLCD year")
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
    main(args.eco_level, args.nlcd_year)
