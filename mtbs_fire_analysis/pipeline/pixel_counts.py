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
    RESULTS_DIR,
    ROOT_TMP_DIR,
)

TARGET_YEAR = 2003


def get_nlcd():
    return NLCD_PATH / NLCD_TIF_FMT.format(year=TARGET_YEAR)


def clip_nlcd():
    eco_regions = (
        gpd.read_file(ECO_REGIONS_PATH)[["eco_lvl_1", "geometry"]]
        .dissolve("eco_lvl_1")
        .reset_index()
    )
    ecos = eco_regions.eco_lvl_1.unique()
    for eco in ecos:
        geom = eco_regions[eco_regions.eco_lvl_1 == eco]
        nlcd = get_nlcd()
        print(f"Clipping NLCD to eco: {eco}")
        try:
            out_path = ROOT_TMP_DIR / f"nlcd_clipped_to_eco_{eco:03}.tif"
            if out_path.exists():
                print(f"{out_path} already exists. Skipping")
                continue
            print(f"Saving to {out_path}")
            nlcd = rts.clipping.clip(geom, nlcd)
            nlcd.save(out_path)
        except RuntimeError:
            print("Clip resulted in empty raster. Skipping")


def compute_counts():
    paths = sorted(glob.glob(str(ROOT_TMP_DIR / "nlcd_clipped_to_eco_*.tif")))
    ecos = [int(p[-7:-4]) for p in paths]
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
        {"eco_lvl_1": results[0], "nlcd": results[1], "count": results[2]}
    ).sort_values(["eco_lvl_1", "nlcd"])


def main():
    with ProgressBar():
        clip_nlcd()
        print("---------")
        counts_df = compute_counts()
    out_file = RESULTS_DIR / "eco_nlcd_pixel_counts.pqt"
    print(f"Saving final counts to {out_file}")
    counts_df.to_parquet(out_file)


if __name__ == "__main__":
    main()
