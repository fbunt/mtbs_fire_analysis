import argparse
import glob
from pathlib import Path

import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar

import raster_tools as rts
from paths import CLEANED_RASTER_DATA_DIR, RAW_RASTER_DATA_DIR

DATA_TIF_FMT = "mtbs_{aoi}_{year}.tif"


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "aoi_code", help="Area of interest code (e.g. CONUS, MT, etc)"
    )
    return p


def get_data_raster_path(year, aoi_code):
    return (
        RAW_RASTER_DATA_DIR
        / f"{year}"
        / DATA_TIF_FMT.format(aoi=aoi_code, year=year)
    )


def main(aoi_code):
    raster_paths = sorted(
        glob.glob(f"{RAW_RASTER_DATA_DIR}/**/*{aoi_code}*.tif")
    )[:-2]
    rasters = [rts.Raster(p) for p in raster_paths]
    combined = rts.band_concat(rasters, join="outer")
    data = da.ones((1, *combined.shape[1:]), dtype=np.int8)
    combined = rts.data_to_raster_like(data, combined)
    geobox = combined.geobox
    combined = None
    rasters_rp = [r.reproject(geobox) for r in rasters]
    for data_path, r in zip(raster_paths, rasters_rp):
        data_path = Path(data_path)
        out_path = CLEANED_RASTER_DATA_DIR / data_path.name
        print(out_path)
        if out_path.exists():
            print("Skipping")
            continue
        with ProgressBar():
            r.save(out_path)


if __name__ == "__main__":
    args = _get_parser().parse_args()
    aoi_code = args.aoi_code
    main(aoi_code)
