import argparse
import glob
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import raster_tools as rts
from dask.diagnostics import ProgressBar

from mtbs_fire_analysis.pipeline.paths import (
    CLEANED_RASTER_DATA_DIR,
    RAW_RASTER_DATA_DIR,
    STATES_PATH,
)

DATA_TIF_FMT = "mtbs_{aoi}_{year}.tif"


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "aoi_code", help="Area of interest code (e.g. CONUS, MT, etc)"
    )
    return p


def get_aoi_geom(aoi_code, crs):
    geoms = gpd.read_file(STATES_PATH)
    if aoi_code in ("ALL", "CONUS"):
        geoms = geoms[geoms.GEOID.astype(int) < 60]
        if aoi_code == "CONUS":
            geoms = geoms[~geoms.STUSPS.isin(("AK", "HI"))]
    else:
        geoms = geoms[aoi_code == geoms.STUSPS]
    return gpd.GeoSeries([geoms.geometry.to_crs(crs).union_all()], crs=crs)


def main(aoi_code):
    is_wus = aoi_code == "WUS"
    if is_wus:
        aoi_code = "CONUS"
    raster_paths = sorted(
        glob.glob(f"{RAW_RASTER_DATA_DIR}/**/*_{aoi_code}_*.tif")
    )[:-2]
    [print(p) for p in raster_paths]
    rasters = [rts.Raster(p) for p in raster_paths]
    if is_wus:
        crs = rasters[0].crs
        # Buffer out 50 km
        gs = get_aoi_geom("WUS", crs).buffer(50_000)
        bounds = gs.total_bounds
        rasters = [rts.clipping.clip_box(r, bounds) for r in rasters]
    combined = rts.band_concat(rasters, join="outer")
    data = da.ones((1, *combined.shape[1:]), dtype=np.int8)
    combined = rts.data_to_raster_like(data, combined)
    geobox = combined.geobox
    combined = None
    rasters_rp = [r.reproject(geobox) for r in rasters]
    for data_path, r in zip(raster_paths, rasters_rp):
        data_path = Path(data_path)
        if is_wus:
            out_path = (
                CLEANED_RASTER_DATA_DIR
                / f"mtbs_WUS_{data_path.name[-8:-4]}.tif"
            )
        else:
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
