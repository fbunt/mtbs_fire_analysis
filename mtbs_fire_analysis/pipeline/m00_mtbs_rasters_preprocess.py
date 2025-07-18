import glob
from pathlib import Path

import raster_tools as rts

from mtbs_fire_analysis.defaults import DEFAULT_GEOHASH_GEOBOX
from mtbs_fire_analysis.pipeline.paths import (
    CLEANED_RASTER_DATA_DIR,
    RAW_RASTER_DATA_DIR,
)
from mtbs_fire_analysis.utils import protected_raster_save_with_cleanup


def main():
    raster_paths = sorted(
        glob.glob(f"{RAW_RASTER_DATA_DIR}/**/*_CONUS_*.tif")
    )[:-2]
    [print(p) for p in raster_paths]
    rasters = [rts.Raster(p) for p in raster_paths]
    rasters_rp = [r.reproject(DEFAULT_GEOHASH_GEOBOX) for r in rasters]
    for data_path, r in zip(raster_paths, rasters_rp, strict=True):
        data_path = Path(data_path)
        out_path = CLEANED_RASTER_DATA_DIR / data_path.name
        print(out_path)
        protected_raster_save_with_cleanup(r, out_path)


if __name__ == "__main__":
    main()
