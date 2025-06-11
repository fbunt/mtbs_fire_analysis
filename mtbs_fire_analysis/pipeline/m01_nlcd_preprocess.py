import glob
from pathlib import Path

import raster_tools as rts
from dask.diagnostics import ProgressBar

from mtbs_fire_analysis.defaults import DEFAULT_GEOHASH_GEOBOX
from mtbs_fire_analysis.pipeline.paths import NLCD_PATH, RAW_NLCD


def main():
    nlcd_paths = [Path(p) for p in sorted(glob.glob(str(RAW_NLCD / "*.tif")))]
    nlcd_rasters = [
        rts.Raster(p).reproject(DEFAULT_GEOHASH_GEOBOX) for p in nlcd_paths
    ]
    for r, path in zip(nlcd_rasters, nlcd_paths, strict=True):
        print(path)
        outpath = NLCD_PATH / path.name
        with ProgressBar():
            r.save(outpath)


if __name__ == "__main__":
    main()
