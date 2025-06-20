import glob
import shutil
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
    path1984 = None
    path1985 = None
    for r, path in zip(nlcd_rasters, nlcd_paths, strict=True):
        outpath = NLCD_PATH / path.name
        if path1984 is None and "1985" in path.name:
            path1984 = NLCD_PATH / path.name.replace("1985", "1984")
            path1985 = outpath
        print(f"{path} --> {outpath}")
        if outpath.exists():
            print("Skipping")
            continue
        with ProgressBar():
            r.save(outpath, tiled=True)
    # Duplicate 1985 to get an NLCD for 1984
    print("Copying 1985 to 1984")
    shutil.copy(path1985, path1984)


if __name__ == "__main__":
    main()
