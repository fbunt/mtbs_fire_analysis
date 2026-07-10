import argparse
import glob
import re
import shutil
from pathlib import Path

import raster_tools as rts

from mtbs_fire_analysis.defaults import DEFAULT_GEOHASH_GEOBOX
from mtbs_fire_analysis.pipeline.paths import NLCD_PATH, RAW_NLCD
from mtbs_fire_analysis.utils import protected_raster_save_with_cleanup


def _get_year(path):
    match = re.search(r"\d{4}", path.name)
    assert match is not None, f"Could not parse year from {path.name}"
    return int(match.group())


def main(start_year):
    nlcd_paths = [Path(p) for p in sorted(glob.glob(str(RAW_NLCD / "*.tif")))]
    nlcd_paths = [p for p in nlcd_paths if _get_year(p) >= start_year]
    if not nlcd_paths:
        raise FileNotFoundError(
            f"No NLCD tifs found in {RAW_NLCD} for years >= {start_year}"
        )
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
        protected_raster_save_with_cleanup(r, outpath)
    # Duplicate 1985 to get an NLCD for 1984
    if path1984 is None:
        print("1985 not processed. Skipping copy of 1985 to 1984.")
    elif not path1984.exists():
        print("Copying 1985 to 1984")
        shutil.copy(path1985, path1984)
    else:
        print("Skipping copy of 1985 to 1984. Already exists.")


DESC = """
Reproject raw NLCD rasters to the common grid and duplicate 1985 as 1984.
"""


def _get_parser():
    p = argparse.ArgumentParser(description=DESC)
    p.add_argument(
        "--start_year",
        default=1985,
        type=int,
        help="Year to start processing at",
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    main(args.start_year)
