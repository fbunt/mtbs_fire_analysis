import argparse
import glob
from pathlib import Path

from dask.diagnostics import ProgressBar

import raster_tools as rts
from paths import NLCD_PATH, RAW_NLCD


def _path(path):
    path = Path(path)
    assert path.exists()
    return path


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("like_path", type=_path, help="Path to like raster")
    return p


def main(like_path):
    like = rts.Raster(like_path)
    nlcd_paths = [Path(p) for p in sorted(glob.glob(str(RAW_NLCD / "*.tif")))]
    nlcd_rasters = [rts.Raster(p).reproject(like.geobox) for p in nlcd_paths]
    for r, path in zip(nlcd_rasters, nlcd_paths):
        print(path)
        outpath = NLCD_PATH / path.name
        with ProgressBar():
            r.save(outpath)


if __name__ == "__main__":
    args = _get_parser().parse_args()
