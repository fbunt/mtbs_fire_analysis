"""Aggregate the 30 m EDNA DEM to 270 m for r.sun input.

The 270 m grid is a clean 9x coarsening of DEFAULT_GEOHASH_GEOBOX
(same origin, 270 m pixel size, shape // 9 with the trailing partial
row/col truncated). Each 270 m cell is a perfect union of 9x9 30 m
cells.

Why 270 m for solar covariate (per SOLAR_COVARIATE_PLAN.md §6 / dry-run
2026-05-13): r.horizon at 90 m on an 800 km buffered tile takes hours
single-threaded; at 270 m it drops to ~10 minutes per tile. Accuracy
loss is small for our covariate use case (per-stratum quantile binning
+ bilinear resample to the 30 m output grid already smooths out
sub-270m features). CONUS terrain shadows are dominated by features
1+ km wide which 270 m captures.
"""

import raster_tools as rts
from affine import Affine
from dask.distributed import Client, LocalCluster
from odc.geo.geobox import GeoBox

from mtbs_fire_analysis.defaults import (
    DEFAULT_CRS,
    DEFAULT_GEOHASH_AFFINE,
    DEFAULT_GEOHASH_GRID_SHAPE,
)
from mtbs_fire_analysis.pipeline.paths import (
    ELEVATION_270M_PATH,
    ELEVATION_PATH,
)
from mtbs_fire_analysis.utils import protected_raster_save_with_cleanup


def _geobox_270m() -> GeoBox:
    shape = (
        DEFAULT_GEOHASH_GRID_SHAPE[0] // 9,
        DEFAULT_GEOHASH_GRID_SHAPE[1] // 9,
    )
    affine = Affine(
        270.0,
        0.0,
        DEFAULT_GEOHASH_AFFINE.c,
        0.0,
        -270.0,
        DEFAULT_GEOHASH_AFFINE.f,
    )
    return GeoBox(shape, affine, DEFAULT_CRS)


def main():
    cluster = LocalCluster()
    Client(cluster)

    geobox_270m = _geobox_270m()
    elevation_30m = rts.Raster(ELEVATION_PATH).set_null_value(-99_999)
    elevation_270m = elevation_30m.reproject(
        geobox_270m, resample_method="average"
    )
    protected_raster_save_with_cleanup(
        elevation_270m, ELEVATION_270M_PATH, progress=False, BIGTIFF="YES"
    )


if __name__ == "__main__":
    main()
