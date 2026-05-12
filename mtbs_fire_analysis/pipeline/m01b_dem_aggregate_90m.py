"""Aggregate the 30 m EDNA DEM to 90 m for r.sun input.

The 90 m grid is a clean 3x coarsening of DEFAULT_GEOHASH_GEOBOX (same
origin, 90 m pixel size, shape truncated to drop the last partial row/col).
Each 90 m cell is a perfect union of 3x3 30 m cells, so the bilinear
resample back to 30 m later has clean nesting and the project-grid
alignment invariant in SOLAR_COVARIATE_PLAN.md §9 holds end-to-end.

GDAL "average" resampling computes the mean of contributing pixels and
ignores nulls (nodata=-99999), so 90 m cells at the CONUS edge with some
valid 30 m children become valid mean-of-the-valid-children rather than
null. See SOLAR_COVARIATE_PLAN.md §3 row 2.
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
    ELEVATION_90M_PATH,
    ELEVATION_PATH,
)
from mtbs_fire_analysis.utils import protected_raster_save_with_cleanup


def _geobox_90m() -> GeoBox:
    shape = (
        DEFAULT_GEOHASH_GRID_SHAPE[0] // 3,
        DEFAULT_GEOHASH_GRID_SHAPE[1] // 3,
    )
    affine = Affine(
        90.0,
        0.0,
        DEFAULT_GEOHASH_AFFINE.c,
        0.0,
        -90.0,
        DEFAULT_GEOHASH_AFFINE.f,
    )
    return GeoBox(shape, affine, DEFAULT_CRS)


def main():
    cluster = LocalCluster()
    Client(cluster)

    geobox_90m = _geobox_90m()
    elevation_30m = rts.Raster(ELEVATION_PATH).set_null_value(-99_999)
    elevation_90m = elevation_30m.reproject(
        geobox_90m, resample_method="average"
    )
    protected_raster_save_with_cleanup(
        elevation_90m, ELEVATION_90M_PATH, progress=False, BIGTIFF="YES"
    )


if __name__ == "__main__":
    main()
