import os

import rasterio as rio
from affine import Affine
from odc.geo.geobox import GeoBox

# --- Analysis-grid resolution selector -------------------------------------
# The native analysis grid is 30 m CONUS Albers. A coarser run (e.g. 120 m for
# cheap iteration) shares this grid's top-left origin and aggregates k x k
# native cells, where k = pixel_m / 30. The resolution is selected per process
# via the FIRE_PIXEL_M env var; unset => 30 m, byte-identical to the historical
# grid, so all existing upstream-mtbs users are unaffected (env-hook, not a
# default swap). See docs/workstreams/120m-resolution.md: a 120 m run pairs
# FIRE_PIXEL_M=120 with the Option-C FIRE_RESULTS_DIR=...120m output root.
BASE_PIXEL_M = 30
_BASE_GEOHASH_AFFINE = Affine(
    30.0, 0.0, -2406135.0, 0.0, -30.0, 3222585.0, 0.0, 0.0, 1.0
)
_BASE_GEOHASH_GRID_SHAPE = (100150, 157144)


def grid_for_pixel_m(pixel_m):
    """``(affine, (height, width))`` for the analysis grid at ``pixel_m`` m.

    The coarse grid shares the native 30 m grid's top-left origin and uses
    floor-division for the shape (matching the m01b/m01c DEM-aggregation
    convention), so a partial edge cell at the bottom/right is dropped rather
    than padded. ``pixel_m`` must be a positive integer multiple of 30 m
    (a clean k x k aggregation of native cells); ``30`` returns the native
    grid unchanged.
    """
    if pixel_m <= 0 or pixel_m % BASE_PIXEL_M != 0:
        raise ValueError(
            f"pixel_m must be a positive integer multiple of "
            f"{BASE_PIXEL_M} m (got {pixel_m!r})"
        )
    factor = pixel_m // BASE_PIXEL_M
    a = _BASE_GEOHASH_AFFINE
    affine = Affine(a.a * factor, a.b, a.c, a.d, a.e * factor, a.f)
    h, w = _BASE_GEOHASH_GRID_SHAPE
    return affine, (h // factor, w // factor)


def pixel_m_from_env():
    """Per-process analysis resolution (metres) from ``FIRE_PIXEL_M``.

    Unset/empty => ``BASE_PIXEL_M`` (30), so the historical 30 m behaviour is
    preserved exactly when the env var is absent.
    """
    raw = os.environ.get("FIRE_PIXEL_M")
    if not raw:
        return BASE_PIXEL_M
    try:
        pixel_m = int(raw)
    except ValueError as e:
        raise ValueError(
            f"FIRE_PIXEL_M must be an integer number of metres, got {raw!r}"
        ) from e
    grid_for_pixel_m(pixel_m)  # validate (raises on non-multiple-of-30)
    return pixel_m


PIXEL_M = pixel_m_from_env()
DEFAULT_GEOHASH_AFFINE, DEFAULT_GEOHASH_GRID_SHAPE = grid_for_pixel_m(PIXEL_M)
# Taken from MTBS raster files
DEFAULT_CRS = rio.CRS.from_wkt(
    """
PROJCRS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",
    BASEGEOGCRS["NAD83",
        DATUM["North American Datum 1983",
            ELLIPSOID["GRS 1980",6378137,298.257222101004,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["degree",0.0174532925199433]],
        ID["EPSG",4269]],
    CONVERSION["unnamed",
        METHOD["Albers Equal Area",
            ID["EPSG",9822]],
        PARAMETER["Latitude of false origin",23,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8821]],
        PARAMETER["Longitude of false origin",-96,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8822]],
        PARAMETER["Latitude of 1st standard parallel",29.5,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8823]],
        PARAMETER["Latitude of 2nd standard parallel",45.5,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8824]],
        PARAMETER["Easting at false origin",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8826]],
        PARAMETER["Northing at false origin",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8827]]],
    CS[Cartesian,2],
        AXIS["easting",east,
            ORDER[1],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
        AXIS["northing",north,
            ORDER[2],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]]]
"""
)
DEFAULT_GEOHASH_GEOBOX = GeoBox(
    DEFAULT_GEOHASH_GRID_SHAPE, DEFAULT_GEOHASH_AFFINE, DEFAULT_CRS
)


def geobox_for_pixel_m(pixel_m):
    """``GeoBox`` for the grid at ``pixel_m`` m (CRS = ``DEFAULT_CRS``).

    The explicit per-resolution counterpart of ``DEFAULT_GEOHASH_GEOBOX``
    (which follows ``FIRE_PIXEL_M``); use it where a builder needs a target
    grid for a resolution other than the process default.
    """
    affine, shape = grid_for_pixel_m(pixel_m)
    return GeoBox(shape, affine, DEFAULT_CRS)

# Allows for some compression gains
DEFAULT_PROX_MAX_DIST = 70_000
