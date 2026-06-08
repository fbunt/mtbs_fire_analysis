import os
import warnings

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

# --- Coarsening edge-drop guard --------------------------------------------
# grid_for_pixel_m floor-divides the base shape, so the partial bottom/right
# edge strip (cells that don't complete a full pixel_m x pixel_m block) is
# dropped rather than padded. For the blessed set {30,120,480,1920} that strip
# is <0.07% of CONUS (the all-nodata frame margin). This guard warns once per
# resolution if a coarsening ever drops more than 1% of the base-grid area --
# a signal that the coarse grid is clipping a non-trivial edge of real data and
# the resolution needs a look before it is trusted. (Requested 2026-06-08.)
_DROPPED_AREA_WARN_THRESHOLD = 0.01
_dropped_area_warned: "set[int]" = set()


def coarsening_dropped_area_fraction(pixel_m):
    """Fraction of the base 30 m grid area dropped by ``grid_for_pixel_m``'s
    floor-division at ``pixel_m``.

    The dropped cells are the partial bottom/right edge that does not complete
    a full ``pixel_m`` x ``pixel_m`` block. Returns ``0.0`` when ``pixel_m``
    divides the base shape exactly on both axes (always at 30 m).
    """
    if pixel_m <= 0 or pixel_m % BASE_PIXEL_M != 0:
        raise ValueError(
            f"pixel_m must be a positive integer multiple of "
            f"{BASE_PIXEL_M} m (got {pixel_m!r})"
        )
    factor = pixel_m // BASE_PIXEL_M
    h, w = _BASE_GEOHASH_GRID_SHAPE
    covered = (h // factor * factor) * (w // factor * factor)
    return 1.0 - covered / (h * w)


def _warn_if_dropped_area_exceeds(pixel_m):
    """Warn once per ``pixel_m`` if coarsening drops more than
    ``_DROPPED_AREA_WARN_THRESHOLD`` of the base-grid area."""
    if pixel_m in _dropped_area_warned:
        return
    frac = coarsening_dropped_area_fraction(pixel_m)
    if frac > _DROPPED_AREA_WARN_THRESHOLD:
        _dropped_area_warned.add(pixel_m)
        warnings.warn(
            f"grid_for_pixel_m({pixel_m} m): floor-division drops "
            f"{frac:.2%} of the base 30 m grid area "
            f"(> {_DROPPED_AREA_WARN_THRESHOLD:.0%}) -- the coarse grid clips "
            f"a non-trivial edge strip of CONUS. Verify that strip is nodata "
            f"frame, not real data, before trusting this resolution.",
            stacklevel=3,
        )


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
    _warn_if_dropped_area_exceeds(pixel_m)
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
