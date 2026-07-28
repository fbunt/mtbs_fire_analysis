import pprint
import subprocess

import geopandas as gpd
import raster_tools as rts
import rasterio as rio
from dask.distributed import Client, LocalCluster

from mtbs_fire_analysis.defaults import DEFAULT_GEOHASH_GEOBOX
from mtbs_fire_analysis.pipeline.paths import (
    ASPECT_PATH,
    ELEVATION_PATH,
    ELEVATION_RAW_PATH,
    SLOPE_PATH,
    STATES_PATH,
)
from mtbs_fire_analysis.utils import protected_raster_save_with_cleanup

NULL_VALUE = -99_999
# EDNA marked flat cells in its aspect raster with -1. gdaldem lumps them
# in with its null value instead, which would make them indistinguishable
# from cells outside the DEM, so they get converted back to -1.
FLAT_ASPECT = -1
# gdaldem's aspect output, before the flat cells have been split back out
ASPECT_GDALDEM_PATH = ASPECT_PATH.with_stem(f"{ASPECT_PATH.stem}_gdaldem")
# Match what protected_raster_save_with_cleanup() writes
GDALDEM_CREATION_OPTIONS = {
    "TILED": "YES",
    "COMPRESS": "ZSTD",
    "ZSTD_LEVEL": "1",
    "BIGTIFF": "YES",
    "NUM_THREADS": "ALL_CPUS",
}


def run_gdaldem(mode, in_path, out_path):
    """Derive `mode` from an already reprojected DEM using gdaldem.

    The derivatives have to be computed on the final grid. Reprojecting
    ready-made slope and aspect rasters resamples the derived values
    themselves, which badly distorts them.
    """
    if out_path.exists():
        print(f"{out_path} already exists. Skipping.")
        return

    # -compute_edges keeps the valid extent identical to the DEM's.
    # Without it, every cell that touches the clip boundary or a DEM void
    # comes out null.
    command = ["gdaldem", mode, "-compute_edges"]
    for name, value in GDALDEM_CREATION_OPTIONS.items():
        command.extend(["-co", f"{name}={value}"])
    command.extend([str(in_path), str(out_path)])
    print(f"gdaldem command:\n{pprint.pformat(command)}")
    # We could do this by importing gdal but rasterio STRONGLY recommends not
    # doing so because it will probably break rasterio. Thus we do it through
    # a shell call.
    # See: https://rasterio.readthedocs.io/en/stable/topics/switch.html#mutual-incompatibilities
    try:
        subprocess.run(command, check=True)
    except (Exception, KeyboardInterrupt) as err:
        print("Removing unfinished file")
        out_path.unlink(missing_ok=True)
        raise err


def set_band_name(path, name):
    """Name the band the way the raw EDNA rasters did.

    The names get dropped somewhere on the way through reproject/clip
    and gdaldem does not set one at all. rioxarray round-trips the band
    description through `long_name`, so set both to match the DEM. This
    only rewrites metadata, not pixels.
    """
    with rio.open(path, "r+") as ds:
        ds.set_band_description(1, name)
        ds.update_tags(long_name=name)


def process_elevation():
    raster_raw = rts.Raster(ELEVATION_RAW_PATH).set_null_value(NULL_VALUE)
    states = gpd.read_file(STATES_PATH)
    conus = states[states.NAME == "CONUS"].copy()
    raster_reprojected = raster_raw.reproject(
        DEFAULT_GEOHASH_GEOBOX, resample_method="cubic"
    )
    # Clip out data in Canada, Mexico, and the ocean
    elevation = rts.clipping.clip(
        conus, raster_reprojected, bounds=raster_reprojected.bounds
    )
    protected_raster_save_with_cleanup(
        elevation, ELEVATION_PATH, progress=False, BIGTIFF="YES"
    )


def process_slope():
    run_gdaldem("slope", ELEVATION_PATH, SLOPE_PATH)
    set_band_name(SLOPE_PATH, "slopedeg")


def process_aspect():
    if ASPECT_PATH.exists():
        print(f"{ASPECT_PATH} already exists. Skipping.")
        return

    run_gdaldem("aspect", ELEVATION_PATH, ASPECT_GDALDEM_PATH)
    aspect = rts.Raster(ASPECT_GDALDEM_PATH)
    assert aspect.null_value is not None, (
        f"{ASPECT_GDALDEM_PATH} has no null value. gdaldem should mark"
        " both flat cells and cells outside the DEM with one."
    )
    elevation = rts.Raster(ELEVATION_PATH)
    # Pull the flat cells back out of gdaldem's null mask and take the
    # null mask from the DEM instead.
    aspect = (
        aspect.where(~aspect.to_null_mask(), FLAT_ASPECT)
        .set_null_value(NULL_VALUE)
        .set_null(elevation.to_null_mask())
    )
    protected_raster_save_with_cleanup(
        aspect, ASPECT_PATH, progress=False, BIGTIFF="YES"
    )
    set_band_name(ASPECT_PATH, "aspect")
    ASPECT_GDALDEM_PATH.unlink()


def main():
    cluster = LocalCluster()
    _client = Client(cluster)

    print("ELEVATION")
    process_elevation()
    print("SLOPE")
    process_slope()
    print("ASPECT")
    process_aspect()


if __name__ == "__main__":
    main()
