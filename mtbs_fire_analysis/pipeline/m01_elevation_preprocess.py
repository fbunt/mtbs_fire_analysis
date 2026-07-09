import geopandas as gpd
import raster_tools as rts
from dask.distributed import Client, LocalCluster

from mtbs_fire_analysis.defaults import DEFAULT_GEOHASH_GEOBOX
from mtbs_fire_analysis.pipeline.paths import (
    ASPECT_PATH,
    ASPECT_RAW_PATH,
    ELEVATION_PATH,
    ELEVATION_RAW_PATH,
    SLOPE_PATH,
    SLOPE_RAW_PATH,
    STATES_PATH,
)
from mtbs_fire_analysis.utils import protected_raster_save_with_cleanup


def process_raster(in_path, out_path):
    raster_raw = rts.Raster(in_path).set_null_value(-99_999)
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
        elevation, out_path, progress=False, BIGTIFF="YES"
    )


def main():
    cluster = LocalCluster()
    client = Client(cluster)

    print("ELEVATION")
    process_raster(ELEVATION_RAW_PATH, ELEVATION_PATH)
    print("ASPECT")
    process_raster(ASPECT_RAW_PATH, ASPECT_PATH)
    print("SLOPE")
    process_raster(SLOPE_RAW_PATH, SLOPE_PATH)


if __name__ == "__main__":
    main()
