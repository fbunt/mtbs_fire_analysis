import geopandas as gpd
import raster_tools as rts
from dask.distributed import Client, LocalCluster

from mtbs_fire_analysis.defaults import DEFAULT_GEOHASH_GEOBOX
from mtbs_fire_analysis.pipeline.paths import (
    ELEVATION_PATH,
    ELEVATION_RAW_PATH,
    STATES_PATH,
)
from mtbs_fire_analysis.utils import protected_raster_save_with_cleanup


def main():
    cluster = LocalCluster()
    client = Client(cluster)

    elevation_raw = rts.Raster(ELEVATION_RAW_PATH).set_null_value(-99_999)
    states = gpd.read_file(STATES_PATH)
    conus = states[states.NAME == "CONUS"].copy()
    elevation_reprojected = elevation_raw.reproject(
        DEFAULT_GEOHASH_GEOBOX, resample_method="cubic"
    )
    # Clip out data in Canada, Mexico, and the ocean
    elevation = rts.clipping.clip(
        conus, elevation_reprojected, bounds=elevation_reprojected.bounds
    )
    protected_raster_save_with_cleanup(
        elevation, ELEVATION_PATH, progress=False, BIGTIFF="YES"
    )


if __name__ == "__main__":
    main()
