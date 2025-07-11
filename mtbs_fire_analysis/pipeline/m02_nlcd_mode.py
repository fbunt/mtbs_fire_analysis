import raster_tools as rts

from mtbs_fire_analysis.pipeline.paths import (
    NLCD_MODE_RASTER_PATH,
    NLCD_STACK_VRT_PATH,
    get_nlcd_raster_path,
)
from mtbs_fire_analysis.utils import (
    protected_raster_save_with_cleanup,
    stack_rasters_as_vrt,
)


def main():
    nlcd_paths = [get_nlcd_raster_path(y) for y in range(1984, 2023)]
    stack_rasters_as_vrt(nlcd_paths, NLCD_STACK_VRT_PATH)
    nlcd_stack = rts.Raster(NLCD_STACK_VRT_PATH).astype("int16")
    mode = rts.general.local_stats(nlcd_stack, "mode")
    protected_raster_save_with_cleanup(mode, NLCD_MODE_RASTER_PATH)


if __name__ == "__main__":
    main()
