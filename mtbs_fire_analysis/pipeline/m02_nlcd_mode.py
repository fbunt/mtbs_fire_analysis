import glob
from pathlib import Path

import raster_tools as rts

from mtbs_fire_analysis.pipeline.paths import (
    NLCD_MODE_RASTER_PATH,
    NLCD_PATH,
    NLCD_STACK_VRT_PATH,
)
from mtbs_fire_analysis.utils import (
    protected_raster_save_with_cleanup,
    stack_rasters_as_vrt,
)


def main():
    nlcd_paths = [Path(p) for p in sorted(glob.glob(str(NLCD_PATH / "*.tif")))]
    stack_rasters_as_vrt(nlcd_paths, NLCD_STACK_VRT_PATH)
    nlcd_stack = rts.Raster(NLCD_STACK_VRT_PATH)
    mode = rts.general.local_stats(nlcd_stack, "mode")
    protected_raster_save_with_cleanup(mode, NLCD_MODE_RASTER_PATH)


if __name__ == "__main__":
    main()
