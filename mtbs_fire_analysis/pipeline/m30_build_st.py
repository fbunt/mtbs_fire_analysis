import argparse
import pprint
import subprocess
from pathlib import Path

import numpy as np
import raster_tools as rts
from dask.diagnostics import ProgressBar

from mtbs_fire_analysis.pipeline.paths import PERIMS_RASTERS_PATH, ST_PATH
from mtbs_fire_analysis.utils import protected_raster_save_with_cleanup


def _get_vrt_path(start_year, end_year):
    return PERIMS_RASTERS_PATH / f"dse_{start_year}_{end_year}.vrt"


def _get_dse_max_path(end_year):
    return PERIMS_RASTERS_PATH / f"dse_max_{end_year}.tif"


def build_st(end_year):
    raster = rts.Raster(_get_dse_max_path(end_year - 1))
    # raster = rts.Raster(f"mtbs_perims/rasters/dse_max_{end_year - 1}.tif")
    data = raster.data
    data = data.map_blocks(
        st_chunk,
        dtype="float32",
        meta=np.array((), dtype="float32"),
        end_year=end_year,
    )
    st = rts.data_to_raster_like(data, like=raster, nv=None)
    path = f"st/st_{end_year}.tif"
    print(f"Saving to {path}")
    with ProgressBar():
        st.save(path)


def stack_dse_years_as_vrt(start_year, end_year):
    # end_year is inclusive here.
    paths = [
        Path(PERIMS_RASTERS_PATH / f"dse_{y}.tif")
        for y in range(start_year, end_year + 1)
    ]
    for p in paths:
        assert p.exists(), f"{p} does not exist"
    paths = map(str, paths)
    out_path = _get_vrt_path(start_year, end_year)
    if out_path.exists():
        out_path.unlink()
    # We could do this by importing gdal but rasterio STRONGLY recommends not
    # doing so because it will probably break rasterio. Thus we do it through
    # a shell call.
    # See: https://rasterio.readthedocs.io/en/stable/topics/switch.html#mutual-incompatibilities
    command = ["gdalbuildvrt", "-separate", str(out_path)]
    command.extend(paths)
    print(f"Build VRT command:\n{pprint.pformat(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as err:
        print(f"STDOUT\n------\n{err.stdout}")
        print(f"STDERR\n------\n{err.stderr}")
        raise err


def calc_dse_max(start_year, end_year):
    # end_year is inclusive here.
    print(f"Computing DSE max for {start_year}-{end_year}")
    prev_result = _get_dse_max_path(end_year - 1)
    raster_to_max = None
    if prev_result.exists():
        # Reuse the result from previous year, if available. Cuts down on
        # computation time.
        raster_to_max = rts.band_concat(
            [prev_result, PERIMS_RASTERS_PATH / f"dse_{end_year}.tif"]
        )
        print(f"Reusing DSE max from {end_year - 1}.")
    else:
        # Compute the max over the full range of rasters
        raster_to_max = rts.Raster(_get_vrt_path(start_year, end_year))
    data = raster_to_max.data
    data = np.max(data, axis=0)
    dse_max = rts.data_to_raster_like(
        data, like=raster_to_max, nv=raster_to_max.null_value
    )
    out_path = _get_dse_max_path(end_year)
    print(f"Saving to {out_path}")
    protected_raster_save_with_cleanup(dse_max, out_path)


def st_chunk(x, start_year, end_year):
    # x is the number of days since linux epoch
    mask = x == -1
    # Convert to date
    x = np.datetime64("1970-01-01") + x.astype("timedelta64[D]")
    # Days until Jan 1, {end_year}
    x = np.datetime64(f"{end_year}-01-01") - x
    # Keep as integer days to avoid float32 precision issues
    x = x.astype("int16")
    x[mask] = (end_year - start_year) * 365
    return x


def calc_st_for_end_year(end_year):
    dse_raster = rts.Raster(_get_dse_max_path(end_year - 1))
    data = dse_raster.data
    data = data.map_blocks(
        st_chunk,
        dtype="float32",
        meta=np.array((), dtype="float32"),
        start_year=1984,
        end_year=end_year,
    )
    st = rts.data_to_raster_like(data, like=dse_raster, nv=None)
    out_path = ST_PATH / f"st_{end_year}.tif"
    print(f"Saving ST to {out_path}")
    protected_raster_save_with_cleanup(st, out_path)


def main(end_years):
    for end_year in end_years:
        stack_dse_years_as_vrt(1984, end_year - 1)
        calc_dse_max(1984, end_year - 1)
        calc_st_for_end_year(end_year)


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "end_years", type=int, nargs="+", help="End years to compute ST for"
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    end_years = args.end_years
    for year in end_years:
        assert year > 1984
    main(end_years)
