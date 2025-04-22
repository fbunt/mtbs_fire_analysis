import argparse
import shutil
from pathlib import Path

import dask
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

import raster_tools as rts
import utils
from paths import RESULTS_DIR

dask.config.set(
    {
        "distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0,
        "distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_": 0,
    }
)


FREQ_OUT_DF_FMT = "count_{aoi}.pqt"
FREQ_RASTER_FMT = "count_{aoi}.tif"


def _path(path):
    path = Path(path)
    assert path.exists()
    return path


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("aoi", help="Area of interest code (e.g. MT, CONUS, etc)")
    p.add_argument("input_path", type=_path, help="The input dataframe path")
    p.add_argument("like_path", type=_path, help="Path to like raster")
    return p


def _set_index(part, *sizes, partition_info=None):
    idx = partition_info["number"]
    sizes = [0] + list(sizes)
    starts = np.cumsum(sizes)
    start = starts[idx]
    stop = starts[idx + 1]
    index = pd.RangeIndex(start, stop)
    part = part.copy()
    part.index = index
    return part


def main(aoi, input_path, like_path):
    with LocalCluster(n_workers=2) as cluster, Client(cluster) as client:
        tmp_loc1 = RESULTS_DIR / "count_tmp1.pqt"
        if not tmp_loc1.exists():
            ddf = dgpd.read_parquet(input_path)
            ddf.spartial_partitions = None
            groups = ddf.groupby("geohash")
            count = groups.Ig_Date.count().astype("uint16").to_frame("count")
            # Move "geohash" from index to column
            count = count.reset_index()
            print(f"Saving to {tmp_loc1}")
            count.to_parquet(tmp_loc1)
            ddf = None

            client.restart()

        count = dd.read_parquet(tmp_loc1)
        n = len(count)
        print(f"N Data Points: {n}")

        tmp_loc2 = RESULTS_DIR / "count_tmp2.pqt"
        if not tmp_loc2.exists():
            n_parts = n // 1_000_000
            count = (
                count.reset_index()
                .repartition(npartitions=n_parts)
                .sort_values("geohash")
            )
            print(f"Saving to {tmp_loc2}")
            count.to_parquet(tmp_loc2)

            client.restart()

        final_count_loc = RESULTS_DIR / FREQ_OUT_DF_FMT.format(aoi=aoi)
        if not final_count_loc.exists():
            count = dd.read_parquet(tmp_loc2)
            assert count["geohash"].is_monotonic_increasing.compute()
            sizes = [p.shape[0] for p in count.partitions]
            count = count.map_partitions(
                _set_index, *sizes, clear_divisions=True, meta=count._meta
            )
            print(f"Saving to {final_count_loc}")
            count.to_parquet(final_count_loc)

        shutil.rmtree(tmp_loc1)
        shutil.rmtree(tmp_loc2)
    like = rts.Raster(like_path)
    count = utils.add_geometry_from_geohash(
        dd.read_parquet(final_count_loc, calculate_divisions=True)
    )
    count_vec = rts.Vector(count, n)
    count_raster = rts.rasterize.rasterize(count_vec, like, field="count")
    with ProgressBar():
        count_raster.save(str(RESULTS_DIR / FREQ_RASTER_FMT.format(aoi=aoi)))


if __name__ == "__main__":
    args = _get_parser().parse_args()
    main(args.aoi, args.input_path, args.like_path)
