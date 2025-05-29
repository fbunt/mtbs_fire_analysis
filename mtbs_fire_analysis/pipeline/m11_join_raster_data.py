import argparse
import time
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import polars as pl
import raster_tools as rts
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from mtbs_fire_analysis.geohasher import GridGeohasher
from mtbs_fire_analysis.pipeline.paths import get_mtbs_raster_path


def parallel_join(points, other, nparts=10):
    print(f"Joining {len(points)} x {len(other)}")
    points = dd.from_pandas(points, npartitions=nparts)
    other = dd.from_pandas(other, npartitions=nparts)
    return points.join(other, on="geohash", how="inner").compute()


def polars_join(points, other):
    start = time.time()
    points = pl.from_pandas(points)
    other_points = pl.from_pandas(other)
    points = points.join(other_points, on="geohash").to_pandas()
    d = time.time() - start
    print(f"Polars join time: {d // 60}min, {d % 60:.2f}s")
    return points


def _add_raster(points, raster_fetcher, name):
    print(f"Adding {name}")
    print("Loading raster as points...")
    other_points = (
        raster_fetcher()
        .to_points()
        .drop(["band", "row", "col"], axis=1)
        .rename(columns={"value": name})
        .compute()
    )
    print("Done")
    print(f"{len(other_points) = :,}")
    hasher = GridGeohasher()
    other_points["geohash"] = hasher.geohash(other_points.geometry)
    other_points = other_points.drop("geometry", axis=1)
    points = polars_join(points, other_points)
    return points


TARGET_POINTS_PER_PARTITION = 1_200_000


def _get_raster_fetcher(path, mtbs_path):
    def fetcher():
        mtbs = rts.Raster(str(mtbs_path))
        return rts.clipping.clip_box(rts.Raster(path), mtbs.bounds).set_null(
            rts.Raster(mtbs.xmask)
        )

    return fetcher


def join_raster_to_frame(name, raster_path, frame_path, mtbs_path, out_path):
    # Use a fetcher function so we can keep the result anonymous. This allows
    # it to be GC'd ASAP. If any references to a dask graph are kept in scope
    # after computing, a large portion of memory will also be kept by dask. I
    # am probably being overly cautious but I have been bitten by dask's memory
    # issues too many times.
    fetcher = _get_raster_fetcher(raster_path, mtbs_path)
    print("Loading dataframe")
    points = dd.read_parquet(frame_path).compute()
    print(f"Size initial: {len(points):,}. Loss/gain: {(len(points) - 0):+,}")
    n = len(points)

    points = _add_raster(points, fetcher, name)
    print(
        f"Size after join({name}): {len(points):,}"
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)

    # Convert to dask dataframe for saving in parallel
    nparts = max(int(np.round(n / TARGET_POINTS_PER_PARTITION)), 1)
    points = dd.from_pandas(points, npartitions=nparts)
    points.to_parquet(out_path)


def main(name, frame_files, raster_files):
    assert len(frame_files) == len(raster_files)
    frame_files = sorted(frame_files)
    raster_files = sorted(raster_files)
    years = range(1984, 1984 + len(frame_files))
    for year, frame_path, raster_path in zip(
        years, frame_files, raster_files, strict=True
    ):
        print(f"---\nProcessing: {year}")
        mtbs_path = get_mtbs_raster_path(year, "CONUS")
        out_path = frame_path.parent / (frame_path.stem + f"_{name}.pqt")
        if out_path.exists():
            print(f"{out_path} already exists. Skipping.")
            continue
        with ProgressBar():
            join_raster_to_frame(
                name, raster_path, frame_path, mtbs_path, out_path
            )


def _path(path):
    path = Path(path)
    assert path.exists()
    return path


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "name", type=str, help="The name to use for the raster data."
    )
    p.add_argument(
        "-f",
        "--frame-files",
        type=_path,
        nargs="+",
        help="The dataframe files to join the raster data into.",
    )
    p.add_argument(
        "-r",
        "--raster-files",
        type=_path,
        nargs="+",
        help="The raster files to join into frame_files.",
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    main(args.name, args.frame_files, args.raster_files)
