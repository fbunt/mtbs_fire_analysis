import argparse

import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from paths import get_points_combined_path, get_points_path

# Try to keep dask from leaking memory
dask.config.set(
    {
        "distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0,
        "distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_": 0,
    }
)


def combine_years(years, aoi_code, num_workers):
    out_path = get_points_combined_path(years, aoi_code)
    if out_path.exists():
        print(
            f"Combined dataframe path '{out_path}' already present. Skipping."
        )
        return
    with (
        LocalCluster(n_workers=num_workers) as cluster,
        Client(cluster) as client,
    ):
        ddfs = []
        for year in years:
            pts_path = get_points_path(year, aoi_code)
            if pts_path.exists():
                ddfs.append(dd.read_parquet(pts_path))
        ddf = dd.concat(ddfs)
        ddf.to_parquet(get_points_combined_path(years, aoi_code))


DESC = """Combine dataframes to form a large dataframe of burn records."""


def _get_parser():
    p = argparse.ArgumentParser(description=DESC)
    p.add_argument(
        "-j",
        "--num_workers",
        type=int,
        default=5,
        help=(
            "Number of workers for Dask cluster when combining dataframes. "
            "Default is 5."
        ),
    )
    p.add_argument(
        "--min_year",
        default=1984,
        type=int,
        help="Minnimum year to pull data from",
    )
    p.add_argument(
        "--max_year",
        default=2022,
        type=int,
        help="Maximum year to pull data from",
    )
    p.add_argument(
        "aoi",
        type=str,
        help="The area of interest (AOI) code to pull points from",
    )
    return p
