import argparse
from pathlib import Path

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
        Client(cluster) as _,
    ):
        ddfs = []
        for year in years:
            pts_path = get_points_path(year, aoi_code)
            if pts_path.exists():
                ddfs.append(dd.read_parquet(pts_path))
        ddf = dd.concat(ddfs)
        ddf.to_parquet(get_points_combined_path(years, aoi_code))


def _path(path):
    path = Path(path)
    assert path.exists()
    return path


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
    p.add_argument("files", type=_path, nargs="+", help="Files to combine")
    p.add_argument("out_path", type=Path, help="Output path")
    return p


def main(files, out_path, num_workers):
    assert not out_path.exists()

    with (
        LocalCluster(n_workers=num_workers) as cluster,
        Client(cluster) as _,
    ):
        ddfs = [dd.read_parquet(path) for path in files]
        ddf = dd.concat(ddfs)
        ddf.to_parquet(out_path)


if __name__ == "__main__":
    args = _get_parser().parse_args()
    main(args.files, args.out_path, args.num_workers)
