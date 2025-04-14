import argparse
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
from paths import (
    CLEANED_RASTER_DATA_DIR,
    ECO_REGIONS_PATH,
    PERIMS_PATH,
    RESULTS_DIR,
    ROOT_TMP_DIR,
    STATES_PATH,
)
import utils

dask.config.set(
    {
        "distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0,
        "distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_": 0,
    }
)

DATA_TIF_FMT = "mtbs_{aoi}_{year}.tif"
TMP_PTS_FMT = "mtbs_{aoi}_{year}"
COMBINED_OUT_FMT = "mtbs_{aoi}_{min_year}_{max_year}"


def _path(path):
    path = Path(path)
    assert path.exists()
    return path


DESC = """
Convert MTBS rasters to a large dataframe of points.


This is done in two steps. First, the raster for each year is converted to a
parquet dataframe on disk. Then, the dataframes are combined into one large
dataframe and saved as a large parquet file(s).
"""


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
        "--crs_file",
        default=CLEANED_RASTER_DATA_DIR / "mtbs_CONUS_1984.tif",
        type=_path,
        help="File to pull the CRS from",
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


def get_aoi_geom(aoi_code, crs):
    geoms = gpd.read_file(STATES_PATH)
    if aoi_code in ("ALL", "CONUS"):
        geoms = geoms[geoms.GEOID.astype(int) < 60]
        if aoi_code == "CONUS":
            geoms = geoms[~geoms.STUSPS.isin(("AK", "HI"))]
    else:
        geoms = geoms[aoi_code == geoms.STUSPS]
    return gpd.GeoSeries([geoms.geometry.to_crs(crs).union_all()], crs=crs)


def get_mtbs_perims_by_year_and_aoi(year, aoi_poly, crs):
    perims = gpd.read_parquet(PERIMS_PATH).sort_values("Ig_Date")
    perims = perims[perims.Ig_Date.dt.year == year]
    perims = perims.to_crs(crs)
    return perims[perims.intersects(aoi_poly)]


def get_eco_regions_by_aoi(aoi_poly, crs):
    eco_regions = gpd.read_file(ECO_REGIONS_PATH).to_crs(crs)
    eco_regions = eco_regions[eco_regions.intersects(aoi_poly.buffer(50_000))]
    return eco_regions


def get_data_raster_path(year, aoi_code):
    return CLEANED_RASTER_DATA_DIR / DATA_TIF_FMT.format(
        aoi=aoi_code, year=year
    )


def get_points_path(year, aoi_code):
    return ROOT_TMP_DIR / TMP_PTS_FMT.format(aoi=aoi_code, year=year)


def get_points_combined_path(years, aoi_code):
    return RESULTS_DIR / COMBINED_OUT_FMT.format(
        aoi=aoi_code, min_year=np.min(years), max_year=np.max(years)
    )


TARGET_POINTS_PER_PARTITION = 1_200_000


def parallel_sjoin(points, other, nparts=10):
    print(f"Joining {len(points)} x {len(other)}")
    if len(points) > 50_000:
        print("Performing parallel join")
        points = dgpd.from_geopandas(points, npartitions=nparts)
        with ProgressBar():
            points = points.sjoin(other, how="inner").compute()
    else:
        print("Performing serial join")
        points = gpd.sjoin(points, other, how="inner")
    return points


def _recover_orphans(points_pre, points_post, eco_regions):
    orphans = points_pre[~points_pre.index.isin(points_post.index)]
    orphans = orphans.sjoin_nearest(eco_regions, how="inner")
    return pd.concat([points_post, orphans])


def _join_with_eco_regions(points, eco_regions):
    print(f"Joining {len(points)} x {len(eco_regions)}")
    n_pre = len(points)
    points_pre = points
    points = parallel_sjoin(points, eco_regions, 20)
    n_post = len(points)
    if n_pre != n_post:
        print(f"Recovering {n_pre - n_post} orphaned points")
        points = _recover_orphans(points_pre, points, eco_regions)
    return points.drop("index_right", axis=1)


def _save_raster_to_points(raster_path, out_path, year, perims, eco_regions):
    raster = rts.Raster(str(raster_path))
    points = raster.to_points().compute()
    # Move index to column and rename to "geohash". The index created by
    # to_points is the flat index in the original array. Use  this rather than
    # dask_geopandas' geohash or hilbert_distance functions because they do not
    # have the resolution to handle rasters of this size (i.e. CONUS scale @
    # 30m).
    points = points.reset_index(names="geohash")
    points = points.drop(["band", "row", "col"], axis=1).rename(
        {"value": "bs"}, axis=1
    )
    # Drop non-severity pixels
    points = points[points.bs < 5]
    points["year"] = np.array(year, dtype="uint16")
    xhalf, yhalf = np.abs(raster.resolution) / 2
    points["cell_box"] = points.geometry.buffer(xhalf, cap_style="square")
    points = points.set_geometry("cell_box")
    points = parallel_sjoin(points, perims, 20)
    points = points.rename({"index_right": "perim_index"}, axis=1)
    assert "perim_index" in points.columns
    assert "index_right" not in points.columns
    # Drop the cell_box column and set the point values back as the geometry
    # column.
    geometry = points["geometry"].rename()
    points = points.drop(["geometry", "cell_box"], axis=1)
    points = gpd.GeoDataFrame(points, geometry=geometry)
    points = _join_with_eco_regions(points, eco_regions)
    geometry = points["geometry"]
    # Drop geometries to avoid dask_geopandas (bugs)
    points = points.drop("geometry", axis=1)
    # Set geohash to be flat index for reference grid defined by the
    # geotransform above.
    hasher = utils.GridGeohasher()
    points["geohash"] = hasher.geohash(geometry)
    npoints = len(points)
    nparts = max(int(np.round(npoints / TARGET_POINTS_PER_PARTITION)), 1)
    points = dd.from_pandas(points, npartitions=nparts)
    # dask_geopandas is currently bugged. Spatial partitions will randomly fail
    # to load later in the pipeline
    # points.spatial_partitions = None
    points.to_parquet(out_path)


def save_raster_to_points(years, aoi_code, crs):
    aoi_gs = get_aoi_geom(aoi_code, crs)
    for year in years:
        raster_path = get_data_raster_path(year, aoi_code)
        if not raster_path.exists():
            print(f"---\nNo raster file. Skipping: {year}")
            continue
        pts_path = get_points_path(year, aoi_code)
        if pts_path.exists():
            print(f"---\nSkipping: {year}")
            continue
        perims = get_mtbs_perims_by_year_and_aoi(year, aoi_gs.values[0], crs)
        eco_regions = get_eco_regions_by_aoi(aoi_gs.values[0], crs)
        print(f"---\nPreprocessing: {year}")
        with ProgressBar():
            _save_raster_to_points(
                raster_path, pts_path, year, perims, eco_regions
            )
        print("Done")


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
                # ddfs.append(dgpd.read_parquet(pts_path))
                ddfs.append(dd.read_parquet(pts_path))
        ddf = dd.concat(ddfs)
        # ddf.spatial_partitons = None
        ddf.to_parquet(get_points_combined_path(years, aoi_code))


def main(args):
    years = list(range(args.min_year, args.max_year + 1))
    aoi = args.aoi
    crs = rts.Raster(args.crs_file).crs
    num_workers = args.num_workers
    save_raster_to_points(years, aoi, crs)
    print("\n****\n")
    combine_years(years, aoi, num_workers)
    print("\n****\n")


if __name__ == "__main__":
    args = _get_parser().parse_args()
    assert args.min_year <= args.max_year
    assert args.num_workers > 0
    main(args)
