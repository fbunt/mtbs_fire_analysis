import argparse
from pathlib import Path

import dask
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import time
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

import raster_tools as rts
import utils
from paths import (
    CLEANED_RASTER_DATA_DIR,
    ECO_REGIONS_PATH,
    NLCD_PATH,
    PERIMS_PATH,
    RESULTS_DIR,
    ROOT_TMP_DIR,
    STATES_PATH,
    WUI_PATH,
)

dask.config.set(
    {
        "distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0,
        "distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_": 0,
    }
)

DATA_TIF_FMT = "mtbs_{aoi}_{year}.tif"
NLCD_TIF_FMT = "Annual_NLCD_LndCov_{year}_CU_C1V0.tif"
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


def get_mtbs_raster_path(year, aoi_code):
    return CLEANED_RASTER_DATA_DIR / DATA_TIF_FMT.format(
        aoi=aoi_code, year=year
    )


def get_nlcd_raster_path(year):
    return NLCD_PATH / NLCD_TIF_FMT.format(year=year)


def get_points_path(year, aoi_code):
    return ROOT_TMP_DIR / TMP_PTS_FMT.format(aoi=aoi_code, year=year)


def _get_wui_path(year, flavor):
    if year < 2000:
        y = 1990
    elif 2000 <= year < 2010:
        y = 2000
    elif 2010 <= year < 2020:
        y = 2010
    else:
        y = 2020
    return WUI_PATH / f"wui_{flavor}_{y}.tif"


def get_wui_flag_path(year):
    return _get_wui_path(year, "flag")


def get_wui_class_path(year):
    return _get_wui_path(year, "class")


def get_points_combined_path(years, aoi_code):
    return RESULTS_DIR / COMBINED_OUT_FMT.format(
        aoi=aoi_code, min_year=np.min(years), max_year=np.max(years)
    )


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
    print("Adding eco regions")
    n_pre = len(points)
    points_pre = points
    points = parallel_sjoin(points, eco_regions, 20)
    n_post = len(points)
    if n_pre != n_post:
        print(f"Recovering {n_pre - n_post:,} orphaned points")
        points = _recover_orphans(points_pre, points, eco_regions)
    return points.drop("index_right", axis=1)


def parallel_join(points, other, nparts=10):
    print(f"Joining {len(points)} x {len(other)}")
    points = dd.from_pandas(points, npartitions=nparts)
    other = dd.from_pandas(other, npartitions=nparts)
    return points.join(other, on="geohash", how="inner").compute()


def _drop_duplicates(points):
    print("Dropping co-located points with same Ig_Date")
    start = time.time()
    points = pl.from_pandas(points)
    # Tack on Event_ID so that we can deterministically choose which event to
    # keep and which to drop.
    points = points.sort("geohash", "Ig_Date", "Event_ID")
    # Keep the last event as determined by Event_ID order
    points = points.unique(subset=["geohash", "Ig_Date"], keep="last")
    points = points.to_pandas()
    d = time.time() - start
    print(f"{d // 60}min, {d % 60:.2f}s")
    return points


def _add_raster(points, raster, name):
    print(f"Adding {name}")
    start = time.time()
    other_points = (
        raster.to_points()
        .drop(["band", "row", "col"], axis=1)
        .rename(columns={"value": name})
        .compute()
    )
    print(f"{len(other_points) = :,}")
    hasher = utils.GridGeohasher()
    other_points["geohash"] = hasher.geohash(other_points.geometry)
    other_points = other_points.drop("geometry", axis=1)
    # return parallel_join(points, other_points, 20)
    points = pl.from_pandas(points)
    other_points = pl.from_pandas(other_points)
    points = points.join(other_points, on="geohash").to_pandas()
    d = time.time() - start
    print(f"{d // 60}min, {d % 60:.2f}s")
    return points


def _get_initial_points(mtbs_path):
    mtbs = rts.Raster(str(mtbs_path))
    points = mtbs.to_points().compute()
    points = points.reset_index(drop=True)
    return points.drop(["band", "row", "col"], axis=1).rename(
        {"value": "bs"}, axis=1
    )


def _get_nlcd_raster(nlcd_path, mtbs_path):
    mtbs = rts.Raster(str(mtbs_path))
    return rts.Raster(nlcd_path).set_null(rts.Raster(mtbs.xmask))


def _get_wui_raster(wui_path, mtbs_path):
    raster = rts.Raster(str(mtbs_path))
    return rts.clipping.clip_box(rts.Raster(wui_path), raster.bounds).set_null(
        rts.Raster(raster.xmask)
    )


TARGET_POINTS_PER_PARTITION = 1_200_000


def _build_dataframe_and_save(
    mtbs_path,
    nlcd_path,
    wui_flag_path,
    wui_class_path,
    out_path,
    year,
    perims,
    eco_regions,
):
    points = _get_initial_points(mtbs_path)
    print(f"Size initial: {len(points):,}. Loss/gain: {(len(points) - 0):+,}")
    n = len(points)

    # Drop non-severity pixels
    points = points[points.bs < 5]
    print(
        f"Size after drop(bs < 5): {len(points):,}."
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)

    # Create boxes representing pixels and perform a spatial join with the MTBS
    # fire perimeter vectors
    points["year"] = np.array(year, dtype="uint16")
    xhalf, yhalf = np.abs(rts.Raster(str(mtbs_path)).resolution) / 2
    points["cell_box"] = points.geometry.buffer(xhalf, cap_style="square")
    points = points.set_geometry("cell_box")
    points = parallel_sjoin(points, perims, 20)
    points = points.rename({"index_right": "perim_index"}, axis=1)
    assert "perim_index" in points.columns
    assert "index_right" not in points.columns
    print(
        f"Size after join(perims): {len(points):,}."
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)

    # Join with eco regions vectors
    geometry = points["geometry"].rename()
    points = points.drop(["geometry", "cell_box"], axis=1)
    points = gpd.GeoDataFrame(points, geometry=geometry)
    points = _join_with_eco_regions(points, eco_regions)
    print(
        f"Size after join(eco_regions): {len(points):,}."
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)

    # Geohash the grid points so we can drop geometry data entirely
    geometry = points["geometry"]
    # Drop geometries to avoid dask_geopandas (bugs)
    points = points.drop("geometry", axis=1)
    hasher = utils.GridGeohasher()
    points["geohash"] = hasher.geohash(geometry)
    geometry = None

    points = _drop_duplicates(points)
    print(
        f"Size after drop(duplicated(geohash, Ig_Date)): {len(points):,}."
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)

    points = _add_raster(
        points, _get_nlcd_raster(nlcd_path, mtbs_path), "nlcd"
    )
    print(
        f"Size after join(nlcd): {len(points):,}"
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)
    points = _add_raster(
        points, _get_wui_raster(wui_flag_path, mtbs_path), "wui_flag"
    )
    print(
        f"Size after join(wui_flag): {len(points):,}"
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)
    points = _add_raster(
        points, _get_wui_raster(wui_flag_path, mtbs_path), "wui_class"
    )
    print(
        f"Size after join(wui_class): {len(points):,}"
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)

    # Convert to dask dataframe for saving in parallel
    nparts = max(int(np.round(n / TARGET_POINTS_PER_PARTITION)), 1)
    points = dd.from_pandas(points, npartitions=nparts)
    points.to_parquet(out_path)


def save_raster_to_points(years, aoi_code, crs):
    aoi_gs = get_aoi_geom(aoi_code, crs)
    for year in years:
        mtbs_path = get_mtbs_raster_path(year, aoi_code)
        nlcd_path = get_nlcd_raster_path(year)
        wui_flag_path = get_wui_flag_path(year)
        wui_class_path = get_wui_class_path(year)
        if not mtbs_path.exists():
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
            _build_dataframe_and_save(
                mtbs_path,
                nlcd_path,
                wui_flag_path,
                wui_class_path,
                pts_path,
                year,
                perims,
                eco_regions,
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
