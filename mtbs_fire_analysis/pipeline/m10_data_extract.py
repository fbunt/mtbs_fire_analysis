import argparse
import os
import time

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pyarrow
import raster_tools as rts
from dask.diagnostics import ProgressBar

from mtbs_fire_analysis.defaults import (
    BASE_PIXEL_M,
    DEFAULT_CRS,
    PIXEL_M,
    geobox_for_pixel_m,
)
from mtbs_fire_analysis.geohasher import GridGeohasher
from mtbs_fire_analysis.pipeline.paths import (
    ECO_REGIONS_PATH,
    ELEVATION_PATH,
    HEX_GRID_PATH,
    NLCD_MODE_RASTER_PATH,
    PERIMS_PATH,
    PERIMS_RASTERS_PATH,
    STATES_PATH,
    get_mtbs_raster_path,
    get_nlcd_raster_path,
    get_points_path,
    get_wui_flavor_path,
)
from mtbs_fire_analysis.raster_read import sample_on_geobox


def _format_elapsed_time(elapsed):
    return f"{int(elapsed // 60)}min, {elapsed % 60:.2f}s"


def skip_wui() -> bool:
    """True iff the four WUI covariate joins should be skipped.

    Back-compat env-hook (default OFF -> WUI joined exactly as before, so
    behaviour is unchanged for every other consumer of this shared upstream
    script). When ``MTBS_SKIP_WUI`` is truthy, m10 omits the four WUI columns
    (``wui_flag``/``wui_class``/``wui_bool``/``wui_prox``). Nothing in the
    fire-interval four-types / fits path reads those columns (verified: zero
    ``wui`` references in a00/a20/observation_data/etl), so the downstream
    output is unaffected. Set it to run m10 when the WUI rasters are not
    staged on the analysis root -- e.g. the 120 m four-types gate run, which
    needs only fire timing, not covariates (phd-research, 2026-06-09).

    Read at call time (not import) so tests / runs can flip it per process.
    """
    return os.environ.get("MTBS_SKIP_WUI", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def wui_flavor_path_or_none(year, flavor):
    """``get_wui_flavor_path(year, flavor)`` unless WUI is skipped (->None)."""
    return None if skip_wui() else get_wui_flavor_path(year, flavor)


def get_conus_geom(crs):
    geoms = gpd.read_file(STATES_PATH).to_crs(crs)
    return geoms[geoms.STUSPS == "CONUS"].geometry


def get_states(crs):
    states = gpd.read_file(STATES_PATH).to_crs(crs)
    return states[["STUSPS", "geometry"]].rename({"STUSPS": "state"}, axis=1)


def get_mtbs_perims_raster_path(year):
    return PERIMS_RASTERS_PATH / f"dse_{year}.tif"


def get_mtbs_perims_by_year_and_aoi(year, aoi_poly, crs):
    perims = gpd.read_file(PERIMS_PATH).sort_values("Ig_Date")
    perims = perims[perims.Ig_Date.dt.year == year]
    perims = perims.to_crs(crs)
    return perims[perims.intersects(aoi_poly)]


def get_eco_regions_by_aoi(aoi_poly, crs):
    eco_regions = gpd.read_file(ECO_REGIONS_PATH).to_crs(crs)
    eco_regions = eco_regions[eco_regions.intersects(aoi_poly.buffer(50_000))]
    return eco_regions


def parallel_sjoin(points, other, nparts=10, predicate="within"):
    print(f"Joining {len(points)} x {len(other)}")
    if len(points) > 50_000:
        print("Performing parallel sjoin")
        points = dgpd.from_geopandas(points, npartitions=nparts)
        with ProgressBar():
            points = points.sjoin(
                other, how="inner", predicate=predicate
            ).compute()
    else:
        print("Performing serial sjoin")
        points = gpd.sjoin(points, other, how="inner", predicate=predicate)
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


def polars_join(points, other, how="inner"):
    if how is None:
        how = "inner"
    points = pl.from_pandas(points)
    other_points = pl.from_pandas(other)
    # Use pyarrow to avoid columns with nulls converting to floats
    return points.join(other_points, on="geohash", how=how).to_pandas(
        use_pyarrow_extension_array=True
    )


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
    print(_format_elapsed_time(d))
    return points


# Resolution-aware covariate read target. The covariate rasters are 30 m COGs;
# at a coarse FIRE_PIXEL_M the read MUST reduce them onto this analysis geobox
# (mode/average) BEFORE the positional geohash-ij index, else the 30 m raster
# is indexed with coarse-grid positions (the 2026-06-09 bug: a 4x misalignment
# that dropped 57% of burned pixels to off-CONUS nodata, mis-valuing the rest).
# None at 30 m -> the legacy native read (byte-identical).
_GEOBOX = None if PIXEL_M == BASE_PIXEL_M else geobox_for_pixel_m(PIXEL_M)
# `coverage` here is the fraction of the BURNED-pixel geohashes (this eco's
# query set) that sample a non-nodata covariate cell — NOT a CONUS-land
# fraction. It's a near-1.0 proxy for land coverage only because burned pixels
# are land and NLCD covers ~99.5% of CONUS land, so a real grid/resolution
# mismatch (the m10 2026-06-09 bug) collapses it far below 0.90 while genuine
# edge-nodata stays just under 1.0. Loud WARN here; the hard regression gate
# lives in fire_interval. (Denominator clarified per 2026-06-10 review.)
_COVERAGE_WARN = 0.90


def _add_raster(
    points, raster_path, name, idxs, *, resample_method="mode", how=None
):
    print(f"Adding {name}")
    start = time.time()
    # Resolution-correct read: at a coarse geobox the source is reduced onto it
    # (mode for categorical, average for continuous) BEFORE the geohash-ij
    # index, so positions are valid. At 30 m (_GEOBOX is None) this is
    # byte-identical to the legacy isel read.
    xvalues, coverage, nv = sample_on_geobox(
        raster_path,
        _GEOBOX,
        idxs,
        resample_method=resample_method,
        require_nodata=(_GEOBOX is not None),
    )
    if _GEOBOX is not None:
        print(f"  {name} coverage: {coverage:.4f}")
        if coverage < _COVERAGE_WARN:
            print(
                f"  WARNING: {name} coverage {coverage:.4f} < "
                f"{_COVERAGE_WARN} — fraction of burned-pixel queries hitting "
                "valid covariate; this low signals a grid/resolution "
                "mismatch, not real edge-nodata (burned pixels are land; "
                "NLCD covers ~99.5% of CONUS land)."
            )
    if nv is not None:
        # Drop any null values we picked up
        mask = ~rts.raster.get_mask_from_data(xvalues, nv)
        idxs = idxs[0][mask], idxs[1][mask]
        xvalues = xvalues[mask]
        mask = None
    hasher = GridGeohasher()
    other_points = pd.DataFrame(
        {name: xvalues, "geohash": hasher.geohash_from_ij(idxs)}
    )
    points = polars_join(points, other_points, how=how)
    d = time.time() - start
    print(_format_elapsed_time(d))
    return points


def _get_initial_points(path):
    raster = rts.Raster(path)
    points = (
        raster.to_points()
        .drop(["band", "row", "col", "value"], axis=1)
        .compute()
    )
    assert isinstance(points, gpd.GeoDataFrame)
    points = points.reset_index(drop=True)
    return points


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
    perims_raster_path,
    mtbs_path,
    nlcd_path,
    wui_flag_path,
    wui_class_path,
    wui_bool_path,
    wui_prox_path,
    elevation_path,
    out_path,
    year,
    perims,
    states,
    eco_regions,
    hex_grid,
    drop_extra,
):
    points = _get_initial_points(perims_raster_path)
    print(f"Size initial: {len(points):,}. Loss/gain: {(len(points) - 0):+,}")
    n = len(points)

    # Create boxes representing pixels and perform a spatial join with the MTBS
    # fire perimeter vectors
    points["year"] = np.array(year, dtype="uint16")
    points = parallel_sjoin(points, perims, 20)
    # Derive `perim_index` from the perims-join index (the perimeter's row in
    # the cleaned gpkg). a00 / build_event_histories REQUIRE this column, but
    # the consolidated `mtbs_perims_trimmed.gpkg` dropped it and m10 used to
    # just discard the sjoin `index_right`. Restore it (back-compat: keep an
    # existing `perim_index` if a future gpkg supplies one). 2026-06-09.
    if "perim_index" in points.columns:
        points = points.drop(columns="index_right")
    else:
        points = points.rename(columns={"index_right": "perim_index"})
    assert "index_right" not in points.columns
    print(
        f"Size after join(perims): {len(points):,}."
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)
    points = parallel_sjoin(points, states, 20)
    points = points.drop(columns="index_right")
    assert "index_right" not in points.columns
    print(
        f"Size after join(states): {len(points):,}."
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)

    # Join with eco regions vectors
    points = _join_with_eco_regions(points, eco_regions)
    print(
        f"Size after join(eco_regions): {len(points):,}."
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)
    if drop_extra:
        extra_cols = [
            "Asmnt_Type",
            "area_acres",
            "days_since_epoch",
            "dNBR_offst",
            "dNBR_stdDv",
            "IncGreen_T",
            "Low_T",
            "Mod_T",
            "High_T",
        ]
        print(f"Dropping extra columns: {extra_cols}")
        points = points.drop(columns=extra_cols)
    points = parallel_sjoin(points, hex_grid, 20)
    points = points.drop(columns="index_right")
    print(
        f"Size after join(hex_grid): {len(points):,}."
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)

    # Geohash the grid points so we can drop geometry data entirely
    geometry = points["geometry"]
    # Drop geometries to avoid dask_geopandas (bugs)
    points = points.drop("geometry", axis=1)
    hasher = GridGeohasher()
    points["geohash"] = hasher.geohash(geometry)
    geometry = None
    print("Adding lon/lat")
    lon, lat = hasher.geohash_to_lonlat(points.geohash.to_numpy())
    points["lon"] = lon
    points["lat"] = lat
    lon = None
    lat = None
    print("Adding x/y")
    x, y = hasher.geohash_to_xy(points.geohash.to_numpy())
    points["x"] = x
    points["y"] = y

    points = _drop_duplicates(points)
    print(
        f"Size after drop(duplicated(geohash, Ig_Date, Event_ID)): "
        f"{len(points):,}."
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)

    burned_indices = hasher.geohash_to_ij(
        points.geohash.drop_duplicates().to_numpy()
    )

    points = _add_raster(points, mtbs_path, "bs", burned_indices, how="left")
    points = points.astype({"bs": pd.ArrowDtype(pyarrow.int8())})
    print(
        f"Size after join(bs): {len(points):,}"
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)
    # Drop non-severity pixels
    # points = points[points.bs < 5]
    # print(
    #     f"Size after drop(bs < 5): {len(points):,}."
    #     f" Loss/gain: {(len(points) - n):+,}"
    # )
    # n = len(points)

    points = _add_raster(points, nlcd_path, "nlcd", burned_indices)
    print(
        f"Size after join(nlcd): {len(points):,}"
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)
    points = _add_raster(
        points, NLCD_MODE_RASTER_PATH, "nlcd_mode", burned_indices
    )
    print(
        f"Size after join(nlcd_mode): {len(points):,}"
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)
    # WUI joins are skipped when the paths are None (MTBS_SKIP_WUI); a real
    # path runs the join exactly as before. Nothing downstream of m10 reads
    # the wui_* columns, so omitting them is safe (phd-research, 2026-06-09).
    if wui_flag_path is not None:
        points = _add_raster(points, wui_flag_path, "wui_flag", burned_indices)
        print(
            f"Size after join(wui_flag): {len(points):,}"
            f" Loss/gain: {(len(points) - n):+,}"
        )
        n = len(points)
    if wui_class_path is not None:
        points = _add_raster(
            points, wui_class_path, "wui_class", burned_indices
        )
        print(
            f"Size after join(wui_class): {len(points):,}"
            f" Loss/gain: {(len(points) - n):+,}"
        )
        n = len(points)
    if wui_bool_path is not None:
        points = _add_raster(points, wui_bool_path, "wui_bool", burned_indices)
        print(
            f"Size after join(wui_bool): {len(points):,}"
            f" Loss/gain: {(len(points) - n):+,}"
        )
        n = len(points)
    if wui_prox_path is not None:
        points = _add_raster(points, wui_prox_path, "wui_prox", burned_indices)
        print(
            f"Size after join(wui_prox): {len(points):,}"
            f" Loss/gain: {(len(points) - n):+,}"
        )
        n = len(points)
    points = _add_raster(
        points,
        elevation_path,
        "elevation",
        burned_indices,
        resample_method="average",
    )
    print(
        f"Size after join(elevation): {len(points):,}"
        f" Loss/gain: {(len(points) - n):+,}"
    )
    n = len(points)

    # Convert to dask dataframe for saving in parallel
    nparts = max(int(np.round(n / TARGET_POINTS_PER_PARTITION)), 1)
    points = dd.from_pandas(points, npartitions=nparts)
    points.to_parquet(out_path)


def save_raster_to_points(years, crs, drop_extra_cols):
    aoi_gs = get_conus_geom(crs)
    states = get_states(crs)
    for year in years:
        perims_raster_path = get_mtbs_perims_raster_path(year)
        mtbs_path = get_mtbs_raster_path(year, "CONUS")
        nlcd_path = get_nlcd_raster_path(year)
        wui_flag_path = wui_flavor_path_or_none(year, "flag")
        wui_class_path = wui_flavor_path_or_none(year, "class")
        wui_bool_path = wui_flavor_path_or_none(year, "bool")
        wui_prox_path = wui_flavor_path_or_none(year, "prox")
        if not perims_raster_path.exists():
            print(f"---\nNo raster file. Skipping: {year}")
            continue
        pts_path = get_points_path(year, "CONUS")
        if pts_path.exists():
            print(f"---\nSkipping: {year}")
            continue
        perims = get_mtbs_perims_by_year_and_aoi(year, aoi_gs.values[0], crs)
        eco_regions = get_eco_regions_by_aoi(aoi_gs.values[0], crs)
        hex_grid = gpd.read_file(HEX_GRID_PATH)
        elevation_path = ELEVATION_PATH
        print(f"---\nPreprocessing: {year}")
        start = time.time()
        with ProgressBar():
            _build_dataframe_and_save(
                perims_raster_path,
                mtbs_path,
                nlcd_path,
                wui_flag_path,
                wui_class_path,
                wui_bool_path,
                wui_prox_path,
                elevation_path,
                pts_path,
                year,
                perims,
                states,
                eco_regions,
                hex_grid,
                drop_extra_cols,
            )
        print("Done")
        d = time.time() - start
        print(f"Total time for {year}: {_format_elapsed_time(d)}")


def main(min_year, max_year, all_columns):
    years = list(range(min_year, max_year + 1))
    crs = DEFAULT_CRS
    save_raster_to_points(years, crs, not all_columns)


DESC = """
Convert MTBS rasters to dataframes and combine them with various datasets.

This script produces a separate parquet output for each year. These can then be
concatenated together later.
"""


def _get_parser():
    p = argparse.ArgumentParser(description=DESC)
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
        "-a",
        "--all-columns",
        action="store_true",
        help="Keep all extra columns",
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    assert args.min_year <= args.max_year
    main(args.min_year, args.max_year, args.all_columns)
