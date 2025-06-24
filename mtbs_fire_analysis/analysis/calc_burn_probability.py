import argparse

import dask.array as da
import numpy as np
import polars as pl
import raster_tools as rts
import rioxarray as xrio
from dask.diagnostics import ProgressBar

from mtbs_fire_analysis.pipeline.paths import (
    ECO_REGIONS_RASTER_PATH,
    MTBS_ROOT,
    ST_PATH,
    get_nlcd_raster_path,
)


def bp_chunk(st, nlcd, eco, geohash, valid, hazard_table_path):
    shape = st.shape
    st = st[valid]
    nlcd = nlcd[valid]
    eco = eco[valid]
    geohash = geohash[valid]
    frame = pl.DataFrame(
        {
            "idx": np.arange(len(st)),
            "st": st,
            "nlcd": nlcd,
            "eco": eco,
            "geohash": geohash,
        }
    )
    # GC as eagerly as possible
    st = None
    nlcd = None
    eco = None
    geohash = None

    hazard_tbl = pl.read_parquet(hazard_table_path)
    frame = frame.join(hazard_tbl, on=["eco", "nlcd", "st"], how="left").sort(
        "idx"
    )
    hazards = frame.select("hazard").to_numpy().flatten()
    # Map nan to -1
    hazards = np.nan_to_num(hazards, nan=-1)
    frame = None
    hazard_tbl = None
    result = np.full(shape, -1, dtype="float32")
    result[valid] = hazards
    return result


def main(eco_level, year):
    geohash_raster = rts.Raster(MTBS_ROOT / "hash_grid.tif")
    st_path = ST_PATH / "st.tif"
    nlcd_path = get_nlcd_raster_path(year)
    eco_path = ECO_REGIONS_RASTER_PATH / f"eco_lvl_{eco_level}.tif"
    # TODO: Add to paths?
    hazard_table_path = MTBS_ROOT / "hazard_table.parquet"

    # The geohash raster is the largest raster in terms of dtype so use its
    # chunksize.
    chunks = geohash_raster.data.chunks
    geohash_data = geohash_raster.data
    # Align the chunks of the other rasters to keep rechunking overhead to 0.
    st_data = xrio.open_rasterio(st_path, chunks=chunks).data
    nlcd_data = xrio.open_rasterio(nlcd_path, chunks=chunks).data
    nlcd_nv = rts.Raster(nlcd_path).null_value
    valid_data = ~rts.raster.get_mask_from_data(nlcd_data, nlcd_nv)
    eco_data = xrio.open_rasterio(eco_path, chunks=chunks).data

    bp_data = da.map_blocks(
        bp_chunk,
        st_data,
        nlcd_data,
        eco_data,
        geohash_data,
        valid_data,
        chunks=chunks,
        dtype="float32",
        meta=np.array((), dtype="float32"),
        # Func params
        hazard_table_path=hazard_table_path,
    )
    bp_raster = rts.data_to_raster_like(bp_data, geohash_raster, nv=-1)
    with ProgressBar():
        bp_raster.save(MTBS_ROOT / f"bp_eco_lvl_{eco_level}_{year}.tif")


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "eco_level", type=int, default=1, help="The eco regions level to use."
    )
    p.add_argument(
        "year", type=int, help="The year to calculate burn probability for"
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    assert args.year > 1984
    main(args.eco_level, args.year)
