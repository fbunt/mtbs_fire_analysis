from pathlib import Path

import dask.array as da
import numpy as np
import polars as pl
import raster_tools as rts
import rioxarray as xrio
from dask.diagnostics import ProgressBar

DATA_DIR = Path("/run/media/fire_analysis/data")


def bp_chunk(st, nlcd, eco, geohash, valid):
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

    hazard_tbl = pl.read_parquet(DATA_DIR / "hazard_table.parquet")
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


geohash_path = DATA_DIR / "hash_grid.tif"
st_path = DATA_DIR / "st/st.tif"
nlcd_path = DATA_DIR / "nlcd/cleaned/Annual_NLCD_LndCov_2022_CU_C1V0.tif"
eco_path = DATA_DIR / "eco_retions/eco_regions.tif"

# The geohash raster is the largest raster in terms of dtype so use its
# chunksize.
geohash_raster = rts.Raster(DATA_DIR / "hash_grid.tif")
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
)
bp_raster = rts.data_to_raster_like(bp_data, geohash_raster, nv=-1)
with ProgressBar():
    bp_raster.save(DATA_DIR / "bp.tif")
