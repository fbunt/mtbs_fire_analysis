"""Resample 270 m solar mosaics to 30 m on DEFAULT_GEOHASH_GEOBOX.

Reads from ``${FIRE_DATA_ROOT}/data/covariates/solar/`` (the
implementer-side-promoted production mosaics from PR-2's
``${FIRE_TMP_DIR}/solar/mosaic_270m/``). Writes to
``${FIRE_RESULTS_DIR}/covariates/solar/`` per SOLAR_COVARIATE_PLAN.md
§9. Bilinear resampling, snap-to-reference invariant against
DEFAULT_GEOHASH_GEOBOX (the 30 m project grid that all other
covariate rasters share).

The bilinear choice (over cubic / cubic_spline / nearest) is per
plan §9 "Resample 270 m → 30 m via bilinear" — the underlying
field is smooth at 270 m, bilinear is the physics-correct call,
cubic adds ringing risk near shadow boundaries.

Each output is ~63 GB float32 uncompressed (100150 × 157144 cells);
deflate-compressed with predictor=3 (float-aware) typically lands
~10-20 GB. 7 outputs total = ~100-140 GB. Confirm FIRE_RESULTS_DIR
free space before running.

Run:
    source tools/set_branch_env.sh
    uv run --extra pipeline python -m mtbs_fire_analysis.pipeline.m02b_solar_resample_30m
"""

from pathlib import Path

import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from mtbs_fire_analysis.defaults import (
    DEFAULT_CRS,
    DEFAULT_GEOHASH_AFFINE,
    DEFAULT_GEOHASH_GRID_SHAPE,
)
from mtbs_fire_analysis.pipeline.paths import MAIN_FOLDER_ALIAS, RESULTS_DIR

SRC_DIR = MAIN_FOLDER_ALIAS / "data" / "covariates" / "solar"
DST_DIR = RESULTS_DIR / "covariates" / "solar"

WINDOWS = [
    "growing_season",
    "monthly_Apr",
    "monthly_May",
    "monthly_Jun",
    "monthly_Jul",
    "monthly_Aug",
    "monthly_Sep",
]


def resample_one(window: str) -> Path:
    src_path = SRC_DIR / f"solar_{window}_270m.tif"
    dst_path = DST_DIR / f"solar_{window}.tif"
    if not src_path.exists():
        raise FileNotFoundError(
            f"missing 270 m mosaic input: {src_path}. Confirm the "
            "main-side promotion step from FIRE_TMP_DIR completed."
        )
    with rio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=DEFAULT_CRS,
            transform=DEFAULT_GEOHASH_AFFINE,
            width=DEFAULT_GEOHASH_GRID_SHAPE[1],
            height=DEFAULT_GEOHASH_GRID_SHAPE[0],
            compress="deflate",
            predictor=3,  # float-aware predictor
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="YES",
        )
        with rio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, band_idx),
                    destination=rio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=DEFAULT_GEOHASH_AFFINE,
                    dst_crs=DEFAULT_CRS,
                    resampling=Resampling.bilinear,
                )
    return dst_path


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for window in WINDOWS:
        print(f"Resampling {window}...", flush=True)
        out = resample_one(window)
        size_gb = out.stat().st_size / (1024**3)
        print(f"  → {out} ({size_gb:.1f} GB)", flush=True)
    print("All 7 windows resampled to 30 m on DEFAULT_GEOHASH_GEOBOX.")


if __name__ == "__main__":
    main()
