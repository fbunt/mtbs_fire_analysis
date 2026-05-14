"""m10b: per-pixel-per-year NLCD frame builder.

Per ``docs/plans/STRATIFICATION_REDESIGN.md`` §4.4 + §8 Stage 3 (strategy
(a) full-annual ratified by Stage 3-spike receipt
``nlcd_mode_per_interval_compute_spike_20260514T024454Z.json``): for
every ever-burned pixel in CONUS, emit the NLCD value for every year
in `[W_0, W_1] = [1984, 2022]`. Downstream consumer:
``mining.py:build_event_histories`` joins this frame and computes
``nlcd_mode_per_interval: List(UInt8)`` per fire interval.

Output schema (one parquet at ``${FIRE_RESULTS_DIR}/per_pixel_per_year_nlcd.parquet``):

    geohash:        UInt64   (project geobox pixel identifier)
    nlcd_per_year:  List(UInt8)  (length 39 = W_1 - W_0 + 1; entry i = NLCD value
                                  for year W_0 + i; cleaned NLCD nodata preserved
                                  as NLCD's nodata value)

Processing approach: for each year 1984..2022, open that year's NLCD
raster via the same ``rts.Raster(...).xdata.isel(...).to_numpy()``
pattern that ``m10_data_extract._add_raster`` uses (faithful to the
Stage 3-spike's measurement; honest extrapolation). Sample at every
ever-burned pixel's (row, col) index. Store as a column in a (n_pixels,
n_years) uint8 array. Convert to polars list-column at the end.

Production-checkpoint gate (per Stage 3 §8 review C1 mitigation):
after the first 3 years complete, compare cumulative wall time against
the linear extrapolation derived from the Stage 3-spike. If
cumulative actual > 2× linear extrapolation, log a warning and abort —
chunk-thrash from scattered-pixel reads at CONUS scale may have
broken the linearity assumption.

Run::

    source tools/set_branch_env.sh
    uv run --extra pipeline python \\
        packages/upstream-mtbs/mtbs_fire_analysis/pipeline/m10b_per_pixel_per_year_nlcd.py

Optional ``--eco-only ECO_LVL_3`` flag restricts processing to a single
eco_lvl_3 region (for smoke tests; writes
``per_pixel_per_year_nlcd_eco{N}.parquet`` instead of the full file).

Receipt: ``tools/data_audits/m10b_per_pixel_per_year_nlcd_<ts>.json``
with per-year wall times + total + parquet provenance + checkpoint
verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import raster_tools as rts
import rasterio
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "tools" / "data_audits"))
from data_pipeline_invariants import rasterize_eco_polygons  # noqa: E402

from mtbs_fire_analysis.geohasher import GridGeohasher  # noqa: E402
from mtbs_fire_analysis.pipeline.paths import (  # noqa: E402
    ECO_REGIONS_PATH,
    EVER_BURNED_MASK_PATH,
    NLCD_MODE_RASTER_PATH,
    RESULTS_DIR,
    get_nlcd_raster_path,
)


YEAR_START = 1984
YEAR_END = 2022  # inclusive
N_YEARS = YEAR_END - YEAR_START + 1
# NLCD nodata sentinel (uint8 250 per the Annual_NLCD_LndCov_*.tif
# rasters). Pixels where every year is nodata are filtered post-loop
# so downstream consumers see only NLCD-valid pixels (Stage 3a review
# C1 fold-in; mirrors m10_data_extract._add_raster's per-year nodata
# drop at the per-pixel granularity m10b emits).
NLCD_NODATA = 250

# Linear-extrapolation reference from the Stage 3-spike (cascades_shrub):
# 8,704,820 pixels × 39 years took 167.1 s → ~4.93e-7 s per pixel per year.
# CONUS extrapolation: 565,406,561 × 4.93e-7 = 278.8 s per year. Used by
# the production-checkpoint gate after the first 3 years.
SPIKE_SECONDS_PER_PIXEL_PER_YEAR = 167.1 / (8_704_820 * 39)
CHECKPOINT_AFTER_N_YEARS = 3
CHECKPOINT_RATIO_LIMIT = 2.0
RECEIPT_DIR = REPO_ROOT / "tools" / "data_audits"
RECEIPT_PREFIX = "m10b_per_pixel_per_year_nlcd"


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_results_dir() -> Path:
    """Confirm FIRE_RESULTS_DIR resolves to a branch-isolated path.

    Aborts if RESULTS_DIR resolves to main's default (the 2026-05-01
    incident this guard defends against). Mirrors the workflow-isolation
    discipline from ``CLAUDE.md``.
    """
    env_val = os.environ.get("FIRE_RESULTS_DIR", "")
    if not env_val:
        raise RuntimeError(
            "FIRE_RESULTS_DIR not set. Run "
            "'source tools/set_branch_env.sh' before m10b. m10b writes "
            "a multi-GB parquet to RESULTS_DIR; running under main's "
            "default would clobber main's results."
        )
    p = Path(env_val)
    if "fire_analysis_branch_spatial_covariates" not in str(p):
        raise RuntimeError(
            f"FIRE_RESULTS_DIR ({p}) does not look branch-isolated for "
            "feat/spatial-covariates. Expected a path containing "
            "'fire_analysis_branch_spatial_covariates'. Aborting."
        )
    if not p.exists():
        raise RuntimeError(f"FIRE_RESULTS_DIR does not exist: {p}")
    return p


def _build_ever_burned_indices(
    eco_only: "int | None" = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute (rows, cols) for the ever-burned pixel set.

    If ``eco_only`` is set, restricts to that single eco_lvl_3 region
    (smoke-test mode). Otherwise CONUS-wide.
    """
    t0 = time.time()
    with rasterio.open(EVER_BURNED_MASK_PATH) as src:
        ever_burned = src.read(1)
    ever_burned_read_seconds = time.time() - t0

    if eco_only is not None:
        t0 = time.time()
        eco_raster = rasterize_eco_polygons(
            ECO_REGIONS_PATH,
            NLCD_MODE_RASTER_PATH,
            eco_lvl_col="eco_lvl_3",
            fill=-1,
        )
        if eco_raster.shape != ever_burned.shape:
            raise RuntimeError(
                f"shape mismatch: ever_burned={ever_burned.shape}, "
                f"eco={eco_raster.shape}"
            )
        mask = (eco_raster == eco_only) & (ever_burned > 0)
        eco_rasterise_seconds = time.time() - t0
    else:
        mask = ever_burned > 0
        eco_rasterise_seconds = 0.0

    rows, cols = np.where(mask)
    n_pixels = int(rows.size)
    stats = {
        "ever_burned_read_seconds": round(ever_burned_read_seconds, 3),
        "eco_rasterise_seconds": round(eco_rasterise_seconds, 3),
        "n_pixels": n_pixels,
        "eco_only": eco_only,
        "raster_shape": list(ever_burned.shape),
    }
    return rows, cols, stats


def _sample_year(
    year: int, rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, float, int, int]:
    """Sample one year's NLCD at (rows, cols). Mirrors m10's _add_raster."""
    nlcd_path = get_nlcd_raster_path(year)
    if not nlcd_path.exists():
        raise FileNotFoundError(f"NLCD raster missing for year {year}: {nlcd_path}")

    t0 = time.time()
    raster = rts.Raster(str(nlcd_path))
    n = int(rows.size)
    values = (
        raster.xdata.isel(
            band=xr.DataArray(np.zeros(n, dtype=int), dims="z"),
            y=xr.DataArray(rows, dims="z"),
            x=xr.DataArray(cols, dims="z"),
        )
        .to_numpy()
        .astype(np.uint8, copy=False)
    )
    wall = time.time() - t0
    if raster.null_value is not None:
        nonnull_mask = ~rts.raster.get_mask_from_data(values, raster.null_value)
        n_nonnull = int(nonnull_mask.sum())
    else:
        n_nonnull = int(values.size)
    return values, wall, int(values.size), n_nonnull


def _checkpoint_gate(
    n_pixels: int,
    cumulative_wall: float,
    n_years_done: int,
) -> tuple[bool, str]:
    """Compare cumulative wall to linear extrapolation; abort if > 2×."""
    expected_per_year = n_pixels * SPIKE_SECONDS_PER_PIXEL_PER_YEAR
    expected_cumulative = expected_per_year * n_years_done
    ratio = cumulative_wall / max(expected_cumulative, 1e-9)
    note = (
        f"cumulative {cumulative_wall:.1f}s vs extrapolated "
        f"{expected_cumulative:.1f}s ({n_years_done} years done at "
        f"{expected_per_year:.1f}s/yr extrapolated); ratio "
        f"{ratio:.2f}× of linear."
    )
    return ratio > CHECKPOINT_RATIO_LIMIT, note


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eco-only",
        type=int,
        default=None,
        help="Smoke-test mode: restrict to a single eco_lvl_3 region "
        "(e.g., 6207 for cascades_shrub).",
    )
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    results_dir = _resolve_results_dir()
    print(f"[m10b] starting at {started_at}")
    print(f"[m10b] FIRE_RESULTS_DIR = {results_dir}")
    if args.eco_only is not None:
        print(f"[m10b] SMOKE TEST: eco_lvl_3 = {args.eco_only}")

    rows, cols, mask_stats = _build_ever_burned_indices(eco_only=args.eco_only)
    n_pixels = mask_stats["n_pixels"]
    if n_pixels == 0:
        print("[m10b] no ever-burned pixels matched. Aborting.")
        return 2
    print(
        f"[m10b] ever-burned pixels: {n_pixels:,} "
        f"(extrapolated total ~{n_pixels * SPIKE_SECONDS_PER_PIXEL_PER_YEAR * N_YEARS / 60:.1f} min)"
    )

    nlcd_per_year = np.empty((n_pixels, N_YEARS), dtype=np.uint8)
    per_year_timings: list[dict] = []
    cumulative_wall = 0.0
    checkpoint_passed = False
    abort_reason: str | None = None

    for year_idx, year in enumerate(range(YEAR_START, YEAR_END + 1)):
        values, wall, n_values, n_nonnull = _sample_year(year, rows, cols)
        nlcd_per_year[:, year_idx] = values
        cumulative_wall += wall
        per_year_timings.append(
            {
                "year": year,
                "wall_seconds": round(wall, 3),
                "n_nonnull": n_nonnull,
            }
        )
        print(
            f"[m10b] year {year}: {wall:6.2f}s "
            f"(cumulative {cumulative_wall:7.1f}s, "
            f"n_nonnull {n_nonnull:,})"
        )

        if (
            year_idx + 1 == CHECKPOINT_AFTER_N_YEARS
            and args.eco_only is None
            and not checkpoint_passed
        ):
            over_limit, note = _checkpoint_gate(
                n_pixels, cumulative_wall, year_idx + 1
            )
            print(f"[m10b] checkpoint after year {year}: {note}")
            if over_limit:
                abort_reason = (
                    f"checkpoint gate tripped: {note}. "
                    f"Per Stage 3 §8 production-checkpoint gate, aborting."
                )
                print(f"[m10b] {abort_reason}")
                break
            checkpoint_passed = True
            print("[m10b] checkpoint PASSED — proceeding with full run.")

    captured_at = datetime.now(timezone.utc).isoformat()

    if abort_reason is not None:
        receipt = {
            "_about": "m10b per-pixel-per-year NLCD aborted at checkpoint",
            "schema_version": 1,
            "started_at": started_at,
            "captured_at": captured_at,
            "verdict": "ABORTED-CHECKPOINT",
            "abort_reason": abort_reason,
            "n_pixels": n_pixels,
            "per_year_timings": per_year_timings,
            "cumulative_wall_seconds": round(cumulative_wall, 1),
        }
        out_path = RECEIPT_DIR / f"{RECEIPT_PREFIX}_{_utc_ts()}.json"
        out_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[m10b] receipt: {out_path}")
        return 3

    # Stage 3a review C1 fold-in: drop pixels where every year's NLCD
    # is nodata (250). These pixels are ever-burned per the MTBS
    # perimeter raster but never NLCD-valid in [W_0, W_1] (1.13M
    # pixels = 0.2% in the 2026-05-14 CONUS run — static-nodata
    # footprint per the receipt's per-year n_nonnull constancy). The
    # downstream `nlcd_mode_per_interval` kernel skips nodata in its
    # bincount, so a pixel with even one valid year is informative;
    # all-nodata pixels carry no signal and would only emit the
    # NLCD_NODATA sentinel for every interval. Drop them upstream so
    # the join in `build_event_histories` operates on a clean frame.
    t_filter = time.time()
    any_valid_mask = (nlcd_per_year != NLCD_NODATA).any(axis=1)
    n_pixels_dropped_all_nodata = int((~any_valid_mask).sum())
    if n_pixels_dropped_all_nodata > 0:
        rows = rows[any_valid_mask]
        cols = cols[any_valid_mask]
        nlcd_per_year = nlcd_per_year[any_valid_mask]
        n_pixels = int(any_valid_mask.sum())
    nodata_filter_seconds = time.time() - t_filter
    print(
        f"[m10b] nodata filter: dropped {n_pixels_dropped_all_nodata:,} "
        f"all-nodata pixels ({nodata_filter_seconds:.2f}s); "
        f"remaining {n_pixels:,} pixels have >=1 NLCD-valid year"
    )

    # Build polars frame with (geohash, nlcd_per_year List[UInt8]).
    print("[m10b] geohashing pixel indices...")
    t0 = time.time()
    hasher = GridGeohasher()
    geohash = hasher.geohash_from_ij((rows, cols))
    geohash_seconds = time.time() - t0
    print(f"[m10b] geohash done in {geohash_seconds:.1f}s")

    # Chunked write to a directory of `part.{N}.parquet` files (matches
    # the m20_combine.py output layout — dask's natural `dd.to_parquet`
    # convention used elsewhere in the upstream pipeline). Splitting
    # keeps each chunk's flat value buffer (chunk_pixels × 39) under the
    # int32 offset limit (~2.1B): at 565M CONUS ever-burned pixels × 39
    # years we have ~22B values total, which would overflow int32
    # offsets in a single `pa.ListArray`. polars 1.40 converts
    # `pa.LargeListArray` back to its internal int32-backed List on
    # read, so a single-file LargeList workaround doesn't survive —
    # chunking + a `part.*.parquet` directory is the robust fix.
    # Downstream consumers read with
    # `pl.scan_parquet(dir / "part.*.parquet")` (or `dd.read_parquet(dir)`);
    # the chunking is transparent at the logical-frame level.
    if args.eco_only is not None:
        out_dir = results_dir / f"per_pixel_per_year_nlcd_eco{args.eco_only}"
        chunk_size = n_pixels  # smoke-mode: single chunk
    else:
        out_dir = results_dir / "per_pixel_per_year_nlcd"
        chunk_size = 50_000_000
    out_dir.mkdir(exist_ok=True)

    print(f"[m10b] writing chunked parquets to {out_dir}/")
    t0 = time.time()
    write_seconds = 0.0
    df_build_seconds = 0.0
    parquet_bytes = 0
    chunk_paths: list[Path] = []
    n_chunks = (n_pixels + chunk_size - 1) // chunk_size
    for chunk_i in range(n_chunks):
        lo = chunk_i * chunk_size
        hi = min(lo + chunk_size, n_pixels)
        chunk_n = hi - lo
        t_build = time.time()
        offsets = np.arange(chunk_n + 1, dtype=np.int32) * N_YEARS
        flat_values = pa.array(nlcd_per_year[lo:hi].ravel())
        list_array = pa.ListArray.from_arrays(pa.array(offsets), flat_values)
        table = pa.table(
            {
                "geohash": pa.array(geohash[lo:hi]),
                "nlcd_per_year": list_array,
            }
        )
        df_chunk = pl.from_arrow(table)
        df_build_seconds += time.time() - t_build
        chunk_path = out_dir / f"part.{chunk_i}.parquet"
        chunk_paths.append(chunk_path)
        t_write = time.time()
        df_chunk.write_parquet(chunk_path, compression="zstd")
        write_seconds += time.time() - t_write
        parquet_bytes += chunk_path.stat().st_size
        print(
            f"[m10b]   chunk {chunk_i + 1}/{n_chunks}: "
            f"{chunk_n:,} rows → {chunk_path.name} "
            f"({chunk_path.stat().st_size / 1e9:.2f} GB)"
        )
    print(
        f"[m10b] all chunks done in {time.time() - t0:.1f}s "
        f"(build {df_build_seconds:.1f}s, write {write_seconds:.1f}s, "
        f"total {parquet_bytes / 1e9:.2f} GB across {n_chunks} chunks)"
    )
    out_parquet = out_dir

    expected_total = n_pixels * SPIKE_SECONDS_PER_PIXEL_PER_YEAR * N_YEARS
    receipt = {
        "_about": (
            "m10b per-pixel-per-year NLCD frame builder per "
            "STRATIFICATION_REDESIGN.md §8 Stage 3 (strategy a "
            "ratified by Stage 3-spike). Emits "
            "per_pixel_per_year_nlcd.parquet for downstream "
            "build_event_histories join."
        ),
        "schema_version": 1,
        "started_at": started_at,
        "captured_at": captured_at,
        "verdict": "PASS",
        "scope": {
            "eco_only": args.eco_only,
            "year_start": YEAR_START,
            "year_end": YEAR_END,
            "n_years": N_YEARS,
        },
        "mask_stats": mask_stats,
        "per_year_timings": per_year_timings,
        "cumulative_wall_seconds": round(cumulative_wall, 1),
        "extrapolated_seconds": round(expected_total, 1),
        "actual_vs_extrapolated_ratio": round(
            cumulative_wall / max(expected_total, 1e-9), 3
        ),
        "checkpoint_after_n_years": CHECKPOINT_AFTER_N_YEARS,
        "checkpoint_ratio_limit": CHECKPOINT_RATIO_LIMIT,
        "checkpoint_passed": checkpoint_passed or args.eco_only is not None,
        "n_pixels_dropped_all_nodata": n_pixels_dropped_all_nodata,
        "nodata_filter_seconds": round(nodata_filter_seconds, 3),
        "invariant_all_pixels_have_one_valid_year": (
            "Every pixel in the output has at least one year in "
            f"[{YEAR_START}, {YEAR_END}] with NLCD != {NLCD_NODATA}. "
            "Downstream consumers (mining.py:build_event_histories) can "
            "join freely; the kernel's per-interval bincount skips "
            f"NLCD={NLCD_NODATA} values and emits {NLCD_NODATA} as the "
            "interval mode iff all years in that interval are nodata."
        ),
        "geohash_seconds": round(geohash_seconds, 1),
        "df_build_seconds": round(df_build_seconds, 1),
        "write_seconds": round(write_seconds, 1),
        "out_dir": str(out_parquet),
        "chunk_paths": [str(p) for p in chunk_paths],
        "n_chunks": len(chunk_paths),
        "parquet_bytes": parquet_bytes,
        "paths": {
            "ever_burned_mask_path": str(EVER_BURNED_MASK_PATH),
            "eco_regions_path": str(ECO_REGIONS_PATH),
            "nlcd_year_format": str(get_nlcd_raster_path(YEAR_START)),
        },
        "spike_reference": (
            "tools/data_audits/nlcd_mode_per_interval_compute_spike_"
            "20260514T024454Z.json"
        ),
    }
    out_receipt = RECEIPT_DIR / f"{RECEIPT_PREFIX}_{_utc_ts()}.json"
    out_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[m10b] receipt: {out_receipt}")
    print("[m10b] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
