# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A staged geospatial pipeline that turns MTBS burn-severity rasters (CONUS, 1984–2022)
plus covariate layers (NLCD land cover, WUI, eco-regions, elevation/slope/aspect, states,
hex grid) into a single per-pixel **burn-record dataframe**, then derives fire-return-interval
and survival statistics from it. There is no application or service — everything is batch
scripts run over a large fixed data volume.

## Commands

Environment is managed by `uv` (Python 3.12, pinned). The package is installed editable.

```bash
# Run any pipeline/analysis module (they are package modules with __main__ blocks)
uv run python -m mtbs_fire_analysis.pipeline.m10_data_extract --min_year 1984 --max_year 2022
uv run python -m mtbs_fire_analysis.pipeline.m20_combine <year_pqt>... <out_path>
uv run python -m mtbs_fire_analysis.analysis.plot_dt <combined_parquet>

# Lint / format (ruff, line-length 79; black config also present but ruff is the linter)
uv run ruff check .
uv run ruff format .

# Rasterization steps are bash + GNU parallel + gdal_rasterize, not Python:
bash mtbs_fire_analysis/pipeline/m01_rasterize_perims.sh
```

Run the tests with `uv run pytest tests` (the catalog integration suite self-skips unless `JLAB_ROOT` points at a store with an index). There is no CI; also verify pipeline changes by running the relevant stage on a small year range.

## Pipeline stages (run in numeric order)

The `m<NN>_` prefix encodes ordering. `m00`/`m01`/`m02` are one-off source-preparation steps
(each reads a raw source and writes a cleaned/reprojected canonical form); `m10`+ are the main
build.

- **m00_** — clean raw vectors/rasters into canonical forms: eco-regions, hex grid, MTBS
  rasters (reprojected to the standard grid), fire perimeters, states, WUI intermediate.
- **m01_** — derive/rasterize: elevation → slope & aspect; NLCD reproject; WUI flavors;
  `*.sh` scripts rasterize eco-regions and perimeters via `gdal_rasterize`.
- **m02_** — NLCD mode raster; WUI rasterization.
- **m10_data_extract** — the core builder. Per year: load burned pixels as points →
  spatial-join perims/states/eco-regions/hex-grid → geohash the points and **drop geometry** →
  attach raster bands (`bs`, `nlcd`, WUI flavors, `elevation`, `slope`, `aspect`) by geohash →
  write one parquet per year.
- **m11_join_raster_data** — bolt an additional raster band onto existing per-year frames.
- **m20_combine** — concat per-year parquets into one combined dataframe (dask LocalCluster).
- **m30_build_st** — build "survival time" (years since last fire) rasters by accumulating a
  running max of `dse` (days-since-epoch of ignition) across years.
- **pixel_counts** — NLCD-mode pixel counts per eco-region.

`analysis/mining.py` converts the combined dataframe into `dt` (time between successive fires
per pixel), survival/censoring times, and event histories using polars. `analysis/plot_dt.py`
plots `dt` distributions by eco-region and burn severity.

## Key architecture concepts

- **Geohash is the universal join key** (`geohasher.py`, `defaults.py`). A single fixed CONUS
  30 m Albers grid (`DEFAULT_GEOHASH_GEOBOX`, derived from the MTBS raster grid) defines an
  integer cell index for every pixel via `GridGeohasher`. Once points are geohashed, geometry
  is dropped entirely and all subsequent joins are integer joins in polars/dask — much faster
  and lower-memory than geopandas spatial joins. When adding a raster attribute, follow the
  existing pattern: read pixel values at burned `(row, col)` indices, convert those indices to
  geohashes, and join.
- **One canonical CRS/grid.** Everything is reprojected to `DEFAULT_CRS` /
  `DEFAULT_GEOHASH_GEOBOX` before joining. Don't introduce a second grid or CRS.
- **`pipeline/paths.py` is the single source of truth for all filesystem locations.** Every
  path derives from one root, `FIRE_DATA_ROOT`, read from a repo-root `.env` (see
  `.env.example`) and falling back to the server default when unset; precedence is shell
  env > `.env` > default. The root is expected to contain `<root>/data` (inputs and results)
  and `<root>/data_tmp` (per-year intermediates). Add new inputs/outputs here (constants +
  `get_*` builder functions) rather than hardcoding paths in stage scripts.
- **Catalog adapter (`pipeline/catalog_paths.py`).** When the catalog is enabled, `paths.py`
  rebinds exactly the 12 catalog-backed inputs (the 9 raster/vector input constants plus
  `get_mtbs_raster_path` / `get_nlcd_raster_path` / `get_wui_flavor_path`) to the jlab client,
  which resolves them from the platform store by `find()`/`localize()` and returns local paths;
  NLCD/WUI temporal substitutions happen inside the client via the loaded profile, not here.
  The remaining scratch/results/cache surface, including `get_points_path`, stays on the
  `FIRE_DATA_ROOT` layout. The adapter is enabled by `FIRE_CATALOG` (1/true/on vs 0/false/off);
  when unset it is on iff `JLAB_ROOT` or `JLAB_API_URL` is set. Resolution is eager at import
  and fails loudly if the platform is misconfigured (no silent fallback).
- **Dask memory is treated as hostile.** Existing code sets `MALLOC_TRIM_THRESHOLD_`, wraps
  raster loads in throwaway fetcher functions so graphs get GC'd, and prefers polars for the
  big joins. Preserve these patterns; don't hold dask graph references in scope after compute.
- **Idempotent, resumable stages.** Most stages skip work whose output already exists (see
  `protected_raster_save_with_cleanup` in `utils.py`, which also deletes partial files on
  interrupt). Use `--clear-cache` (m10) to force a rebuild.

## Conventions

- Column shorthands: `bs` = burn severity, `dse` = days since epoch (ignition date),
  `st` = survival time, `dt` = time between fires (years), `eco_lvl_1/2/3` = EPA eco-region
  levels, WUI flavors are `flag`/`class`/`bool`/`prox`.
- ruff line-length is 79; isort/black first-party is `mtbs_fire_analysis`.
- Rasters are saved tiled + zstd-compressed via the `utils.py` helpers.
