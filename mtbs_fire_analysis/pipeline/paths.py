import os
from pathlib import Path

# ---------------------------------------------------------------------------
#  Path config
# ---------------------------------------------------------------------------
# Inputs (raw rasters, NLCD, eco regions, hex grid, DEM, etc.) are large and
# read-only — shared across branches via ``FIRE_DATA_ROOT``.
#
# Outputs (results, tmp, cache) are per-workflow writable artefacts —
# isolated per branch via ``FIRE_RESULTS_DIR`` / ``FIRE_TMP_DIR`` /
# ``FIRE_CACHE_DIR``. Defaults preserve the historical "everything under
# ``FIRE_DATA_ROOT``" layout so main's workflow keeps working unchanged.
#
# Branches working in parallel with main (e.g. ``feat/spatial-covariates``)
# **must** override ``FIRE_RESULTS_DIR`` (and typically also ``FIRE_TMP_DIR``
# + ``FIRE_CACHE_DIR``) before invoking any pipeline script that writes
# parquets/rasters/scores. See ``docs/plans/COVARIATE_BRANCH_KICKOFF.md``
# §"Workflow isolation from main" for the canonical branch-isolated paths
# and the rationale (Phase 3 lookup-rebaseline incident, 2026-04-30).

MAIN_FOLDER_ALIAS = Path(
    os.environ.get(
        "FIRE_DATA_ROOT",
        "/run/data_raid5/shared_data/fire_analysis",
    )
)

MTBS_ROOT = MAIN_FOLDER_ALIAS / "data"
MTBS_RASTER_DIR = MTBS_ROOT / "mtbs_bs_rasters"
RAW_RASTER_DATA_DIR = MTBS_RASTER_DIR / "raw"
CLEANED_RASTER_DATA_DIR = MTBS_RASTER_DIR / "cleaned"

# Severity-raster READ path is env-configurable so a deployment can point
# the analysis (m10/m11 via get_mtbs_raster_path) at a different MTBS
# severity product without editing this shared file. Default reproduces
# the legacy ``mtbs_bs_rasters/cleaned`` layout (back-compat — Fred's
# other upstream-mtbs users are unaffected). This group's deployment sets
# FIRE_MTBS_BS_SUBDIR=mtbs/cog + FIRE_MTBS_TIF_FMT="{year}.tif" to read
# the canonical fde driver product (the MTBS analog of the NLCD "C1b"
# repoint; see DECISION_REGISTER D-2026-06-05-mtbs-severity-repoint).
# The m00 preprocess + RAW_RASTER_DATA_DIR / CLEANED_RASTER_DATA_DIR
# above stay on the legacy layout so the legacy producer keeps working.
MTBS_SEVERITY_READ_DIR = MTBS_ROOT / os.environ.get(
    "FIRE_MTBS_BS_SUBDIR", "mtbs_bs_rasters/cleaned"
)

# --- Writable output dirs (per-branch isolatable) --------------------------
ROOT_TMP_DIR = Path(
    os.environ.get("FIRE_TMP_DIR", str(MAIN_FOLDER_ALIAS / "data_tmp"))
)
RESULTS_DIR = Path(
    os.environ.get(
        "FIRE_RESULTS_DIR", str(MAIN_FOLDER_ALIAS / "data" / "results")
    )
)
CACHE_DIR = Path(
    os.environ.get("FIRE_CACHE_DIR", str(MAIN_FOLDER_ALIAS / "data" / "cache"))
)

PERIMS_DIR = MTBS_ROOT / "mtbs_perims"
RAW_PERIMS_PATH = PERIMS_DIR / "raw" / "mtbs_perims_DD.shp"
PERIMS_CLEANED_DIR = PERIMS_DIR / "cleaned"
PERIMS_PATH = PERIMS_CLEANED_DIR / "mtbs_perims_trimmed.gpkg"
PERIMS_BY_YEAR_PATH = PERIMS_CLEANED_DIR / "mtbs_perims_by_year.gpkg"

# Coarse-resolution fire-history rebuilds (FIRE_PIXEL_M != 30 m) isolate their
# derived input rasters -- the per-year dse stack (m01) + the ever-burned /
# first-burn-year masks (m02b/m02c) -- under a resolution-stamped sibling dir,
# so a 120 m rebuild can NEVER clobber the canonical 30 m production rasters
# that main + every other consumer read. Default (FIRE_PIXEL_M unset => 30 m)
# => empty suffix => byte-identical legacy paths (back-compat; Fred's other
# upstream-mtbs users are unaffected). This applies
# D-2026-06-04-input-layer-rebuild-isolation decision (a) -- "emit additive /
# distinct names, never the canonical fixed path" -- to the fire-history layer
# at coarse resolution. The env var is read directly (not via
# defaults.pixel_m_from_env) to keep paths.py free of the rasterio/odc.geo
# import weight defaults.py carries; grid_for_pixel_m does the authoritative
# multiple-of-30 validation where the grid is actually computed.
_fire_pixel_m_raw = os.environ.get("FIRE_PIXEL_M")
if _fire_pixel_m_raw:
    try:
        _FIRE_PIXEL_M = int(_fire_pixel_m_raw)
    except ValueError as _e:
        raise ValueError(
            f"FIRE_PIXEL_M must be an integer number of metres, "
            f"got {_fire_pixel_m_raw!r}"
        ) from _e
else:
    _FIRE_PIXEL_M = 30
_RES_SUFFIX = "" if _FIRE_PIXEL_M == 30 else f"_{_FIRE_PIXEL_M}m"

PERIMS_RASTERS_PATH = PERIMS_DIR / f"rasters{_RES_SUFFIX}"
# MTBS perim raster year coverage (single source of truth for m02b
# ever-burned mask + I9 perim-coverage invariant — see Stage 1 review C1).
# Update both ends together when extending the window beyond 2022.
MTBS_PERIM_YEAR_START = 1984
MTBS_PERIM_YEAR_END = 2022  # inclusive
# Derived-input artefacts under mtbs_perims/ (m01-class shared write per
# SOLAR_COVARIATE_PLAN.md §2 "Deliberate exception: m01-class derived
# inputs"). Stratification-redesign Stage 1 lands the ever-burned mask
# here.
PERIMS_DERIVED_DIR = PERIMS_DIR / f"derived{_RES_SUFFIX}"
EVER_BURNED_MASK_PATH = PERIMS_DERIVED_DIR / "ever_burned_mask.tif"
EVER_BURNED_MASK_LATEST_JSON_PATH = (
    PERIMS_DERIVED_DIR / "ever_burned_mask_LATEST.json"
)
EVER_BURNED_STACK_VRT_PATH = PERIMS_DERIVED_DIR / "dse_stack_1984_2022.vrt"
# Stratification-redesign Stage 2c: single-artefact alternative to the
# fixed-window ever-burned mask. Pixel value = first year (1984..2022)
# the pixel burned per the dse_*.tif stack; nodata if never burned.
# Mask(T) for max_date_year T derived on-the-fly via
# `(first_burn_year != nodata) & (first_burn_year <= T)` — see
# STRATIFICATION_REDESIGN.md §8 Stage 2c + D-2026-05-14.
FIRST_BURN_YEAR_PATH = PERIMS_DERIVED_DIR / "first_burn_year.tif"
FIRST_BURN_YEAR_LATEST_JSON_PATH = (
    PERIMS_DERIVED_DIR / "first_burn_year_LATEST.json"
)
# Stage 2c review C3: separate VRT artefact from m02b's so concurrent
# m02b + m02c invocations don't race on the same scratch path. Both
# scripts rebuild their VRT unconditionally via gdalbuildvrt; without
# separate paths a concurrent rebuild would corrupt one of the two
# readers. Sequential operation is unaffected.
FIRST_BURN_YEAR_STACK_VRT_PATH = (
    PERIMS_DERIVED_DIR / "dse_stack_1984_2022_m02c.vrt"
)
STATES_DIR = MTBS_ROOT / "state_borders"
RAW_STATES_PATH = STATES_DIR / "raw" / "cb_2018_us_state_5m.shp"
STATES_PATH = STATES_DIR / "cleaned" / "states.shp"
HEX_GRID_RAW_PATH = MTBS_ROOT / "hex_grid" / "raw" / "hex_grid.shp"
HEX_GRID_PATH = MTBS_ROOT / "hex_grid" / "cleaned" / "hex_grid.gpkg"
ECO_REGIONS_DIR = MTBS_ROOT / "eco_regions"
RAW_ECO_REGIONS_PATH = ECO_REGIONS_DIR / "raw" / "NA_CEC_Eco_Level3.shp"
ECO_REGIONS_PATH = ECO_REGIONS_DIR / "cleaned" / "eco_regions.gpkg"
ECO_REGIONS_RASTER_PATH = ECO_REGIONS_DIR / "rasters"
NLCD_DIR = MTBS_ROOT / "nlcd"
RAW_NLCD = NLCD_DIR / "raw"
# NLCD subdir + annual filename are env-configurable so a deployment can
# point the analysis at a different NLCD product without editing this
# shared file. Defaults reproduce the legacy "cleaned/" C1V0 layout
# (back-compat — existing users are unaffected). This group's deployment
# sets FIRE_NLCD_SUBDIR=cog + FIRE_NLCD_TIF_FMT="{year}.tif" to use the
# canonical C1V1 driver product (the "C1b repoint").
NLCD_PATH = NLCD_DIR / os.environ.get("FIRE_NLCD_SUBDIR", "cleaned")
NLCD_STACK_VRT_PATH = NLCD_PATH / "nlcd_1984_2022.vrt"
# The static modal-NLCD census raster (one file spanning all years) is a
# *separate* artifact from the per-year NLCD rasters — it is NOT rebuilt into
# the flat-overview ``grouped/`` layout. Give it its own subdir hook so the
# cutover flip (FIRE_NLCD_SUBDIR=grouped, to serve the per-year reads off the
# flat-from-base PATH-B COGs) does not drag the census into ``grouped/`` (no
# nlcd_mode_*.tif there → FileNotFound for every static-mode reader). Unset
# FIRE_NLCD_MODE_SUBDIR falls back to FIRE_NLCD_SUBDIR, so today's deployments
# (incl. FIRE_NLCD_SUBDIR=cog) stay byte-identical (back-compat). At cutover,
# set FIRE_NLCD_MODE_SUBDIR=cog_<N>m to keep the census on its per-res
# precompute while the per-year leg moves to ``grouped/``. See the register
# D-2026-06-15-static-mode-env-hook (option b) + FLAT_OVERVIEW_CUTOVER_PLAN.
_NLCD_MODE_SUBDIR = os.environ.get("FIRE_NLCD_MODE_SUBDIR") or os.environ.get(
    "FIRE_NLCD_SUBDIR", "cleaned"
)
NLCD_MODE_RASTER_PATH = (
    NLCD_DIR / _NLCD_MODE_SUBDIR / "nlcd_mode_1984_2022.tif"
)
ELEVATION_DIR = MTBS_ROOT / "edna"
ELEVATION_RAW_PATH = ELEVATION_DIR / "raw" / "us_orig_dem.tif"
ELEVATION_CLEANED_DIR = ELEVATION_DIR / "cleaned"
# Elevation filename is env-configurable so a deployment can point the
# analysis (m10 via ELEVATION_PATH) at a precomputed analysis-resolution
# DEM (e.g. FIRE_ELEVATION_TIF=edna_dem_120m.tif, built by
# fire_interval.etl.build_coarse_covariates) without editing this shared
# file. Default reproduces the legacy edna_dem.tif (back-compat — Fred's
# other upstream-mtbs users are unaffected). Same pattern as
# FIRE_NLCD_SUBDIR / FIRE_MTBS_BS_SUBDIR above.
ELEVATION_PATH = ELEVATION_DIR / os.environ.get(
    "FIRE_ELEVATION_TIF", "edna_dem.tif"
)
ELEVATION_90M_PATH = ELEVATION_DIR / "edna_dem_90m.tif"
ELEVATION_270M_PATH = ELEVATION_DIR / "edna_dem_270m.tif"

WUI_DIR = MTBS_ROOT / "wui"
RAW_WUI = WUI_DIR / "raw" / "CONUS_WUI_block_1990_2020_change_v4.gdb"
INTERMEDIATE_WUI = WUI_DIR / "intermediate" / "wui.gpkg"
WUI_PATH = WUI_DIR / "cleaned"

# Survival-time rasters are a derived fire-history product (m30 builds them
# from the dse stack), so a coarse rebuild (FIRE_PIXEL_M != 30 m) must isolate
# its output from the 30 m production rasters that a40/a45 read — same
# _RES_SUFFIX treatment as PERIMS_RASTERS_PATH/PERIMS_DERIVED_DIR. Default
# (30 m) resolves to the unchanged "st" dir.
ST_PATH = MTBS_ROOT / f"st{_RES_SUFFIX}"


# --- Formats ---
# Env-configurable (see MTBS_SEVERITY_READ_DIR above); default = legacy
# per-aoi naming. Canonical fde COGs are `{year}.tif`.
MTBS_TIF_FMT = os.environ.get("FIRE_MTBS_TIF_FMT", "mtbs_{aoi}_{year}.tif")
# Env-configurable (see NLCD_PATH above); default = legacy C1V0 naming.
NLCD_TIF_FMT = os.environ.get(
    "FIRE_NLCD_TIF_FMT", "Annual_NLCD_LndCov_{year}_CU_C1V0.tif"
)
TMP_PTS_FMT = "mtbs_{aoi}_{year}"
COMBINED_OUT_FMT = "mtbs_{aoi}_{min_year}_{max_year}"
PIXEL_COUNT_OUT_FMT = "eco_nlcd_mode_pixel_counts_eco{eco_level}.pqt"


# --- Path Builders ---
def get_points_path(year, aoi_code):
    return ROOT_TMP_DIR / TMP_PTS_FMT.format(aoi=aoi_code, year=year)


def get_mtbs_raster_path(year, aoi_code):
    # Reads from the env-configurable severity path (default = legacy
    # cleaned/; this group sets it to the canonical fde cog/). The
    # `{aoi}` placeholder is simply ignored by the canonical `{year}.tif`
    # format string.
    return MTBS_SEVERITY_READ_DIR / MTBS_TIF_FMT.format(
        aoi=aoi_code, year=year
    )


def get_nlcd_raster_path(year):
    return NLCD_PATH / NLCD_TIF_FMT.format(year=year)


def get_wui_flavor_path(year, flavor):
    if year < 2000:
        y = 1990
    elif 2000 <= year < 2010:
        y = 2000
    elif 2010 <= year < 2020:
        y = 2010
    else:
        y = 2020
    return WUI_PATH / f"wui_{flavor}_{y}.tif"


def get_points_combined_path(years, aoi_code):
    return RESULTS_DIR / COMBINED_OUT_FMT.format(
        aoi=aoi_code, min_year=min(years), max_year=max(years)
    )
