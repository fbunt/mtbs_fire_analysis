from pathlib import Path

MAIN_FOLDER_ALIAS = Path("/run/media/fire_analysis")

MTBS_ROOT = MAIN_FOLDER_ALIAS / "data"
MTBS_RASTER_DIR = MTBS_ROOT / "mtbs_bs_rasters"
RAW_RASTER_DATA_DIR = MTBS_RASTER_DIR / "raw"
CLEANED_RASTER_DATA_DIR = MTBS_RASTER_DIR / "cleaned"
ROOT_TMP_DIR = MAIN_FOLDER_ALIAS / "data_tmp"
RESULTS_DIR = MAIN_FOLDER_ALIAS / "data" / "results"
CACHE_DIR = MAIN_FOLDER_ALIAS / "data" / "cache"

PERIMS_DIR = MTBS_ROOT / "mtbs_perims"
RAW_PERIMS_PATH = PERIMS_DIR / "raw" / "mtbs_perims_DD.shp"
PERIMS_CLEANED_DIR = PERIMS_DIR / "cleaned"
PERIMS_PATH = PERIMS_CLEANED_DIR / "mtbs_perims_trimmed.gpkg"
PERIMS_BY_YEAR_PATH = PERIMS_CLEANED_DIR / "mtbs_perims_by_year.gpkg"
PERIMS_RASTERS_PATH = PERIMS_DIR / "rasters"
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
NLCD_PATH = NLCD_DIR / "cleaned"
NLCD_MODE_RASTER_PATH = NLCD_PATH / "nlcd_mode_1984_2022.tif"

WUI_DIR = MTBS_ROOT / "wui"
RAW_WUI = WUI_DIR / "raw" / "CONUS_WUI_block_1990_2020_change_v4.gdb"
INTERMEDIATE_WUI = WUI_DIR / "intermediate" / "wui.gpkg"
WUI_PATH = WUI_DIR / "cleaned"

ST_PATH = MTBS_ROOT / "st"


# --- Formats ---
MTBS_TIF_FMT = "mtbs_{aoi}_{year}.tif"
NLCD_TIF_FMT = "Annual_NLCD_LndCov_{year}_CU_C1V0.tif"
TMP_PTS_FMT = "mtbs_{aoi}_{year}"
COMBINED_OUT_FMT = "mtbs_{aoi}_{min_year}_{max_year}"
PIXEL_COUNT_OUT_FMT = "eco_nlcd_mode_pixel_counts_eco{eco_level}.pqt"


# --- Path Builders ---
def get_points_path(year, aoi_code):
    return ROOT_TMP_DIR / TMP_PTS_FMT.format(aoi=aoi_code, year=year)


def get_mtbs_raster_path(year, aoi_code):
    return CLEANED_RASTER_DATA_DIR / MTBS_TIF_FMT.format(
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
