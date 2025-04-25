from pathlib import Path

MTBS_ROOT = Path("/var/mnt/fastdata02/mtbs")
MTBS_RASTER_DIR = MTBS_ROOT / "mtbs_bs_rasters"
RAW_RASTER_DATA_DIR = MTBS_RASTER_DIR / "raw"
CLEANED_RASTER_DATA_DIR = MTBS_RASTER_DIR / "cleaned"
ROOT_TMP_DIR = Path("/var/mnt/fastdata01/fire_tmp/")
RESULTS_DIR = Path("/var/mnt/fastdata02/mtbs/results")

PERIMS_DIR = MTBS_ROOT / "mtbs_perims"
RAW_PERIMS_PATH = PERIMS_DIR / "raw" / "mtbs_perims_DD.shp"
PERIMS_PATH = PERIMS_DIR / "cleaned" / "mtbs_perims_trimmed.pqt"
STATES_DIR = MTBS_ROOT / "state_borders"
RAW_STATES_PATH = STATES_DIR / "raw" / "cb_2018_us_state_5m.shp"
STATES_PATH = STATES_DIR / "cleaned" / "states.shp"
ECO_REGIONS_DIR = MTBS_ROOT / "eco_regions"
RAW_ECO_REGIONS_PATH = ECO_REGIONS_DIR / "raw" / "NA_CEC_Eco_Level3.shp"
ECO_REGIONS_PATH = ECO_REGIONS_DIR / "cleaned" / "eco_regions.shp"
NLCD_DIR = MTBS_ROOT / "nlcd"
RAW_NLCD = NLCD_DIR / "raw"
NLCD_PATH = NLCD_DIR / "cleaned"

WUI_DIR = MTBS_ROOT / "wui"
RAW_WUI = WUI_DIR / "raw" / "CONUS_WUI_block_1990_2020_change_v4.gdb"
INTERMEDIATE_WUI = WUI_DIR / "intermediate" / "wui.gpkg"
WUI_PATH = WUI_DIR / "cleaned"
