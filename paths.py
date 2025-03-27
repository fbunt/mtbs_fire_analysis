from pathlib import Path

MTBS_ROOT = Path("/var/mnt/fastdata02/mtbs")
RAW_RASTER_DATA_DIR = MTBS_ROOT / "MTBS_BSmosaics"
CLEANED_RASTER_DATA_DIR = MTBS_ROOT / "MTBS_BSmosaics_cleaned"
ROOT_TMP_DIR = Path("/var/mnt/fastdata01/fire_tmp/")
RESULTS_DIR = Path("/var/mnt/fastdata02/mtbs/results")

RAW_PERIMS_PATH = MTBS_ROOT / "mtbs_perims" / "mtbs_perims_DD.shp"
PERIMS_PATH = MTBS_ROOT / "mtbs_perims" / "mtbs_perims_trimmed.pqt"
RAW_STATES_PATH = MTBS_ROOT / "state_borders" / "cb_2018_us_state_5m.shp"
STATES_PATH = MTBS_ROOT / "state_borders" / "states.shp"
RAW_ECO_REGIONS_PATH = MTBS_ROOT / "eco_regions" / "NA_CEC_Eco_Level3.shp"
ECO_REGIONS_PATH = MTBS_ROOT / "eco_regions" / "eco_regions.shp"
