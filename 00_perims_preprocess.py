import geopandas as gpd
import numpy as np

from defaults import DEFAULT_CRS
from paths import PERIMS_PATH, RAW_PERIMS_PATH

INCID_TYPE_MAPPING = {
    "Unknown": 0,
    "Wildfire": 1,
    "Prescribed Fire": 2,
    "Wildland Fire Use": 3,
}
ASMNT_TYPE = {
    "Initial": 0,
    "Initial (SS)": 1,
    "Extended": 2,
    "Extended (SS)": 3,
}
MAP_PROG_MAPPING = {"MTBS": 0}
MAPPINGS = {
    "Incid_Type": INCID_TYPE_MAPPING,
    "Asmnt_Type": ASMNT_TYPE,
    "Map_Prog": MAP_PROG_MAPPING,
}

DROP_COLUMNS = ["Comment"]


if __name__ == "__main__":
    perims = gpd.read_file(RAW_PERIMS_PATH).to_crs(DEFAULT_CRS)
    perims = perims.drop(DROP_COLUMNS, axis=1)
    for k, mapping in MAPPINGS.items():
        perims[k] = (
            perims[k]
            .apply(lambda x, mapping=mapping: mapping[x])
            .astype(np.uint8)
        )
    perims["area_m2"] = perims.area
    perims["area_acres"] = perims.area * 0.0002471054
    perims.to_parquet(PERIMS_PATH)
