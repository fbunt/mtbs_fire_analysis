import datetime

import geopandas as gpd
import numpy as np

from mtbs_fire_analysis.defaults import DEFAULT_CRS
from mtbs_fire_analysis.pipeline.paths import (
    PERIMS_BY_YEAR_PATH,
    PERIMS_PATH,
    RAW_PERIMS_PATH,
)

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
MAPPINGS = {
    "Incid_Type": INCID_TYPE_MAPPING,
    "Asmnt_Type": ASMNT_TYPE,
}

DROP_COLUMNS = [
    "irwinID",
    "Map_ID",
    "Map_Prog",
    "BurnBndAc",
    "BurnBndLat",
    "BurnBndLon",
    "Pre_ID",
    "Post_ID",
    "Perim_ID",
    "NoData_T",
    "Comment",
]


def clean_perims():
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
    perims["days_since_epoch"] = (
        perims.Ig_Date - datetime.datetime(1970, 1, 1)
    ).dt.days.astype("int16")
    perims.to_file(PERIMS_PATH)


def split_perims_by_year():
    print("Splitting perims by year")
    perims = gpd.read_file(PERIMS_PATH).sort_values("Ig_Date")
    for y, grp in perims.groupby(perims.Ig_Date.dt.year):
        if y in set(range(1984, 2023)):
            print(f"Saving {y}")
            grp.to_file(PERIMS_BY_YEAR_PATH, layer=str(y))


def main():
    clean_perims()
    split_perims_by_year()


if __name__ == "__main__":
    main()
