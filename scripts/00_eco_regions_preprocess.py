# Preprocess eco regions
# ref: https://www.epa.gov/eco-research/ecoregions-north-america
import geopandas as gpd

from defaults import DEFAULT_CRS
from paths import ECO_REGIONS_PATH, RAW_ECO_REGIONS_PATH

if __name__ == "__main__":
    eco_full_df = gpd.read_file(RAW_ECO_REGIONS_PATH).to_crs(DEFAULT_CRS)
    # Drop water
    eco_full_df = eco_full_df[eco_full_df.NA_L1CODE != "0"]
    code123 = (
        # Split "XX.Y.ZZ --> ["XX", "Y", "ZZ"]
        eco_full_df.NA_L3CODE.str.split(".", n=2)
        # ["XX", "Y", "ZZ"] --> [XX, Y, ZZ]
        .map(lambda x: list(map(int, x)))
        # [XX, Y, ZZ] --> XXYZZ
        .map(lambda x: (x[0] * 1000) + (x[1] * 100) + x[2])
        .astype("int16")
    )
    eco_lvl_1 = (code123 // 1000).astype("uint8")
    eco_lvl_2 = (code123 // 100).astype("uint8")
    eco_lvl_3 = code123.copy()
    out_df = gpd.GeoDataFrame(
        {
            "eco_lvl_1": eco_lvl_1,
            "eco_lvl_2": eco_lvl_2,
            "eco_lvl_3": eco_lvl_3,
            "geometry": eco_full_df.geometry
        },
    )
    out_df.to_file(ECO_REGIONS_PATH)
