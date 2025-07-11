import geopandas as gpd

from mtbs_fire_analysis.defaults import DEFAULT_CRS
from mtbs_fire_analysis.pipeline.paths import HEX_GRID_PATH, HEX_GRID_RAW_PATH


def main():
    raw_hex_grid = gpd.read_file(HEX_GRID_RAW_PATH).to_crs(DEFAULT_CRS)
    hex = raw_hex_grid.drop(columns=["id", "left", "right", "top", "bottom"])
    # Start IDs at 0 instead of 1
    hex["hexel_id"] = raw_hex_grid["id"].astype("int16") - 1
    hex.to_file(HEX_GRID_PATH)


if __name__ == "__main__":
    main()
