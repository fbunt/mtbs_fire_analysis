import dask
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
from dask.diagnostics import ProgressBar
from paths import INTERMEDIATE_WUI, RAW_WUI

# Water               : open water
# Uninhabited_Veg     : housing density = 0 & wildland vegetation > 50%
# Uninhabited_NoVeg   : housing density = 0 & wildland vegetation ≤ 50%
# Very_Low_Dens_Veg   : housing density < 6.177635 & wildland vegetation > 50%
# Very_Low_Dens_NoVeg : housing density < 6.177635 & wildland vegetation ≤ 50%
# Low_Dens_Intermix   : housing density between 6.177635
#                       & 49.42108 & wildland vegetation > 50%
# Low_Dens_Interface  : housing density between 6.177635
#                       & 49.42108 & wildland vegetation ≤ 50%
#                       & within 2.414km of area with ≥ 75% wildland vegetation
# Low_Dens_NoVeg      : housing density between 6.177635
#                       & 49.42108 & wildland vegetation ≤ 50%
# Med_Dens_Intermix   : housing density between 49.42108
#                       & 741.3162 & wildland vegetation > 50%
# Med_Dens_Interface  : housing density between 49.42108
#                       & 741.3162 & wildland vegetation ≤ 50%
#                       & within 2.414km of area with ≥ 75% wildland vegetation
# Med_Dens_NoVeg      : housing density between 49.42108
#                       & 741.3162 & wildland vegetation ≤ 50%
# High_Dens_Intermix  : housing density ≥ 741.3162
#                       & wildland vegetation > 50%
# High_Dens_Interface : housing density ≥ 741.3162
#                       & wildland vegetation ≤ 50%
#                       & within 2.414km of area with ≥ 75% wildland vegetation
# High_Dens_NoVeg     : housing density ≥ 741.3162 & wildland vegetation ≤ 50%


WUI_CLASS_MAPPING = {
    "Water": np.uint8(0),
    "Uninhabited_Veg": np.uint8(11),
    "Uninhabited_NoVeg": np.uint8(12),
    "Very_Low_Dens_Veg": np.uint8(21),
    "Very_Low_Dens_NoVeg": np.uint8(22),
    "Low_Dens_Intermix": np.uint8(33),
    "Low_Dens_Interface": np.uint8(34),
    "Low_Dens_NoVeg": np.uint8(35),
    "Med_Dens_Intermix": np.uint8(43),
    "Med_Dens_Interface": np.uint8(44),
    "Med_Dens_NoVeg": np.uint8(45),
    "High_Dens_Intermix": np.uint8(53),
    "High_Dens_Interface": np.uint8(54),
    "High_Dens_NoVeg": np.uint8(55),
}
KEEP_COLS = [
    "WUICLASS1990",
    "WUICLASS2000",
    "WUICLASS2010",
    "WUICLASS2020",
    "WUIFLAG1990",
    "WUIFLAG2000",
    "WUIFLAG2010",
    "WUIFLAG2020",
]


def _renamer(name):
    if name == "geometry":
        return "geometry"

    year = name[-4:]
    if "CLASS" in name:
        new_name = f"wui_class_{year}"
    elif "FLAG" in name:
        new_name = f"wui_flag_{year}"
    else:
        new_name = name
    return new_name


def load_raw_wui_gdf(path):
    print("Loading WUI data from raw")
    ddf = dgpd.read_file(path, npartitions=20)[KEEP_COLS + ["geometry"]]
    parts = list(ddf.partitions)
    with ProgressBar():
        (parts,) = dask.compute(parts)
    print("Fixing invalid geometries")
    for part in tqdm.tqdm(parts, ncols=80):
        part["geometry"] = part.geometry.make_valid()
    frame = pd.concat(parts)
    parts = None
    print("Mapping strings to ints")
    for col in [col for col in KEEP_COLS if "CLASS" in col]:
        frame[col] = frame[col].map(WUI_CLASS_MAPPING)
    frame = frame.astype(dict.fromkeys(KEEP_COLS, np.uint8))
    return frame.rename(columns=_renamer)


def write_wui_gdf_to_gpkg(gdf, gpkg_path):
    print("Writing WUI data to intermediate gpkg")
    gdf.to_file(gpkg_path, driver="GPKG", layer="wui")
    print("Done")


def main(in_path, out_path):
    write_wui_gdf_to_gpkg(load_raw_wui_gdf(in_path), out_path)


if __name__ == "__main__":
    main(RAW_WUI, INTERMEDIATE_WUI)
