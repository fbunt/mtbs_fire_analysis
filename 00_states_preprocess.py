import geopandas as gpd
import pandas as pd

import raster_tools as rts
from paths import RAW_RASTER_DATA_DIR, RAW_STATES_PATH, STATES_PATH

if __name__ == "__main__":
    states = gpd.read_file(RAW_STATES_PATH).sort_values("GEOID")
    states = states[states.GEOID.astype(int) < 60]
    states = states.to_crs(
        rts.Raster(RAW_RASTER_DATA_DIR / "1984" / "mtbs_CONUS_1984.tif").crs
    )
    wus_states = states[
        states.STUSPS.isin(
            ["WA", "OR", "ID", "CA", "MT", "NV", "WY", "UT", "AZ", "NM", "CO"]
        )
    ]
    wus_geom = wus_states.geometry.union_all()
    wus = gpd.GeoDataFrame(
        {
            "STATEFP": "--",
            "STATENS": "--",
            "AFFGEOID": "0000000US90",
            "GEOID": "90",
            "STUSPS": "WUS",
            "NAME": "Western US (WA, OR, ID, CA, MT, NV, WY, UT, AZ, NM, CO)",
            "LSAD": "00",
            "ALAND": wus_states.ALAND.sum(),
            "AWATER": wus_states.AWATER.sum(),
            "geometry": [wus_geom],
        },
        crs=states.crs,
        index=[99],
    )
    conus_states = states[~states.STUSPS.isin(["AK", "HI"])]
    conus_geom = conus_states.union_all()
    conus = gpd.GeoDataFrame(
        {
            "STATEFP": "--",
            "STATENS": "--",
            "AFFGEOID": "0000000US99",
            "GEOID": "99",
            "STUSPS": "CONUS",
            "NAME": "CONUS",
            "LSAD": "00",
            "ALAND": conus_states.ALAND.sum(),
            "AWATER": conus_states.AWATER.sum(),
            "geometry": [conus_geom],
        },
        crs=states.crs,
        index=[99],
    )
    states = pd.concat([states, wus, conus])
    states.to_file(STATES_PATH)
