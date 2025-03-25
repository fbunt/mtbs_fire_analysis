from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from affine import Affine

DEFAULT_GEOHASH_AFFINE = Affine(
    30.0, 0.0, -2406135.0, 0.0, -30.0, 3222585.0, 0.0, 0.0, 1.0
)
DEFAULT_GEOHASH_GRID_SHAPE = (100150, 157144)
DEFAULT_CRS = rio.CRS.from_wkt(
"""
PROJCRS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",
    BASEGEOGCRS["NAD83",
        DATUM["North American Datum 1983",
            ELLIPSOID["GRS 1980",6378137,298.257222101004,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["degree",0.0174532925199433]],
        ID["EPSG",4269]],
    CONVERSION["unnamed",
        METHOD["Albers Equal Area",
            ID["EPSG",9822]],
        PARAMETER["Latitude of false origin",23,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8821]],
        PARAMETER["Longitude of false origin",-96,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8822]],
        PARAMETER["Latitude of 1st standard parallel",29.5,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8823]],
        PARAMETER["Latitude of 2nd standard parallel",45.5,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8824]],
        PARAMETER["Easting at false origin",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8826]],
        PARAMETER["Northing at false origin",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8827]]],
    CS[Cartesian,2],
        AXIS["easting",east,
            ORDER[1],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
        AXIS["northing",north,
            ORDER[2],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]]]
""")


class GridGeohasher:
    def __init__(
        self,
        grid_affine=DEFAULT_GEOHASH_AFFINE,
        grid_shape=DEFAULT_GEOHASH_GRID_SHAPE,
        crs=DEFAULT_CRS,
    ):
        assert isinstance(grid_affine, Affine)
        assert isinstance(grid_shape, (tuple, list))
        assert len(grid_shape) == 2
        self.affine = grid_affine
        self.xres_half = grid_affine.a / 2
        self.yres_half = grid_affine.e / 2
        self.grid_shape = grid_shape
        self.crs = crs

    def geohash(self, geometry):
        x = geometry.x.to_numpy()
        y = geometry.y.to_numpy()
        inv_matrix = np.array(list(~self.affine)).reshape(3, 3)
        n = len(x)
        # x, y, 1 for each column
        xy1 = np.ones((3, n), dtype="float64")
        xy1[0, :] = x
        xy1[1, :] = y

        cr1 = inv_matrix @ xy1
        rows = np.floor(cr1[1]).astype("int64")
        cols = np.floor(cr1[0]).astype("int64")
        return np.ravel_multi_index((rows, cols), self.grid_shape)

    def reverse_geohash(self, geohash):
        if isinstance(geohash, pd.Series):
            index = geohash.index
            geohash = geohash.to_numpy()
        else:
            index = None
            geohash = np.asarray(geohash)
        affine_matrix = np.array(list(self.affine)).reshape(3, 3)
        row, col = np.unravel_index(geohash, self.grid_shape)
        cr1 = np.ones((3, len(geohash)), dtype="float64")
        cr1[0, :] = col
        cr1[1, :] = row
        xy1 = affine_matrix @ cr1
        x = xy1[0] + self.xres_half
        y = xy1[1] + self.yres_half
        return gpd.GeoSeries.from_xy(x, y, index=index, crs=self.crs)


def _add_geom(df, hasher):
    geometry = hasher.reverse_geohash(df["geohash"])
    return gpd.GeoDataFrame(df, geometry=geometry)


def add_geometry_from_geohash(ddf, hasher=None):
    if hasher is None:
        hasher = GridGeohasher()

    func = partial(_add_geom, hasher=hasher)
    meta = ddf._meta.copy()
    meta = gpd.GeoDataFrame(meta, geometry=gpd.GeoSeries([], crs=hasher.crs))
    return ddf.map_partitions(func, meta=meta)

