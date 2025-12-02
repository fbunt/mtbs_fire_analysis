import argparse
from functools import partial, reduce
from pathlib import Path

import dask
import numpy as np
import polars as pl
import raster_tools as rts
import yaml
from dask.diagnostics import ProgressBar

from mtbs_fire_analysis.pipeline.paths import (
    ECO_REGIONS_RASTER_PATH,
    PERIMS_RASTERS_PATH,
    get_nlcd_raster_path,
)


def or_(a, b):
    return a | b


def and_(a, b):
    return a & b


def main(year, config_path, bp_path, out_path):
    with open(config_path) as fd:
        config = yaml.safe_load(fd)["eco3_config"]
    bp = rts.Raster(bp_path)
    nlcd = rts.Raster(get_nlcd_raster_path(year))
    eco3 = rts.Raster(ECO_REGIONS_RASTER_PATH / "eco_lvl_3.tif")
    burn = rts.Raster(PERIMS_RASTERS_PATH / f"dse_{year}.tif").set_null_value(
        None
    )
    bp_data = bp.data.astype("float64")

    bpt_data = np.where(burn.data >= 1, np.log(np.maximum(bp_data, 1e-9)), np.log(np.maximum(1 - bp_data, 1e-9)))
    valid = ~bp.mask & ~nlcd.mask & ~eco3.mask
    results = []
    counts = []
    for conf in config:
        print(conf["name"])
        ecos = conf["eco_lvl_3"]
        ns = conf["nlcd"]
        selector = reduce(
            and_, [valid, np.isin(eco3.data, ecos), np.isin(nlcd.data, ns)]
        )
        results.append(bpt_data[selector].sum())
        counts.append(selector.sum())
    with ProgressBar():
        print("Computing raw scores")
        results, counts = dask.compute(results, counts)
    results_df = pl.DataFrame(
        {
            "name": [conf["name"] for conf in config],
            "eco3": [conf["eco_lvl_3"] for conf in config],
            "nlcd": [conf["nlcd"] for conf in config],
            "raw_score": results,
            "N": counts,
        }
    ).with_columns(
        score=pl.when(pl.col("N") > 0)
        .then(pl.col("raw_score") / pl.col("N"))
        .otherwise(0)
    )
    results_df.write_parquet(out_path)


def _path(path, check=True):
    path = Path(path)
    if check:
        assert path.exists()
    return path


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "year", type=int, help="The year to calculate burn probability for"
    )
    p.add_argument("config_path", type=_path, help="Config file path")
    p.add_argument("bp_path", type=_path, help="Burn probability raster path")
    p.add_argument(
        "out_path",
        type=partial(_path, check=False),
        help="Output parquet location",
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    assert args.year > 1984
    main(args.year, args.config_path, args.bp_path, args.out_path)
