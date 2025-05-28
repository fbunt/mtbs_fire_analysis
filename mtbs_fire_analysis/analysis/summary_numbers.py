import polars as pl
import yaml

from pathlib import Path

from mtbs_fire_analysis.analysis.utils import (
    get_dt_polygons,
    get_dts,
    get_st_polygons,
    get_sts,
)

configs = Path("mtbs_fire_analysis") / "analysis" / "configs.yaml"

with open(configs) as f:
    config_data = yaml.safe_load(f)

configs = config_data["main"]


data_path = Path("mtbs_fire_analysis") / "data"

pixel_counts_path = data_path / "pixel_counts.csv"


dts = get_dts()
sts = get_sts()
dt_polygons = get_dt_polygons()
st_polygons = get_st_polygons(refresh=True)

pixel_counts = pl.read_csv(pixel_counts_path, separator="\t")

# Get counts of each of the eco, nlcd combinations from each of the above and put in one table

records = []

for config in configs:
    sub_dt_polygons = dt_polygons.filter(
        (pl.col("eco").is_in(config["eco"]))
        & (pl.col("nlcd2").is_in(config["nlcd"]))
    )
    num_dts = sub_dt_polygons.get_column("Pixel Count").sum()
    num_dt_polygons = sub_dt_polygons.shape[0]
    sub_st_polygons = st_polygons.filter(
        (pl.col("eco").is_in(config["eco"]))
        & (pl.col("nlcd").is_in(config["nlcd"]))
    )
    num_sts = sub_st_polygons.get_column("Pixel Count").sum()
    num_st_polygons = sub_st_polygons.shape[0]
    num_pixels_total = pixel_counts.filter(
        (pl.col("eco").is_in(config["eco"]))
        & (pl.col("nlcd").is_in(config["nlcd"]))
    ).get_column("count").sum()
    records.extend(
        [
            {
                "Config": config["name"],
                "dt_pixels": num_dts,
                "dt_polygons": num_dt_polygons,
                "st_pixels": num_sts,
                "st_polygons": num_st_polygons,
                "total_pixels": num_pixels_total,
                "empty_pixels": num_pixels_total - num_sts,
                "pc_empty": (num_pixels_total - num_sts) / num_pixels_total,
            }
        ]
    )

summary_tbl = pl.from_dicts(records).with_columns(
    pl.col("Config"),
    pl.col("dt_pixels").cast(pl.Int32),
    pl.col("dt_polygons").cast(pl.Int32),
    pl.col("st_pixels").cast(pl.Int32),
    pl.col("st_polygons").cast(pl.Int32),
    pl.col("total_pixels").cast(pl.Int32),
    pl.col("empty_pixels").cast(pl.Int32),
    pl.col("pc_empty").cast(pl.Float32),
)
