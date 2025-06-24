import geopandas as gpd
import polars as pl
import yaml

from mtbs_fire_analysis.pipeline.paths import CACHE_DIR

eco_lvl = 3

# get event histories
event_histories = pl.read_parquet(
    CACHE_DIR / "event_histories_2023-01-01.parquet"
)


num_fire_pixels = (
    event_histories.explode(["Ig_Date", "Event_ID", "nlcd"])
    .group_by(["eco_lvl_3", "nlcd"])
    .agg(
        pl.col("Pixel_Count").sum().alias("Fire Time Pixels"),
        pl.col("Event_ID").n_unique().alias("Num Fires"),
    )
    .sort("eco_lvl_3", "nlcd")
)

nlcd_map = pl.DataFrame(
    {
        "nlcd": [41, 42, 43] + [52] + [71] + [90, 95],
        "nlcd_name": ["Forest"] * 3
        + ["Schrub"]
        + ["Grassland"]
        + ["Wetland"] * 2,
    }
)

pix_counts = pl.read_parquet(
    "mtbs_fire_analysis/data/eco_nlcd_pixel_counts_2003_3.pqt"
).rename(
    {
        "count": "Total num pixels",
    }
)

if eco_lvl != 3:
    if eco_lvl == 1:
        div = 1000
    if eco_lvl == 2:
        div = 100
    pix_counts = (
        pix_counts.with_columns(pl.col("eco_lvl_3").cast(pl.Int64) // div)
        .rename({"eco_lvl_3": "eco_lvl_3"})
        .drop_nulls(subset=["eco_lvl_3"])
    )


summary = (
    num_fire_pixels.join(pix_counts, on=["eco_lvl_3", "nlcd"], how="inner")
    .join(nlcd_map, on="nlcd", how="inner")
    .sort("eco_lvl_3", "nlcd")
    .group_by(["eco_lvl_3", "nlcd_name"])
    .agg(
        pl.col("Fire Time Pixels").sum().alias("Total Fire Time Pixels"),
        pl.col("Num Fires").sum().alias("Total Num Fires"),
        pl.col("Total num pixels").sum().alias("Total num pixels"),
        pl.col("nlcd").alias("nlcds"),
    )
    .with_columns(
        (
            38 * pl.col("Total num pixels") / pl.col("Total Fire Time Pixels")
        ).alias("Naive Fire Interval (years)"),
        (pl.col("Total num pixels") * 900 / 10_000).alias("Total Area (ha)"),
    )
    .sort("Naive Fire Interval (years)", descending=True)
)

gdf = gpd.read_file(
    "mtbs_fire_analysis/data/NA_CEC_Eco_Level3/NA_CEC_Eco_Level3.shp"
)
lvl3_codes = gdf[["NA_L3CODE", "NA_L3NAME"]].drop_duplicates()

gdf[["NA_L3CODE", "NA_L3NAME"]].drop_duplicates()
lvl3_codes["IntCode"] = (
    # Split "XX.Y.ZZ --> ["XX", "Y", "ZZ"]
    lvl3_codes.NA_L3CODE.str.split(".", n=2)
    # ["XX", "Y", "ZZ"] --> [XX, Y, ZZ]
    .map(lambda x: list(map(int, x)))
    # [XX, Y, ZZ] --> XXYZZ
    .map(lambda x: (x[0] * 1000) + (x[1] * 100) + x[2])
    .astype("int32")
)

lvl3_codes = pl.DataFrame(lvl3_codes)
# Swap out "/" for "+" in NA_L3NAME to avoid issues with writing files later
lvl3_codes = lvl3_codes.with_columns(
    pl.col("NA_L3NAME").str.replace_all(r"/", r"+")
)

summary = (
    summary.join(
        lvl3_codes, left_on="eco_lvl_3", right_on="IntCode", how="inner"
    )
    .drop("NA_L3CODE")
    .rename({"NA_L3NAME": "Eco Name"})
)

out_order = [
    "eco_lvl_3",
    "nlcds",
    "Eco Name",
    "nlcd_name",
    "Total Fire Time Pixels",
    "Total Num Fires",
    "Total num pixels",
    "Naive Fire Interval (years)",
    "Total Area (ha)",
]

candidates = (
    summary.filter(
        (pl.col("Total Num Fires") > 100)
        & (pl.col("Naive Fire Interval (years)") < 125)
        & (pl.col("Total Area (ha)") > 100_000)
    )
    .select(out_order)
    .sort("Naive Fire Interval (years)", descending=True)
)

# Why these filters?
# Area to not waste time on small areas
# Total Num Fires will reduce variance in our fits
# (ie. our confidence intervals will be smaller when resampling)
# Naive Fire Interval because we only have 38 years of data,
# so we will have poor estimates for intervals significantly larger than that.
# Likely we can relax the interval filter if we add covariates,
# as we will allocate large intervals to combinations of covariates that don't
# burn, while still having enough data where there are fires


# push candidates to yaml config file

with open("mtbs_fire_analysis/data/eco_nlcd_candidates.yaml", "w") as f:
    yaml.dump(
        {
            "main2": [
                {
                    "name": f"{row['Eco Name']} : {row['nlcd_name']}",
                    "eco_lvl_3": [row["eco_lvl_3"]],
                    "nlcd": row["nlcds"],
                }
                for row in candidates.to_dicts()
            ]
        },
        f,
        default_flow_style=False,
        sort_keys=False,
    )

# push summaries to csv
summary.select(pl.exclude("nlcds")).write_csv(
    "mtbs_fire_analysis/data/eco_nlcd_summary.csv",
    include_header=True,
    separator=",",
)
