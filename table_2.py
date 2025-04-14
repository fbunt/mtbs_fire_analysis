import polars as pl

eco_to_str = {
    5: "Northern Forest",
    6: "NW Forested Mountains",
    7: "Marine West Coast Forest",
    8: "Eastern Temperate Forests",
    9: "Great Plains",
    10: "North American Deserts",
    11: "Mediterranean California",
    12: "Southern Semiarid Highlands",
    13: "Temperate Sierras",
}

lf = pl.scan_parquet("../results/mtbs_CONUS_1984_2022").rename(
    {"eco_lvl_1": "eco"}
)
sites_per_eco = (
    lf.group_by("eco")
    .agg(pl.col("geohash").n_unique().alias("n_sites"))
    .sort("eco")
    .collect()
    # water
    .filter(pl.col("eco") != 0)
)
burns_per_year_eco = (
    lf.select("year", "geohash", "eco")
    .group_by("year", "eco")
    .agg(pl.col("geohash").n_unique().alias("n_burn_sites"))
    .collect()
    .sort("year", "eco")
)
burns_per_year_eco = (
    burns_per_year_eco.join(sites_per_eco, on="eco")
    .with_columns(
        percent_sites_burned=pl.col("n_burn_sites") / pl.col("n_sites") * 100
    )
    .select(pl.exclude("n_sites"))
)
tbl = (
    burns_per_year_eco
    # .filter(pl.col("eco") != 7)
    .with_columns(p_rounded=pl.col("percent_sites_burned").round())
    .group_by("eco")
    .agg(
        pl.col("year", "percent_sites_burned", "p_rounded").filter(
            pl.col("p_rounded") == pl.col("p_rounded").max()
        )
    )
    .select(pl.exclude("p_rounded"))
    .explode("year", "percent_sites_burned")
    .group_by("eco")
    .agg(pl.all().top_k_by("percent_sites_burned", 3))
    .explode("year", "percent_sites_burned")
    .sort("eco", "year")
    .group_by("eco")
    .agg(pl.all())
    .join(
        pl.DataFrame(
            {"eco": list(eco_to_str), "eco_name": list(eco_to_str.values())}
        ),
        on="eco",
    )
    .select("eco", "eco_name", "year", "percent_sites_burned")
)

parks_tbl2 = pl.DataFrame(
    {
        "eco": [8, 9, 11, 10, 5, 6, 12, 13],
        "parks_year": [
            [2019],
            [2012],
            [1985, 2017],
            [2011],
            [2008, 2015, 2018],
            [2021],
            [2020],
            [2012],
        ],
        "parks % sites burned": [6, 19, 20, 10, 1, 8, 22, 8],
    }
)

tbl2_comp = tbl.join(parks_tbl2, on="eco")
