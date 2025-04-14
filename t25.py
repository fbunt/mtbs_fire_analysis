import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


def build_dts_df(lf):
    return (
        lf.group_by("geohash")
        .agg(
            pl.len().alias("n"),
            pl.col("eco_lvl_1").first().alias("eco"),
            pl.col("Ig_Date", "bs"),
        )
        .filter(pl.col("n") >= 2)
        .select(pl.exclude("n"))
        .explode("Ig_Date", "bs")
        .sort("geohash", "Ig_Date")
        .group_by("geohash")
        .agg(
            pl.col("eco").first(),
            pl.col("Ig_Date").diff().shift(-1).dt.total_days().alias("dt")
            / 365,
            pl.col("bs").alias("bs1"),
            pl.col("bs").shift(-1).alias("bs2"),
        )
        .explode("dt", "bs1", "bs2")
        .drop_nulls()
    )


def ecdf(v, n=None):
    x, counts = np.unique(v, return_counts=True)
    cs = np.cumsum(counts)
    y = cs / cs[-1] if n is None else cs / n
    return x, y


def ecdf_norm_value(v):
    x, counts = np.unique(v, return_counts=True)
    cs = np.cumsum(counts)
    return cs[-1]


def find_time_to_p(x, y, p):
    assert p <= 1
    try:
        return x[y > p][0]
    except IndexError:
        return None


def time_to_p_from(dts, p, n=None):
    return find_time_to_p(*ecdf(dts.select("dt").to_numpy().flatten(), n), p)


ecos = [6, 9, 10, 11, 12, 13]
eco_to_str = {
    6: "NW Forested Mountains",
    7: "Marine West Coast Forest",
    9: "Great Plains",
    10: "North American Deserts",
    11: "Mediterranean California",
    12: "Southern Semiarid Highlands",
    13: "Temperate Sierras",
}
bss = [1, 2, 3, 4]
bs_to_str = {1: "very low", 2: "low", 3: "med", 4: "high"}

lf = pl.scan_parquet("../results/mtbs_WUS_1984_2022")
ldts = build_dts_df(lf)
dts = ldts.collect()
eco_to_dts = {eco: dts.filter(pl.col("eco") == eco) for eco in ecos}
eco_to_n = {
    eco: ecdf_norm_value(df.select("dt").to_numpy().flatten())
    for eco, df in eco_to_dts.items()
}
eco_to_bs_to_dts = {}
for eco in ecos:
    eco_to_bs_to_dts[eco] = {}
    for bs in bss:
        eco_to_bs_to_dts[eco][bs] = dts.filter(
            (pl.col("eco") == eco) & (pl.col("bs1") == bs)
        )
bs_to_eco_to_dts = {}
for bs in bss:
    bs_to_eco_to_dts[bs] = {}
    for eco in ecos:
        bs_to_eco_to_dts[bs][eco] = eco_to_bs_to_dts[eco][bs]
tdf = pl.DataFrame(
    {
        "eco": ecos,
        "eco_str": [eco_to_str[eco] for eco in ecos],
        "t25": [time_to_p_from(df, 0.25) for eco, df in eco_to_dts.items()],
        **{
            f"t25_bs={bs}": [
                time_to_p_from(df, 0.25)  # , eco_to_n[eco])
                for df in bs_to_eco_to_dts[bs].values()
            ]
            for bs in bss
        },
        "t50": [time_to_p_from(df, 0.5) for eco, df in eco_to_dts.items()],
        **{
            f"t50_bs={bs}": [
                time_to_p_from(df, 0.5)  # , eco_to_n[eco])
                for df in bs_to_eco_to_dts[bs].values()
            ]
            for bs in bss
        },
    }
)
