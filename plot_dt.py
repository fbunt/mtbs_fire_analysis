import argparse
import itertools

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


def flatmap(func, iterable):
    return itertools.chain.from_iterable(map(func, iterable))


def _if_names(col):
    return f"{col}1", f"{col}2"


def _if_exprs(col):
    name1, name2 = _if_names(col)
    return pl.col(col).alias(name1), pl.col(col).shift(-1).alias(name2)


def build_dts_df(lf, if_cols=None):
    """Build a dataframe of times between fires (dt).

    Parameters
    ----------
    lf : polars.LazyFrame, polars.DataFrame
        The dataframe to process.
    if_cols : list, optional
        The columns to get initial and final (if) values for at each dt. The
        resulting dataframe will two columns for each input column. As an
        example, if 'nlcd' is one of the input columns, the result will have
        `'nlcd1'` and `'nlcd2'`.

    Returns
    -------
    polars.LazyFrame, polars.DataFrame

    """
    if_cols = if_cols or []
    assert "bs" not in if_cols
    if_cols = ["bs"] + if_cols
    return (
        lf.group_by("geohash")
        .agg(
            pl.len().alias("n"),
            pl.col("eco_lvl_1").first().alias("eco"),
            pl.col("Ig_Date", *if_cols),
        )
        .filter(pl.col("n") >= 2)
        .select(pl.exclude("n"))
        .explode("Ig_Date", *if_cols)
        .sort("geohash", "Ig_Date")
        .group_by("geohash")
        .agg(
            pl.col("eco").first(),
            pl.col("Ig_Date").diff().shift(-1).dt.total_days().alias("dt")
            / 365,
            *flatmap(_if_exprs, if_cols),
        )
        .explode("dt", *flatmap(_if_names, if_cols))
        .drop_nulls()
    )


def basic_hist(ax, df, bs):
    dt = df.select(pl.col("dt")).to_numpy().flatten()
    ax.hist(dt, bins=np.arange(0, 40))
    ax.vlines(dt.mean(), 0, 1, transform=ax.get_xaxis_transform(), colors="r")
    ax.text(0.75, 0.8, f"Severity: {bs_to_str[bs]}", transform=ax.transAxes)
    ax.set_xlim([0, 39])


def sns_hist(ax, df, bs, weighting=False):
    weights = None
    if weighting:
        df = df.with_columns(w=(38 / (38 - pl.col("dt").cast(pl.Int32))))
        weights = "w"
    pdf = df.to_pandas()
    sns.histplot(
        data=pdf, x="dt", weights=weights, bins=list(np.arange(0, 39)), ax=ax
    )
    ax.set_xlim([0, 39])


def sns_hist_stacked(ax, df, bs, weighting=False):
    weights = None
    if weighting:
        df = df.with_columns(w=(38 / (38 - pl.col("dt").cast(pl.Int32))))
        weights = "w"
    pdf = df.to_pandas()
    legend = df.select(pl.first("eco")).item() == 13 and bs == 1
    sns.histplot(
        data=pdf,
        x="dt",
        hue="bs2",
        weights=weights,
        bins=list(np.arange(0, 39)),
        multiple="stack",
        ax=ax,
        legend=legend,
    )
    if legend:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.1, 2))
    # ax.text(0.8, 0.8, f"Severity: {bs}", transform=ax.transAxes)
    ax.set_xlim([0, 39])


def sns_cdf(ax, df, bs):
    pdf = df.to_pandas()
    legend = df.select(pl.first("eco")).item() == 13 and bs == 1
    sns.ecdfplot(
        data=pdf,
        x="dt",
        hue="bs2",
        stat="count",
        ax=ax,
        legend=legend,
    )
    if legend:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.1, 2))
    # ax.text(0.8, 0.8, f"Severity: {bs}", transform=ax.transAxes)
    ax.set_xlim([0, 39])


def make_basic_bar_plots(dts, ax_func, **kwargs):
    ecos = (
        dts.filter(pl.col("eco") != 7)
        .select("eco")
        .unique()
        .sort("eco")
        .to_numpy()
        .flatten()
    )
    bss = [1, 2, 3, 4]
    eco_to_bs_to_dts = {}
    for eco in ecos:
        eco_to_bs_to_dts[eco] = {}
        for bs in bss:
            eco_to_bs_to_dts[eco][bs] = dts.filter(
                (pl.col("eco") == eco) & (pl.col("bs1") == bs)
            )

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3, figure=fig)
    for k, eco in enumerate(ecos):
        i, j = np.unravel_index(k, (2, 3))
        gsij = gs[i, j].subgridspec(4, 1)
        for n, bs in enumerate(bss):
            df = eco_to_bs_to_dts[eco][bs]
            ax = fig.add_subplot(gsij[n, 0])
            ax_func(ax, df, bs, **kwargs)
        ax.set_xlabel("dt (years)")
        axx = fig.add_subplot(gsij[:])
        axx.axis("off")
        axx.set_title(f"{eco_to_str[eco]}: Severity = 1 | 2 | 3 | 4")


sns.set_theme(style="ticks")

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
bs_to_str = {1: "very low", 2: "low", 3: "med", 4: "high"}


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("data_loc", help="Data location")
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    lf = pl.scan_parquet(args.data_loc)
    ldts = build_dts_df(lf)
    dts = ldts.collect()
    make_basic_bar_plots(dts, sns_hist_stacked)
    plt.show()
