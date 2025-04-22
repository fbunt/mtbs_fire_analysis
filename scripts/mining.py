import itertools

import polars as pl


def flatmap(func, iterable):
    return itertools.chain.from_iterable(map(func, iterable))


def _extra_names(col):
    return f"{col}1", f"{col}2"


def _extra_exprs(col):
    name1, name2 = _extra_names(col)
    return pl.col(col).alias(name1), pl.col(col).shift(-1).alias(name2)


def build_dts_df(lf, extra_cols=None):
    """Build a dataframe of times between fires (dt).

    Parameters
    ----------
    lf : polars.LazyFrame, polars.DataFrame
        The dataframe to process.
    extra_cols : list, optional
        The extra columns to get initial and final values for at each dt. The
        resulting dataframe will have two columns for each input column. As an
        example, if 'nlcd' is one of the input columns, the result will have
        `'nlcd1'` and `'nlcd2'`.

    Returns
    -------
    polars.LazyFrame, polars.DataFrame

    """
    extra_cols = extra_cols or []
    assert "bs" not in extra_cols
    extra_cols = ["bs"] + extra_cols
    ig_dt = "_ig_date_"
    return (
        lf.group_by("geohash")
        .agg(
            pl.len().alias("n"),
            pl.col("eco_lvl_1").first().alias("eco"),
            pl.col("Ig_Date").alias(ig_dt),
            pl.col(*extra_cols),
        )
        .filter(pl.col("n") >= 2)
        .select(pl.exclude("n"))
        .explode(ig_dt, *extra_cols)
        .sort("geohash", ig_dt)
        .group_by("geohash")
        .agg(
            pl.col("eco").first(),
            pl.col(ig_dt).diff().shift(-1).dt.total_days().alias("dt") / 365,
            *flatmap(_extra_exprs, extra_cols),
        )
        .explode("dt", *flatmap(_extra_names, extra_cols))
        .drop_nulls()
    )
