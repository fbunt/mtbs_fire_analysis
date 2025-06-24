import polars as pl

from mtbs_fire_analysis.utils import flatmap


def _extra_names(col):
    return f"{col}_1", f"{col}_2"


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
            pl.col("eco_lvl_1").first().alias("eco1"),
            pl.col("eco_lvl_2").first().alias("eco2"),
            pl.col("eco_lvl_3").first().alias("eco3"),
            pl.col("Ig_Date").alias(ig_dt),
            pl.col(*extra_cols),
        )
        .filter(pl.col("n") >= 2)
        .select(pl.exclude("n"))
        .explode(ig_dt, *extra_cols)
        .sort("geohash", ig_dt)
        .group_by("geohash")
        .agg(
            pl.col("eco1").first(),
            pl.col("eco2").first(),
            pl.col("eco3").first(),
            pl.col(ig_dt).diff().shift(-1).dt.total_days().alias("dt") / 365,

            *flatmap(_extra_exprs, extra_cols),
        )
        .explode("dt", *flatmap(_extra_names, extra_cols))
        .drop_nulls()
    )

def build_survival_times(lf, max_date=None, extra_cols=None):
    """Build a dataframe of survival times. This is the time from the last fire
    per pixel to the most recent fire in the entire dataset. This is used to fit
    the hazard function to the data.

    Parameters
    ----------
    lf : polars.LazyFrame, polars.DataFrame
        The dataframe to process.
    extra_cols : list, optional
        The extra columns to get the values for the last fire per pixel.

    Returns
    -------
    polars.LazyFrame, polars.DataFrame

    """
    if max_date is None:
        max_date = lf.select(pl.col("Ig_Date").max()).collect().item()

    extra_cols = extra_cols or []
    assert "bs" not in extra_cols
    extra_cols = ["bs"] + extra_cols
    return (
        lf.group_by("geohash")
        .agg(
            pl.len().alias("n"),
            pl.col("eco_lvl_1").last(),
            pl.col("eco_lvl_2").last(),
            pl.col("eco_lvl_3").last(),
            (max_date - pl.col("Ig_Date").last()).dt.total_days().alias("st") /365 ,
            pl.col(*extra_cols).last(),
        )
        .drop_nulls()
    )


def build_event_histories(lf, max_date=None, fixed_pivots=None,varied_pivots=None):
    if max_date is None:
        max_date = lf.select(pl.col("Ig_Date").max()).collect().item()
    fixed_pivots = fixed_pivots or []
    varied_pivots = varied_pivots or []
    return (
        lf.filter(pl.col("Ig_Date") <= max_date)
        .group_by(["geohash"] + fixed_pivots)
        .agg(pl.col("Ig_Date").sort_by("Ig_Date"),
            pl.col("Event_ID").sort_by("Ig_Date"),
            pl.col(*varied_pivots).sort_by("Ig_Date"),
            pl.len().alias("# Fires")
        )
        .group_by(["Ig_Date", "# Fires","Event_ID"] + fixed_pivots + varied_pivots)
        .agg(
            pl.len().alias("Pixel_Count")
        )
        .drop_nulls()
    )

def event_hist_to_dts(event_hist, varied_filters=None):
    """Convert a fire list event histogram to a dataframe of dt's."""
    varied_pivots = [col for col, vals in (varied_filters or {}).items()]
    return (
        event_hist.filter(pl.col("# Fires") > 1)
        .select(pl.exclude("# Fires"))
        .with_columns(
            pl.col("Ig_Date").list.eval(
                pl.element().diff().dt.total_days()
            )
            .alias("dt") / 365
        )
        .explode("dt", *varied_pivots)
        .filter(
            pl.all_horizontal([pl.col(col).is_in(vals) for col, vals in (varied_filters or {}).items()])
        )
        .drop_nulls()
        .select(
            pl.col('dt'),
            pl.col('Pixel_Count'),
            *varied_pivots
        )
    )

def event_hist_to_sts(event_hist, min_date = None, max_date=None, fixed_pivots=None, varied_pivots=None):
    """Convert a fire list event histogram to a dataframe of survival times."""
    if min_date is None:
        min_date = event_hist.select(pl.col("Ig_Date").list.min()).min().item()
    if max_date is None:
        max_date = event_hist.select(pl.col("Ig_Date").list.max()).max().item()
    fixed_pivots = fixed_pivots or []
    varied_pivots = varied_pivots or []
    return (
        event_hist
        .with_columns(
            (max_date - pl.col("Ig_Date").list.last()).alias("ct"),
            (pl.col("Ig_Date").list.first() - min_date).alias("ut")
        )
        .select(
            pl.col('ct'),
            pl.col('ut'),
            pl.col('Pixel_Count'),
            *fixed_pivots,
            *varied_pivots
        )
    )

def event_hist_to_uts(event_hist, min_date = None, varied_filters=None):
    """Convert a fire list event histogram to a dataframe of survival times."""
    if min_date is None:
        min_date = event_hist.select(pl.col("Ig_Date").list.min()).min().item()
    varied_pivots = [col for col, vals in (varied_filters or {}).items()]
    return (
        event_hist
        .with_columns(
            (pl.col("Ig_Date").list.first() - min_date).dt.total_days().alias("ut") / 365
        ).filter(
            pl.all_horizontal([pl.col(col).list.first().is_in(vals) for col, vals in varied_filters.items()])
        )
        .select(
            pl.col('ut'),
            pl.col('Pixel_Count'),
            *varied_pivots
        )
    )

def event_hist_to_cts(event_hist, max_date = None, varied_filters=None):
    """Convert a fire list event histogram to a dataframe of survival times."""
    if max_date is None:
        max_date = event_hist.select(pl.col("Ig_Date").list.max()).max().item()
    varied_pivots = [col for col, vals in (varied_filters or {}).items()]
    return (
        event_hist
        .with_columns(
            (max_date - pl.col("Ig_Date").list.last()).dt.total_days().alias("ct") / 365
        ).filter(
            pl.all_horizontal([pl.col(col).list.last().is_in(vals) for col, vals in varied_filters.items()])
        )
        .select(
            pl.col('ct'),
            pl.col('Pixel_Count'),
            *varied_pivots
        )
    )

def event_hist_to_blank_pixels_events(event_hist, total_pixels):
    """Convert a fire list event histogram to a dataframe of no events."""
    return (
        total_pixels - event_hist.select(
            pl.col('Pixel_Count')
        ).sum().item()
    )

