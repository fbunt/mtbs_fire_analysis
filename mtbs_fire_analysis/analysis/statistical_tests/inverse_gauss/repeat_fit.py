"""Repeated-fit smoke test for Inverse Gaussian (IG) akin to Weibull repeat.

This mirrors the structure of the Weibull repeat simulation but uses the IG
distribution integrated in scipy_dist. It samples renewal data with events
(dt), right-censor (ct), forward gaps (ut), and empty windows, then fits
several variants and summarizes recovery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mtbs_fire_analysis.analysis.statistical_tests.test_helpers as th
from mtbs_fire_analysis.analysis.scipy_dist import InverseGauss


def run_inverse_gauss_simulation(
    num_pixels: int,
    time_interval: int,
    iterations: int,
    truth: InverseGauss,
    properties: tuple[str, ...] = ("mean",),
    prop_args: dict | None = None,
    pre_window: int = 1000,
    random_seed: int | None = None,
):
    """Run repeated IG fitting simulation and return a summary DataFrame.

    Returns
    -------
    output_df : pd.DataFrame
        Summary statistics for each fit type.
    fit_outs : dict
        Lists of fit results for each fit type.
    """

    if prop_args is None:
        prop_args = {prop: [] for prop in properties}

    def get_values(
        fit: InverseGauss,
        properties: tuple[str, ...] = ("mean",),
        prop_args: dict | None = None,
    ):
        if prop_args is None:
            prop_args = {prop: [] for prop in properties}
        # Ensure pure Python floats to avoid object dtypes downstream
        params = {k: float(v) for k, v in fit.params.items()}
        for prop in properties:
            params[prop] = float(getattr(fit, prop)(*prop_args.get(prop, [])))
        return params

    rng = np.random.default_rng(random_seed)

    fit_names = ["truth", "dt_only", "dtct", "dtctut", "dtctutet", "naive"]
    fit_outs: dict[str, list[dict]] = {name: [] for name in fit_names}
    fail_counts = {name: 0 for name in fit_names if name != "truth"}

    stat_names = list(truth.params.keys()) + list(properties)

    for _ in range(iterations):
        (
            dts,
            dt_counts,
            uts,
            ut_counts,
            cts,
            ct_counts,
            ets,
            et_counts,
        ) = th.create_sample_data_all(
            num_pixels=num_pixels,
            time_interval=time_interval,
            truth=truth,
            start_up_time=pre_window,
            round_to_nearest=0.1,
            rng=rng,
        )

        # naive rate estimate (events per exposure)
        fit_outs["naive"].append(
            dict.fromkeys(truth.params, 0)
            | {
                "mean": num_pixels
                * time_interval
                / (dt_counts.sum() + ct_counts.sum()),
                **{prop: 0 for prop in properties if prop != "mean"},
            }
        )

        # dt_only
        try:
            dt_only_fitter = InverseGauss(**truth.params)
            dt_only_fitter.fit(dts, dt_counts)
            fit_outs["dt_only"].append(
                get_values(dt_only_fitter, properties, prop_args)
            )
            if not dt_only_fitter.quick_health()["ok"]:
                fail_counts["dt_only"] += 1
        except RuntimeError:
            fail_counts["dt_only"] += 1

        # dtct
        try:
            dtct_fitter = InverseGauss(**truth.params)
            dtct_fitter.fit(dts, dt_counts, cts, ct_counts)
            fit_outs["dtct"].append(
                get_values(dtct_fitter, properties, prop_args)
            )
            if not dtct_fitter.quick_health()["ok"]:
                fail_counts["dtct"] += 1
        except RuntimeError:
            fail_counts["dtct"] += 1

        # dtctut
        try:
            dtctut_fitter = InverseGauss(**truth.params)
            dtctut_fitter.fit(dts, dt_counts, cts, ct_counts, uts, ut_counts)
            fit_outs["dtctut"].append(
                get_values(dtctut_fitter, properties, prop_args)
            )
            if not dtctut_fitter.quick_health()["ok"]:
                fail_counts["dtctut"] += 1
        except RuntimeError:
            fail_counts["dtctut"] += 1

        # dtctutet
        try:
            dtctutet_fitter = InverseGauss(**truth.params)
            dtctutet_fitter.fit(
                dts, dt_counts, cts, ct_counts, uts, ut_counts, ets, et_counts
            )
            fit_outs["dtctutet"].append(
                get_values(dtctutet_fitter, properties, prop_args)
            )
            if not dtctutet_fitter.quick_health()["ok"]:
                fail_counts["dtctutet"] += 1
        except RuntimeError:
            fail_counts["dtctutet"] += 1

    # Add truth row
    fit_outs["truth"] = [truth.params]
    fit_outs["truth"][0].update(
        {
            prop: float(getattr(truth, prop)(*prop_args.get(prop, [])))
            for prop in properties
        }
    )

    def create_statistics(df: pd.DataFrame) -> dict:
        # Convert all columns to numeric explicitly—object dtypes can cause
        # quantile/mean to return NaN silently.
        df_num = df.apply(pd.to_numeric, errors="coerce")
        # Drop completely empty columns (all NaN) early.
        df_num = df_num.loc[:, ~(df_num.isna().all())]

        def _col_stats(s: pd.Series) -> dict:
            if s.dropna().empty:
                return {"10pc": np.nan, "mean": np.nan, "90pc": np.nan}
            return {
                "10pc": s.quantile(0.10),
                "mean": s.mean(),
                "90pc": s.quantile(0.90),
            }

        summary = df_num.apply(_col_stats)
        # summary is a Series of dicts; expand to DataFrame
        summary_df = (
            summary.apply(pd.Series)
            .reset_index()
            .melt(id_vars="index", var_name="pc", value_name="value")
        )
        # index column holds original column names (stat); pc holds quantile
        # tag
        return {
            f"{row['index']}_{row['pc']}": row["value"]
            for _, row in summary_df.iterrows()
        }

    all_stats = [
        {
            "name": name,
            "fail_prob": fail_counts.get(name, 0) / iterations,
            **create_statistics(pd.DataFrame(fit_outs[name])),
        }
        for name in fit_names
    ]

    columns = [
        "name",
        *[
            f"{stat}_{pc}"
            for stat in stat_names
            for pc in ["10pc", "mean", "90pc"]
        ],
        "fail_prob",
    ]

    output_df = pd.DataFrame(all_stats, columns=columns)
    return output_df, fit_outs


if __name__ == "__main__":  # pragma: no cover
    out, fits = run_inverse_gauss_simulation(
        num_pixels=5000,
        time_interval=39,
        iterations=100,
        truth=InverseGauss(mu=75.0, lam=1.0),
        properties=("mean",),
        prop_args={},
        pre_window=1000,
        random_seed=1989,
    )
    print(out)
