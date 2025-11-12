from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import mtbs_fire_analysis.analysis.statistical_tests.test_helpers as th
from mtbs_fire_analysis.analysis.lifetime_base import BaseLifetime


def _random_init_around_truth(
    truth_params: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Random init where each param is scaled by U[0.25, 4]."""
    init = {}
    for k, v in truth_params.items():
        scale = rng.uniform(0.25, 4.0)
        init[k] = float(max(scale * float(v), np.finfo(float).tiny))
    return init


def _get_values(
    fit: BaseLifetime,
    properties: Tuple[str, ...],
    prop_args: Dict[str, Iterable],
) -> Dict[str, float]:
    """Copy params, compute requested properties, return as plain floats."""
    params = {k: float(v) for k, v in fit.params.items()}
    for prop in properties:
        args = list(prop_args.get(prop, [])) if prop_args is not None else []
        params[prop] = float(getattr(fit, prop)(*args))
    return params


def _create_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """Robust summary stats (10pc/mean/90pc) with numeric coercion.

    Returns a flat dict mapping "col_pc" to value.
    """
    if df.empty:
        return {}
    df_num = df.apply(pd.to_numeric, errors="coerce")
    # Drop columns that are entirely NaN
    df_num = df_num.loc[:, ~(df_num.isna().all())]

    def _col_stats(s: pd.Series) -> Dict[str, float]:
        s = s.dropna()
        if s.empty:
            return {"10pc": np.nan, "mean": np.nan, "90pc": np.nan}
        return {
            "10pc": s.quantile(0.10),
            "mean": s.mean(),
            "90pc": s.quantile(0.90),
        }

    summary = df_num.apply(_col_stats)
    summary_df = summary.apply(pd.Series).reset_index().melt(
        id_vars="index", var_name="pc", value_name="value"
    )
    return {
        f"{row['index']}_{row['pc']}": row["value"]
        for _, row in summary_df.iterrows()
    }


def run_repeat_simulation(
    *,
    model_ctor: Callable[..., BaseLifetime],
    truth: BaseLifetime,
    num_pixels: int,
    time_interval: int,
    iterations: int,
    properties: Tuple[str, ...] = ("mean",),
    prop_args: Dict[str, Iterable] | None = None,
    pre_window: int = 1000,
    random_seed: int | None = None,
    modes: Tuple[str, ...] | None = None,
):
    """Generic repeated-fit simulation for renewal models with
    censoring and gaps.

    Parameters
    ----------
    model_ctor : Callable[..., BaseLifetime]
    Factory that accepts the same kwargs as ``truth.params``
    to construct a model.
    truth : BaseLifetime
        The truth model (used for data generation and for reference row).
    properties : tuple of str
        Model methods to evaluate per fit (e.g., ("mean",)).
    prop_args : dict
        Mapping prop -> argument list for evaluation.
    """
    if prop_args is None:
        prop_args = {prop: [] for prop in properties}

    rng = np.random.default_rng(random_seed)

    all_modes = ("truth", "dt_only", "dtct", "dtctut", "dtctutet", "naive")
    fit_names = list(all_modes if modes is None else modes)
    fit_outs: Dict[str, list[dict]] = {name: [] for name in fit_names}
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
        if "naive" in fit_names:
            fit_outs["naive"].append(
                dict.fromkeys(truth.params, 0)
                | {
                    "mean": (
                        num_pixels
                        * time_interval
                        / (dt_counts.sum() + ct_counts.sum())
                    ),
                    **{prop: 0 for prop in properties if prop != "mean"},
                }
            )

        # Helper to build and fit with random initialisation around truth
        def _fit_with(data_kwargs: Dict[str, object]) -> BaseLifetime:
            init = _random_init_around_truth(truth.params, rng)
            model = model_ctor(**init)
            # Prefer named arguments to be compatible across implementations
            try:
                # HLH accepts print_errors; SciPy-backed ones will TypeError
                model.fit(
                    print_errors=False,
                    **data_kwargs,
                )
            except TypeError:
                # Some implementations (HLH) support extra kwargs like
                # print_errors; retry clean
                model.fit(**dict(data_kwargs))
            return model

        # dt_only
        if "dt_only" in fit_names:
            try:
                dt_only = _fit_with({
                    "data": dts,
                    "data_counts": dt_counts,
                })
                fit_outs["dt_only"].append(
                    _get_values(dt_only, properties, prop_args)
                )
                if not dt_only.quick_health()["ok"]:
                    fail_counts["dt_only"] += 1
            except RuntimeError:
                fail_counts["dt_only"] += 1

        # dtct
        if "dtct" in fit_names:
            try:
                dtct = _fit_with({
                    "data": dts,
                    "data_counts": dt_counts,
                    "survival_data": cts,
                    "survival_counts": ct_counts,
                })
                fit_outs["dtct"].append(
                    _get_values(dtct, properties, prop_args)
                )
                if not dtct.quick_health()["ok"]:
                    fail_counts["dtct"] += 1
            except RuntimeError:
                fail_counts["dtct"] += 1

        # dtctut
        if "dtctut" in fit_names:
            try:
                dtctut = _fit_with({
                    "data": dts,
                    "data_counts": dt_counts,
                    "survival_data": cts,
                    "survival_counts": ct_counts,
                    "initial_gaps": uts,
                    "initial_counts": ut_counts,
                })
                fit_outs["dtctut"].append(
                    _get_values(dtctut, properties, prop_args)
                )
                if not dtctut.quick_health()["ok"]:
                    fail_counts["dtctut"] += 1
            except RuntimeError:
                fail_counts["dtctut"] += 1

        # dtctutet
        if "dtctutet" in fit_names:
            try:
                dtctutet = _fit_with({
                    "data": dts,
                    "data_counts": dt_counts,
                    "survival_data": cts,
                    "survival_counts": ct_counts,
                    "initial_gaps": uts,
                    "initial_counts": ut_counts,
                    "empty_windows": ets,
                    "empty_counts": et_counts,
                })
                fit_outs["dtctutet"].append(
                    _get_values(dtctutet, properties, prop_args)
                )
                if not dtctutet.quick_health()["ok"]:
                    fail_counts["dtctutet"] += 1
            except RuntimeError:
                fail_counts["dtctutet"] += 1

    # Add truth row
    if "truth" in fit_names:
        fit_outs["truth"] = [truth.params]
        fit_outs["truth"][0].update(
            {
                prop: float(getattr(truth, prop)(*prop_args.get(prop, [])))
                for prop in properties
            }
        )

    # Compile stats rows
    all_stats = [
        {
            "name": name,
            "fail_prob": fail_counts.get(name, 0) / iterations,
            **_create_statistics(pd.DataFrame(fit_outs[name])),
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
