import numpy as np
import pandas as pd

import mtbs_fire_analysis.analysis.statistical_tests.test_helpers as th
from mtbs_fire_analysis.analysis.hlh_dist import (
    HalfLifeHazardDistribution as HLHD,
)


def run_hlh_simulation(
    num_pixels: int,
    time_interval: int,
    iterations: int,
    truth,
    properties: tuple[str] = ("mean",),
    prop_args: dict | None = None,
    pre_window: int = 1000,
    random_seed: int | None = None,
):
    """
    Run Repeated HLHD fitting simulation and return summary DataFrame.

    Returns
    -------
    output_df : pd.DataFrame
        Summary statistics for each fit type.
    fit_outs : dict
        Dictionary of lists of fit results for each fit type.
    """

    if prop_args is None:
        prop_args = {prop: [] for prop in properties}

    def get_values(
        fit, properties: tuple[str] = ("mean",), prop_args: dict | None = None
    ):
        if prop_args is None:
            prop_args = {prop: [] for prop in properties}
        params = fit.params  # return dictionary of parameters
        for prop in properties:
            params[prop] = getattr(fit, prop)(
                *prop_args.get(prop, [])
            )  # add properties to params dict
        return params

    rng = np.random.default_rng(random_seed)

    fit_names = ["truth", "dt_only", "dtct", "dtctut", "dtctutet", "naive"]
    fit_outs = {name: [] for name in fit_names}
    fail_counts = {name: 0 for name in fit_names if name != "truth"}

    stat_names = list(truth.params.keys()) + list(properties)

    for i in range(iterations):
        dts, dt_counts, cts, ct_counts, uts, ut_counts, ets, et_counts = (
            th.create_sample_data_all(
                num_pixels=num_pixels,
                time_interval=time_interval,
                truth=truth,
                start_up_time=pre_window,
                round_to_nearest=0.1,
                rng=rng,
            )
        )
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
            dt_only_fitter = HLHD()
            dt_only_fitter.fit(dts, dt_counts, print_errors=False)
            fit_outs["dt_only"].append(
                get_values(dt_only_fitter, properties, prop_args)
            )
            if not dt_only_fitter.quick_health()["ok"]:
                fail_counts["dt_only"] += 1
        except RuntimeError:
            fail_counts["dt_only"] += 1

        # dtct
        try:
            dtct_fitter = HLHD()
            dtct_fitter.fit(dts, dt_counts, cts, ct_counts, print_errors=False)
            fit_outs["dtct"].append(
                get_values(dtct_fitter, properties, prop_args)
            )
            if not dtct_fitter.quick_health()["ok"]:
                fail_counts["dtct"] += 1
        except RuntimeError:
            fail_counts["dtct"] += 1

        # dtctut
        try:
            dtctut_fitter = HLHD()
            dtctut_fitter.fit(
                dts,
                dt_counts,
                cts,
                ct_counts,
                uts,
                ut_counts,
                print_errors=False,
            )
            fit_outs["dtctut"].append(
                get_values(dtctut_fitter, properties, prop_args)
            )
            if not dtctut_fitter.quick_health()["ok"]:
                fail_counts["dtctut"] += 1
        except RuntimeError:
            fail_counts["dtctut"] += 1

        # dtctutet
        try:
            dtctutet_fitter = HLHD()
            dtctutet_fitter.fit(
                dts,
                dt_counts,
                cts,
                ct_counts,
                uts,
                ut_counts,
                ets,
                et_counts,
                print_errors=False,
            )
            fit_outs["dtctutet"].append(
                get_values(dtctutet_fitter, properties, prop_args)
            )
            if not dtctutet_fitter.quick_health()["ok"]:
                fail_counts["dtctutet"] += 1
        except RuntimeError:
            fail_counts["dtctutet"] += 1

    # Add truth
    fit_outs["truth"] = [truth.params]
    fit_outs["truth"][0].update(
        {
            prop: getattr(truth, prop)(*prop_args.get(prop, []))
            for prop in properties
        }
    )

    def create_statistics(df):
        out_stats = (
            df.apply(
                lambda s: pd.Series(
                    {
                        "10pc": s.quantile(0.10),
                        "mean": s.mean(),
                        "90pc": s.quantile(0.90),
                    },
                )
            )
            .stack()  # long form (MultiIndex)
            .rename_axis(["stat", "prop"])  # (col, 10pc/mean/90pc)
            .swaplevel()
        ).to_dict()
        return {f"{k[0]}_{k[1]}": v for k, v in out_stats.items()}

    # Compute summary stats
    all_stats = [
        {
            "name": name,
            "fail_prob": fail_counts.get(name, 0) / iterations,
            **create_statistics(pd.DataFrame(fit_outs[name])),
        }
        for name in fit_names
    ]

    # Create DataFrame with statistics for each fit type

    output_df = pd.DataFrame(
        all_stats,
        columns=["name"]
        + [
            f"{stat}_{pc}"
            for stat in stat_names
            for pc in ["10pc", "mean", "90pc"]
        ]
        + ["fail_prob"],
    )

    return output_df, fit_outs


if __name__ == "__main__":
    o, fo = run_hlh_simulation(
        num_pixels=5000,
        time_interval=39,
        iterations=100,
        truth=HLHD(0.03, 50),
        properties=("mean",),#), "expected_hazard_ge"),
        prop_args={},#{"expected_hazard_ge": [39]},
        pre_window=1000,
        random_seed=1989,
    )

    print(o)
