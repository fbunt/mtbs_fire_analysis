from __future__ import annotations

from mtbs_fire_analysis.analysis.hlh_dist import HalfLifeHazardDistribution
from mtbs_fire_analysis.analysis.statistical_tests.defaults import HLH_DEF, SIM
from mtbs_fire_analysis.analysis.statistical_tests.repeat_helpers import (
    run_repeat_simulation,
)


def run_hlh_simulation(
    num_pixels: int | None = None,
    time_interval: int | None = None,
    iterations: int | None = None,
    truth: HalfLifeHazardDistribution | None = None,
    properties: tuple[str, ...] = ("mean",),
    prop_args: dict | None = None,
    pre_window: int | None = None,
    random_seed: int | None = None,
):
    truth_model = HLH_DEF.build() if truth is None else truth
    return run_repeat_simulation(
        model_ctor=HalfLifeHazardDistribution,
        truth=truth_model,
        num_pixels=SIM.num_pixels if num_pixels is None else num_pixels,
        time_interval=(
            SIM.time_interval if time_interval is None else time_interval
        ),
        iterations=SIM.iterations if iterations is None else iterations,
        properties=properties,
        prop_args=prop_args,
        pre_window=SIM.pre_window if pre_window is None else pre_window,
        random_seed=SIM.random_seed if random_seed is None else random_seed,
    )


if __name__ == "__main__":  # pragma: no cover
    out, fits = run_hlh_simulation(
        properties=("mean",),
        prop_args={},
    )
    print(out)
