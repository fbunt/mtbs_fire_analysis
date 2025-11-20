from mtbs_fire_analysis.analysis.statistical_tests.hlh.repeat_fit import (
    run_hlh_simulation,
)


if __name__ == "__main__":
    from mtbs_fire_analysis.analysis.hlh_dist import (
        HalfLifeHazardDistribution,
    )

    o, fo = run_hlh_simulation(
        num_pixels=5000,
        time_interval=39,
        iterations=100,
        truth=HalfLifeHazardDistribution(0.03, 50),
        properties=("mean",),
        prop_args={},
        pre_window=1000,
        random_seed=1989,
    )

    print(o)
