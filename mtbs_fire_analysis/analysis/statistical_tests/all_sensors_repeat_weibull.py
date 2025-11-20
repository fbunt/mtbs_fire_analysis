from mtbs_fire_analysis.analysis.statistical_tests.weibull.repeat_fit import (
    run_weibull_simulation,
)


if __name__ == "__main__":
    from mtbs_fire_analysis.analysis.scipy_dist import Weibull

    out, fits = run_weibull_simulation(
        num_pixels=5000,
        time_interval=39,
        iterations=100,
        truth=Weibull(shape=1.5, scale=75),
        properties=("mean",),
        prop_args={},
        pre_window=1000,
        random_seed=1989,
    )

    print(out)
