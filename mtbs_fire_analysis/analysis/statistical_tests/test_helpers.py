import numpy as np

import mtbs_fire_analysis.analysis.distributions as cd


def create_sample_data(
    num_pixels,
    time_interval,
    truth,
    left_censoring_time=0,
    discard_no_events=False,
):
    """
    Data simulation for inspection and testing dist objects

    Parameters
    ----------
    num_pixels : int
        Number of pixels to sample.
    time_interval : float
        Time interval for the samples.
    truth : dist object
        The distribution object to sample from.
    left_censoring : bool, optional
        If True, samples that exceed the time interval are discarded.
    discard_no_events : bool, optional
        If True, pixels that have no events are discarded.
        Alternatively this can be done outside of this function trivially.

    Returns
    -------
    dts : np.ndarray
        The sampled times between events.
    last_times : np.ndarray
        The last times for each pixel (right censoring).
    """

    # Count total time taken
    remaining_times = (left_censoring_time + time_interval) * np.ones(
        num_pixels
    )
    last_times = np.zeros(num_pixels)
    expand_samples = np.zeros(num_pixels)

    dts = []

    remaining = remaining_times > 0

    while any(remaining):
        # Generate samples for remaining pixels
        expand_samples[remaining] = truth.rvs(size=sum(remaining))
        # Check if any samples exceed the remaining time
        finished = expand_samples > remaining_times
        # Store the last time for finished pixels
        last_times[finished & remaining] = remaining_times[
            finished & remaining
        ]
        # Update remaining times
        remaining_times = np.maximum(remaining_times - expand_samples, 0)
        # Store the new dts for unfinished pixels

        entered_window = remaining_times - time_interval < left_censoring_time
        # If the sample is in the window, we can add it to the dts

        new_dts = expand_samples[
            ~finished & remaining & entered_window
        ].tolist()
        dts.extend(new_dts)
        # Find remaining pixels
        remaining = remaining_times > 0.0

    if discard_no_events:
        last_times = last_times[last_times < time_interval]

    return np.array(dts), last_times


def create_sample_data_all(
    num_pixels: int,
    time_interval: float,
    truth,
    *,
    start_up_time: float = 0.0,
    round_to_nearest: float | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Vectorised simulator for the renewal process described earlier.

    Returns
    -------
    dts, dt_count,
    uts, ut_count,
    cts, ct_count : np.ndarray pairs
        Unique values and multiplicities (or the raw lists when
        ``round_to_nearest is None`` — exactly as before).
    n0t, n0_count   – interval length(s) and multiplicities
        Number of pixels with no fires inside the observation window.
    """
    rng = np.random.default_rng() if rng is None else rng

    # Window bounds
    t_open = start_up_time
    t_close = start_up_time + time_interval

    # Pixel-wise state
    t_prev = np.zeros(num_pixels)
    alive = np.ones(num_pixels, bool)
    seen = np.zeros(num_pixels, bool)
    last_ev = np.full(num_pixels, np.nan)

    # Event buffers
    dts, uts, cts = [], [], []

    while alive.any():
        idx = np.nonzero(alive)[0]
        dt = truth.rvs(size=len(idx), rng=rng)
        t_new = t_prev[idx] + dt

        before = t_new < t_open
        inside = (t_new >= t_open) & (t_new <= t_close)
        after = t_new > t_close

        # events before window
        if before.any():
            t_prev[idx[before]] = t_new[before]

        # events inside window
        if inside.any():
            iwin = idx[inside]
            tn = t_new[inside]

            first = ~seen[iwin]
            if first.any():  # forward recurrence (ut)
                uts.extend(tn[first] - t_open)
                seen[iwin[first]] = True

            repeat = ~first  # fully observed (dt)
            if repeat.any():
                dts.extend(tn[repeat] - last_ev[iwin[repeat]])

            last_ev[iwin] = tn
            t_prev[iwin] = tn

        # events after window
        if after.any():
            iaft = idx[after]
            has_ev = seen[iaft]
            if has_ev.any():  # right-censor (ct)
                cts.extend(t_close - last_ev[iaft[has_ev]])
            alive[iaft] = False  # simulation done for pixel

    # -------------------------------------------------------------------------
    # Helper for rounding / collapsing duplicates
    def _collapse(arr, step):
        if not arr:
            return np.empty(0, float), np.empty(0, int)
        arr = np.asarray(arr, float)
        if step is not None:
            arr = np.round(arr / step) * step
        uniq, counts = np.unique(arr, return_counts=True)
        return uniq, counts.astype(int)

    dts, dt_count = _collapse(dts, round_to_nearest)
    uts, ut_count = _collapse(uts, round_to_nearest)
    cts, ct_count = _collapse(cts, round_to_nearest)

    # -------------------------------------------------------------------------
    # No-fire pixels: treat the entire window length as their right-censor time
    n0 = np.count_nonzero(~seen)
    n0t, n0_count = _collapse([time_interval] * n0, round_to_nearest)

    return (
        dts,
        dt_count,
        uts,
        ut_count,
        cts,
        ct_count,
        n0t,
        n0_count,
    )


def evaluate_fits(dts, last_times, fits, names=None):
    """
    Outputs fitted params, neg log likelihood in print statements.

    Parameters
    ----------
    dts : np.ndarray
        The sampled times between events.
    last_times : np.ndarray
        The last times for each pixel (right censoring).
    fits : list of dist objects
        The distribution objects that have been fit to data.
    names : list of str
        The names of the distributions/fits.
    """
    if names is None:
        names = [f"Fit {i}" for i in range(len(fits))]

    for fit, name in zip(fits, names, strict=True):
        print(f"Summary of {name}:")
        print(f"{name} params: ")
        print(fit.params)
        print(f"{name} neg log likelihood values with censoring: ")
        print(fit.neg_log_likelihood(dts, survival_data=last_times))


def evaluate_fits_all(
    fits,
    dts,
    dt_counts=None,
    cts=None,
    ct_counts=None,
    uts=None,
    ut_counts=None,
    num_empty=None,
    names=None,
):
    """
    Outputs fitted params, neg log likelihood in print statements.

    Parameters
    ----------
    dts : np.ndarray
        The sampled times between events.
    last_times : np.ndarray
        The last times for each pixel (right censoring).
    fits : list of dist objects
        The distribution objects that have been fit to data.
    names : list of str
        The names of the distributions/fits.
    """
    if names is None:
        names = [f"Fit {i}" for i in range(len(fits))]

    for fit, name in zip(fits, names, strict=True):
        print(f"Summary of {name}:")
        print(f"{name} params: ")
        print(fit.params)
        print(f"Return interval: {fit.mean()}")
        print(f"{name} neg log likelihood values with censoring: ")
        print(
            fit.neg_log_likelihood(
                dts, dt_counts, cts, ct_counts, uts, ut_counts, num_empty
            )
        )


def output_plots(dts, last_times, fits, folder, names=None):
    """
    Outputs plots of the fitted distributions.

    Parameters
    ----------
    dts : np.ndarray
        The sampled times between events.
    last_times : np.ndarray
        The last times for each pixel (right censoring).
    fits : list of dist objects
        The distribution objects that have been fit to data.
    names : list of str
        The names of the distributions/fits.
    """

    if names is None:
        names = [f"Fit {i}" for i in range(len(fits))]

    for fit, name in zip(fits, names, strict=True):
        cd.plot_fit(
            fit,
            dts,
            survivals=last_times,
            output_name=folder / (f"{name}.png"),
            max_dt=75,
        )
