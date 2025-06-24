from pathlib import Path

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
    remaining_times = (left_censoring_time + time_interval) * np.ones(num_pixels)
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

        new_dts = expand_samples[~finished & remaining & entered_window].tolist()
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
    t_open  = start_up_time
    t_close = start_up_time + time_interval

    # Pixel-wise state
    t_prev  = np.zeros(num_pixels)
    alive   = np.ones(num_pixels, bool)
    seen    = np.zeros(num_pixels, bool)
    last_ev = np.full(num_pixels, np.nan)

    # Event buffers
    dts, uts, cts = [], [], []

    while alive.any():
        idx = np.nonzero(alive)[0]
        dt  = truth.rvs(size=len(idx), rng=rng)
        t_new = t_prev[idx] + dt

        before = t_new < t_open
        inside = (t_new >= t_open) & (t_new <= t_close)
        after  = t_new > t_close

        # events before window
        if before.any():
            t_prev[idx[before]] = t_new[before]

        # events inside window
        if inside.any():
            iwin = idx[inside]
            tn   = t_new[inside]

            first = ~seen[iwin]
            if first.any():                      # forward recurrence (ut)
                uts.extend(tn[first] - t_open)
                seen[iwin[first]] = True

            repeat = ~first                     # fully observed (dt)
            if repeat.any():
                dts.extend(tn[repeat] - last_ev[iwin[repeat]])

            last_ev[iwin] = tn
            t_prev[iwin]  = tn

        # events after window
        if after.any():
            iaft = idx[after]
            has_ev = seen[iaft]
            if has_ev.any():                    # right-censor (ct)
                cts.extend(t_close - last_ev[iaft[has_ev]])
            alive[iaft] = False                 # simulation done for pixel

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
        dts, dt_count,
        uts, ut_count,
        cts, ct_count,
        n0t, n0_count,
    )


# %%
def create_sample_data_all_old(
    num_pixels: int,
    time_interval: float,
    truth,
    start_up_time: float = 0.0,
    round_to_nearest: float | None = None,
):
    """
    Simulate fire-history data and return the four disjoint observation types
    needed for the likelihood:

    • dt   – fully-observed inter-fire intervals **inside** the window  
    • ut   – forward-recurrence intervals from the window start to first fire  
    • ct   – right-censor times from last fire to the window end  
    • n0   – number of pixels with *no* fire in the window

    Parameters
    ----------
    num_pixels      : int
        Number of independent pixels.
    time_interval   : float
        Length of the observation window (years, months, …).
    truth           : distribution object with an ``rvs(size=…)`` method.
    start_up_time   : float, default 0
        “Burn-in” time simulated *before* the window opens.
    round_to_nearest: float or None, default None
        If given, round every dt/ut/ct to the nearest multiple of this value
        and return *counts* for the unique rounded times.  When ``None`` each
        observation is kept in full precision and its count is 1.

    Returns
    -------
    dts, dt_count,
    uts, ut_count,
    cts, ct_count : 1-D ``np.ndarray`` pairs
        Either the full lists (if ``round_to_nearest is None``) or the unique
        rounded values with their multiplicities.
    n0            : int
        Number of pixels with zero events inside the window.
    """
    # ------------------------------------------------------------------ helpers
    def _aggregate(values, step):
        """Round to nearest *step* and collapse duplicates."""
        if not len(values):
            return np.empty(0, float), np.empty(0, int)

        if step is None:
            return np.asarray(values, float), np.ones(len(values), int)

        rounded = np.round(np.asarray(values) / step) * step
        uniq, counts = np.unique(rounded, return_counts=True)
        return uniq.astype(float), counts.astype(int)

    # ------------------------------------------------------------------ simulate
    dts, uts, cts = [], [], []
    n0 = 0  # pixels with no events in the window

    total_time = start_up_time + time_interval

    rng = np.random.default_rng()  # single generator for reproducibility

    for _ in range(num_pixels):
        t = 0.0
        fires = []

        # draw successive inter-fire intervals until we run past total_time
        while True:
            dt = truth.rvs(size=1, rng=rng)[0]
            if t + dt > total_time:
                break
            t += dt
            fires.append(t)

        # ---------------------------------------------------------------- events
        # Any fire ≥ start_up_time is within the observation window
        in_window = [f for f in fires if f >= start_up_time]

        if not in_window:
            # zero events ⇒ whole window is right-censored but we treat these
            # pixels as a count, not individual ct observations.
            n0 += 1
            continue

        # forward-recurrence (window start → first fire)
        uts.append(in_window[0] - start_up_time)

        # fully-observed intervals *between* fires inside the window
        for i in range(1, len(in_window)):
            dts.append(in_window[i] - in_window[i - 1])

        # right-censor (last fire → window end)
        cts.append(total_time - in_window[-1])

    # ------------------------------------------------------------------ output
    dts, dt_count = _aggregate(dts, round_to_nearest)
    uts, ut_count = _aggregate(uts, round_to_nearest)
    cts, ct_count = _aggregate(cts, round_to_nearest)

    return dts, dt_count, uts, ut_count, cts, ct_count, np.array([time_interval]), np.array([n0])


def evaluate_fits(
    dts,
    last_times,
    fits,
    names = None
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
    names = None,
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
        print(fit.neg_log_likelihood(dts, dt_counts, cts, ct_counts, uts, ut_counts, num_empty))


def output_plots(dts,last_times, fits, folder, names=None):
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
        #cd.plot_fit(fit, dts, last_times, out_dir / (prefix+f"{name}Fit.png"))
        cd.plot_fit(
            fit,
            dts,
            survivals=last_times,
            output_name=folder / (f"{name}.png"),
            max_dt=75,
        )
