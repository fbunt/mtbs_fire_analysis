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
        print(fit.neg_log_likelihood(dts, last_times))


def output_plots(dts,last_times, fits, prefix=None, names=None):
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
    out_dir = Path("Outputs")
    if names is None:
        names = [f"Fit {i}" for i in range(len(fits))]
    prefix = prefix or ""
    for fit, name in zip(fits, names, strict=True):
        cd.plot_fit(fit, dts, last_times, out_dir / (prefix+f"{name}Fit.png"))