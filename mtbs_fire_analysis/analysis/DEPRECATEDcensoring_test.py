### NEEDS TO BE UPDATED TO USE NEW TEST HELPERS

from pathlib import Path

import numpy as np
from scipy.stats import weibull_min

import mtbs_fire_analysis.analysis.distributions as cd

num_pixels = 10000
time_interval = 4

# Count total time taken
remaining_times = time_interval * np.ones(num_pixels)
last_times = np.zeros(num_pixels)
expand_samples = np.zeros(num_pixels)

dts = []

# Generate Weibull samples
shape = 1.2
scale = 1.2

remaining = remaining_times > 0


while any(remaining):
    # Generate Weibull samples for remaining pixels

    expand_samples[remaining] = weibull_min.rvs(
        shape, scale=scale, size=sum(remaining)
    )
    # Check if any samples exceed the remaining time
    finished = expand_samples > remaining_times
    # Store the last time for finished pixels
    last_times[finished & remaining] = remaining_times[finished & remaining]
    # Update remaining times
    remaining_times = np.maximum(remaining_times - expand_samples, 0)
    # Store the new dts for unfinished pixels
    new_dts = expand_samples[~finished & remaining].tolist()
    dts = dts + new_dts
    # Find remaining pixels
    remaining = remaining_times > 0.0

# Counts of dts and last_times
num_dts = len(dts)
num_last_times = len(last_times)
print(f"Number of dts: {num_dts}")
print(f"Number of last_times: {num_last_times}")


# Check total time taken
total_time = np.sum(dts) + np.sum(last_times)
print(f"Total time taken: {total_time}")
# Original time
print(f"Original time: {time_interval * num_pixels}")

# Produce fit ignoring sensoring

fitter = cd.WeibullDistribution(1.0, 1.0)
fitter.fit(dts)
# Fit Weibull distribution to the samples
print(
    f"Fitted Weibull parameters: shape = {fitter.shape}, scale = {fitter.scale}"
)

print(
    "Neg Log likelihood of the fitted Weibull "
    f"distribution: {fitter.neg_log_likelihood(dts)}"
)
print(
    "Neg Log likelihood of the fitted Weibull distribution"
    f" with censoring: {fitter.neg_log_likelihood(dts, last_times)}"
)
# Now fit with censoring


fitter_censor = cd.WeibullDistribution(1.0, 1.0)
fitter_censor.fit(dts, last_times)
# Fit Weibull distribution to the samples
print(
    "Fitted Weibull parameters with censoring: "
    f"shape = {fitter_censor.shape}, scale = {fitter_censor.scale}"
)

print(
    "Neg Log likelihood of the fitted Weibull "
    f"distribution: {fitter_censor.neg_log_likelihood(dts)}"
)
print(
    "Neg Log likelihood of the fitted Weibull distribution "
    f"with censoring: {fitter_censor.neg_log_likelihood(dts, last_times)}"
)

# Create fit object with actual params

actual = cd.WeibullDistribution(shape, scale)

print(
    "Neg Log likelihood of the fitted Weibull "
    f"distribution: {actual.neg_log_likelihood(dts)}"
)
print(
    "Neg Log likelihood of the fitted Weibull distribution "
    f"with censoring: {actual.neg_log_likelihood(dts, last_times)}"
)

out_dir = Path("analysis/Outputs")

cd.plot_fit(dts, fitter, out_dir / "WeibullFitNoCensor.png")
cd.plot_fit(dts, fitter_censor, out_dir / "WeibullFitCensor.png")
cd.plot_fit(dts, actual, out_dir / "WeibullFitActual.png")
