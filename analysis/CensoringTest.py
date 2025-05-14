import Distributions as cd
import numpy as np
from scipy.stats import gamma, lognorm, norm, weibull_min
from pathlib import Path

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

remaining_pixels = remaining_times > 0


while any(remaining_pixels):

    # Generate Weibull samples
    expand_samples[remaining_pixels] = weibull_min.rvs(shape, scale=scale, size = sum(remaining_pixels)) # Generate Weibull samples for remaining pixels
    finished_mask = expand_samples > remaining_times # Check if any samples exceed the remaining time
    last_times[finished_mask&remaining_pixels] = remaining_times[finished_mask&remaining_pixels] # Store the last time for finished pixels
    remaining_times = np.maximum(remaining_times - expand_samples,0) # Update remaining times
    new_dts = expand_samples[~finished_mask&remaining_pixels].tolist() # Store the new dts for unfinished pixels
    dts=dts + new_dts
    remaining_pixels = remaining_times > 0.0 # Find remaining pixels

# Counts of dts and last_times
num_dts = len(dts)
num_last_times = len(last_times)
print(f"Number of dts: {num_dts}")
print(f"Number of last_times: {num_last_times}")


# Check total time taken
total_time = np.sum(dts)+ np.sum(last_times)
print(f"Total time taken: {total_time}")
# Original time
print(f"Original time: {time_interval*num_pixels}")

# Produce fit ignoring sensoring

fitter = cd.WeibullDistribution(1.0,1.0)
fitter.fit(dts)
# Fit Weibull distribution to the samples
print(f"Fitted Weibull parameters: shape = {fitter.shape}, scale = {fitter.scale}")

print(f"Neg Log likelihood of the fitted Weibull distribution: {fitter.neg_log_likelihood(dts)}")
print(f"Neg Log likelihood of the fitted Weibull distribution with censoring: {fitter.neg_log_likelihood(dts, last_times)}")
# Now fit with censoring


fitter_censor = cd.WeibullDistribution(1.0,1.0)
fitter_censor.fit(dts, last_times)
# Fit Weibull distribution to the samples
print(f"Fitted Weibull parameters with censoring: shape = {fitter_censor.shape}, scale = {fitter_censor.scale}")

print(f"Neg Log likelihood of the fitted Weibull distribution: {fitter_censor.neg_log_likelihood(dts)}")
print(f"Neg Log likelihood of the fitted Weibull distribution with censoring: {fitter_censor.neg_log_likelihood(dts, last_times)}")

# Create fit object with actual params

actual = cd.WeibullDistribution(shape, scale)

print(f"Neg Log likelihood of the fitted Weibull distribution: {actual.neg_log_likelihood(dts)}")
print(f"Neg Log likelihood of the fitted Weibull distribution with censoring: {actual.neg_log_likelihood(dts, last_times)}")

out_dir = Path('Outputs')

cd.plot_fit(dts, fitter, out_dir / "WeibullFitNoCensor.png")
cd.plot_fit(dts, fitter_censor, out_dir / "WeibullFitCensor.png")
cd.plot_fit(dts, actual, out_dir / "WeibullFitActual.png")