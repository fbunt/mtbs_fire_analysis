from pathlib import Path

import distributions as cd
import numpy as np
from distributions import HalfLifeHazardDistribution as HLHD

num_pixels = 10000
time_interval = 40

# Count total time taken
remaining_times = time_interval * np.ones(num_pixels)
last_times = np.zeros(num_pixels)
expand_samples = np.zeros(num_pixels)

dts = []

halflife=3
hazard_inf = 0.2

truth = HLHD(hazard_inf=hazard_inf, half_life=halflife)

remaining = remaining_times > 0


while any(remaining):
    # Generate samples for remaining pixels

    expand_samples[remaining] = truth.rvs(size=sum(remaining)
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

fitter = HLHD(hazard_inf=hazard_inf, half_life=halflife)
fitter.fit(dts)
fitter_censor = HLHD(hazard_inf=hazard_inf, half_life=halflife)
fitter_censor.fit(dts, last_times)

# Print the original params, first params, then censor params
print("Original params: ")
print(truth.params)
print("Fitted parameters without censoring: ")
print(fitter.params)
print("Fitted parameters with censoring: ")
print(fitter_censor.params)

print("Neg Log likelihood values ignoring censoring: ")
print("Original params: ")
print(truth.neg_log_likelihood(dts))
print("Fitted parameters without censoring: ")
print(fitter.neg_log_likelihood(dts))
print("Fitted parameters with censoring: ")
print(fitter_censor.neg_log_likelihood(dts, last_times))

print("Neg log likelihood values with censoring: ")
print("Original params: ")
print(truth.neg_log_likelihood(dts, last_times))
print("Fitted parameters without censoring: ")
print(fitter.neg_log_likelihood(dts, last_times))
print("Fitted parameters with censoring: ")
print(fitter_censor.neg_log_likelihood(dts, last_times))


out_dir = Path("analysis/Outputs")

cd.plot_fit(dts, fitter, out_dir / "HLHDFitNoCensor.png")
cd.plot_fit(dts, fitter_censor, out_dir / "HLHDFitCensor.png")
cd.plot_fit(dts, truth, out_dir / "HLHDFitActual.png")
