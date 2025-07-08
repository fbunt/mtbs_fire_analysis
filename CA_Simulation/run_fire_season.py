"""
Supersedes run_single_fire3.py but re-uses its weather helpers.
"""
import numpy as np
from pathlib import Path

from CA_helpers import RainGenerator, MeanRevertingProcess, GaussianFieldProcess, Fuel
from fire_spread3 import (
    FireSpreadEngine,
    LegacyCollector as FrameCollector,
)

def rounded_exponential(mean: float, rng):
    x = rng.exponential(mean)
    k = int(np.floor(x))
    return k if rng.random() < 1 - (x - k) else k + 1

ignition_density = 1 /10_000


shape = (256, 256)
burn_spread_base = 0.3
burn_rate=0.1

av_size = (shape[0]+shape[1])/2
rng   = np.random.default_rng()

# Fuel: spatially correlated U(0,1)
fuel_div = 8
fuel = Fuel(shape, sigma=av_size/fuel_div, rng=rng)

# Rain generator (copied from your existing pattern, but smaller & faster)

thr_proc  = MeanRevertingProcess.from_half_life(
    mu=0.8, half_life_steps=30, mean_disp=0.3, abs_disp=True, rng=rng
)
wind_disp_div = 100

wind_u    = MeanRevertingProcess.from_half_life(
    mu=0.0, half_life_steps=20,
    mean_disp=av_size/wind_disp_div,
    abs_disp=True, rng=rng
)
wind_v    = MeanRevertingProcess.from_half_life(
    mu=0.0, half_life_steps=20,
    mean_disp=av_size/wind_disp_div,
    abs_disp=True, rng=rng
)

field_proc = GaussianFieldProcess(shape, sigma=av_size/30, rho=0.9, rng=rng)
rain_gen   = RainGenerator(thr_proc, wind_u, wind_v, field_proc)


collector = FrameCollector(enable=True)

engine = FireSpreadEngine(burn_spread_base=burn_spread_base,
                          kernel_sigma=3,
                          burn_rate=burn_rate,
                          rng=rng,
                          step_callback=collector)

fire_wind_disp_div = 200  # wind dispersion divisor for fire spread
fire_wind_u = MeanRevertingProcess.from_half_life(
    mu=0.0, half_life_steps=20,
    mean_disp=av_size/fire_wind_disp_div,
    abs_disp=True, rng=rng
)
fire_wind_v = MeanRevertingProcess.from_half_life(
    mu=0.0, half_life_steps=20,
    mean_disp=av_size/fire_wind_disp_div,
    abs_disp=True, rng=rng
)


num_ignitions = rounded_exponential(ignition_density * shape[0] * shape[1], rng)
print(f"Number of ignitions: {num_ignitions}")

for fire_id in range(num_ignitions):
    # Random ignition point
    ignition_xy = tuple(rng.integers(0, n, size=1)[0] for n in shape[::-1])[::-1]  # (row,col)
    print(f"Ignition {fire_id+1} at {ignition_xy}")

    # Burn the fire
    burnt_mask = engine.burn(ignition_xy=ignition_xy,
                             fuel=fuel,
                             rain_mask_iter=rain_gen,
                             wind_iter=zip(fire_wind_u, fire_wind_v, strict=False)
                             )

    # Reset the wind and rain generators for the next fire
    rain_gen.reset()
    fire_wind_u.reset()
    fire_wind_v.reset()
file_name = "single_season2.mp4"
collector.flush_to_mp4(Path("CA_Simulation/outputs") / file_name)
print(f"{file_name} written")

