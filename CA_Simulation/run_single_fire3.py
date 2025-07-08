# %% run_single_fire.py
"""
Quick demo: 256×256 grid, one random ignition, animated MP4 written to ./outputs.
"""

from pathlib import Path
import numpy as np

from CA_Simulation.CA_helpers import Fuel, RainGenerator, GaussianFieldProcess, MeanRevertingProcess
from CA_Simulation.fire_spread import FireSpreadEngine

# ── Configuration ────────────────────────────────────────────────────
shape = (128, 128)
burn_spread_base = 0.35
burn_rate = 0.075
kernel_radius = 2
rng = np.random.default_rng()

# Fuel: spatially correlated U(0,1)
fuel = Fuel(shape, sigma=10, rng=rng)

# %%

# Rain generator (copied from your existing pattern, but smaller & faster)
thr_proc  = MeanRevertingProcess.from_half_life(
    mu=0.85, half_life_steps=20, mean_disp=0.3, abs_disp=True, rng=rng
)
wind_disp_div = 100

wind_u    = MeanRevertingProcess.from_half_life(
    mu=0.0, half_life_steps=20,
    mean_disp=shape[0]/wind_disp_div,
    abs_disp=True, rng=rng
)
wind_v    = MeanRevertingProcess.from_half_life(
    mu=0.0, half_life_steps=20,
    mean_disp=shape[1]/wind_disp_div,
    abs_disp=True, rng=rng
)

field_proc = GaussianFieldProcess(shape, sigma=4, rho=0.9, rng=rng)
rain_gen   = RainGenerator(thr_proc, wind_u, wind_v, field_proc)

# Single ignition point
ignition = tuple(rng.integers(0, n, size=1)[0] for n in shape[::-1])[::-1]  # (row,col)

print(f"Ignition at {ignition}")

engine = FireSpreadEngine(burn_spread_base, kernel_radius, burn_rate, rng=None)

fire_wind_disp_div = 200  # wind dispersion divisor for fire spread
fire_wind_u = MeanRevertingProcess.from_half_life(
    mu=0.0, half_life_steps=20,
    mean_disp=shape[0]/fire_wind_disp_div,
    abs_disp=True, rng=rng
)
fire_wind_v = MeanRevertingProcess.from_half_life(
    mu=0.0, half_life_steps=20,
    mean_disp=shape[1]/fire_wind_disp_div,
    abs_disp=True, rng=rng
)

burnt_mask = engine.burn(
    ignition_xy=ignition,
    fuel=fuel,
    rain_mask_iter=rain_gen,
    wind_iter=zip(fire_wind_u, fire_wind_v, strict=False),  # wind for fire spread
    record_frames=True,
    out_mp4=Path("CA_Simulation/outputs") / "single_fireold14.mp4",
)

print(f"Fire area (pixels): {burnt_mask.sum()}")

# %%
