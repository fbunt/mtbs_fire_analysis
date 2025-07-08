# %% run_single_fire.py
"""
Quick demo: 256×256 grid, one random ignition, animated MP4 written to ./outputs.
"""

from pathlib import Path
import numpy as np

from CA_Simulation.CA_helpers import Fuel, RainGenerator, GaussianFieldProcess, MeanRevertingProcess
from CA_Simulation.fire_spread import FireSpreadEngine

# ── Configuration ────────────────────────────────────────────────────
shape = (512, 512)
burn_spread_base = 0.35
burn_rate = 0.075
kernel_radius = 3
rng = np.random.default_rng()

# Fuel: spatially correlated U(0,1)
fuel = Fuel(shape, sigma=10, rng=rng)

# %%

# Rain generator (copied from your existing pattern, but smaller & faster)
thr_proc  = MeanRevertingProcess(mu=0.85, alpha=0.02, sigma=0.1, clamp=(0, 1.5), rng=rng)
wind_u    = MeanRevertingProcess(mu=0.0, alpha=0.02, sigma=0.5, clamp=(-4, 4), rng=rng)
wind_v    = MeanRevertingProcess(mu=0.0, alpha=0.02, sigma=0.5, clamp=(-4, 4), rng=rng)
field_proc = GaussianFieldProcess(shape, sigma=16, rho=0.9, rng=rng)
rain_gen   = RainGenerator(thr_proc, wind_u, wind_v, field_proc)

# Single ignition point
ignition = tuple(rng.integers(0, n, size=1)[0] for n in shape[::-1])[::-1]  # (row,col)

print(f"Ignition at {ignition}")

engine = FireSpreadEngine(burn_spread_base, kernel_radius, burn_rate, rng=None)

fire_wind_u = MeanRevertingProcess(mu=0.0, alpha=0.01, sigma=0.15, clamp=(-4, 4), rng=rng)
fire_wind_v = MeanRevertingProcess(mu=0.0, alpha=0.01, sigma=0.15, clamp=(-4, 4), rng=rng)

burnt_mask = engine.burn(
    ignition_xy=ignition,
    fuel=fuel,
    rain_mask_iter=rain_gen,
    wind_iter=zip(fire_wind_u, fire_wind_v, strict=False),  # wind for fire spread
    record_frames=True,
    out_mp4=Path("CA_Simulation/outputs") / "single_fire13.mp4",
)

print(f"Fire area (pixels): {burnt_mask.sum()}")

# %%
