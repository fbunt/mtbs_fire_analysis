# %% run_single_fire.py
"""
Quick demo: 256×256 grid, one random ignition, animated MP4 written to ./outputs.
"""

from pathlib import Path
import numpy as np

from CA_Simulation.CA_helpers import Fuel, RainGenerator, GaussianFieldProcess, MeanRevertingProcess
from CA_Simulation.fire_spread2 import FireSpreadEngine

# ── Configuration ────────────────────────────────────────────────────
shape = (512, 512)
burn_spread_base = 0.1
burn_rate = 0.075
kernel_sigma = 2.5
rng = np.random.default_rng()

# Fuel: spatially correlated U(0,1)
fuel = Fuel(shape, sigma=3, rng=rng)

# %%

# Rain generator (copied from your existing pattern, but smaller & faster)
thr_proc  = MeanRevertingProcess(mu=0.75, alpha=0.02, sigma=0.1, clamp=(0, 1.5), rng=rng)
wind_u    = MeanRevertingProcess(mu=0.0, alpha=0.02, sigma=0.05, clamp=(-4, 4), rng=rng)
wind_v    = MeanRevertingProcess(mu=0.0, alpha=0.02, sigma=0.05, clamp=(-4, 4), rng=rng)
field_proc = GaussianFieldProcess(shape, sigma=4, rho=0.9, rng=rng)
rain_gen   = RainGenerator(thr_proc, wind_u, wind_v, field_proc)

def rain_iterator():
    """Infinite generator that yields a fresh rain mask every CA step."""
    while True:
        yield rain_gen.update()

# Single ignition point
ignition_point = tuple(rng.integers(0, n, size=1)[0] for n in shape[::-1])[::-1]  # (row,col)

print(f"Ignition at {ignition_point}")

fire_wind_u = MeanRevertingProcess(mu=0.0, alpha=0.02, sigma=0.05, clamp=(-4, 4), rng=rng)
fire_wind_v = MeanRevertingProcess(mu=0.0, alpha=0.02, sigma=0.05, clamp=(-4, 4), rng=rng)

engine = FireSpreadEngine(fuel,burn_rate,burn_spread_base, kernel_sigma, zip(fire_wind_u, fire_wind_v, strict=False), rain_iter=rain_gen, rng=rng)

# random ignition point
#ignition_point = np.random.Generator.uniform(0, shape[0]), np.random.Generator.uniform(0, shape[1])


stats = engine.burn(100_000,ignition_point)

for k,v in stats.items():
    print(f"{k}: {v}")

# %%
