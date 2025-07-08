import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from CA_Simulation.CA_helpers import gauss_to_uniform   # unchanged helper

# ---------------------------------------------------------------------
#  Imports for the new stochastic-process classes
# ---------------------------------------------------------------------
from CA_Simulation.CA_helpers import (        # wherever you put the classes
    GaussianFieldProcess,
    MeanRevertingProcess,
    RainGenerator,
)

# ---------------------------------------------------------------------
# Parameters (same as before)
# ---------------------------------------------------------------------
shape        = (200, 200)
sigma_field  = 8
n_steps      = 1000
rng          = np.random.default_rng(42)

# Wind processes (pixels per step)
wind_u_proc = MeanRevertingProcess(
    mu=0.0, alpha=0.025, sigma=0.5, clamp=(-10, 10), rng=rng
)
wind_v_proc = MeanRevertingProcess(
    mu=0.0, alpha=0.025, sigma=0.5, clamp=(-10, 10), rng=rng
)

# Threshold process (rain probability level)
thr_proc = MeanRevertingProcess(
    mu=0.8, alpha=0.05, sigma=0.05, clamp=(0.0, 1.25), rng=rng
)

# Spatial rain-intensity field (AR(1) in time, Gaussian in space)
field_proc = GaussianFieldProcess(
    shape=shape, sigma=sigma_field, rho=0.9, rng=rng
)

# Master generator that ties everything together
rain = RainGenerator(thr_proc, wind_u_proc, wind_v_proc, field_proc)

# ---------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------
frames_original, frames_masked, thresholds = [], [], []

for _ in range(n_steps):
    mask = rain.update()                       # binary rain / no-rain
    field_uniform = gauss_to_uniform(rain.field.state)   # viewing convenience
    frames_original.append(field_uniform)

    # Mask out dry pixels for visualisation
    frames_masked.append(np.where(mask, field_uniform, np.nan))
    thresholds.append(rain.thr.x)              # keep the threshold for titles

print('finished simulation, starting plots')

# ---------------------------------------------------------------------
# Plotting / animation (unchanged)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im0 = axes[0].imshow(frames_original[0], origin="lower",
                     vmin=0, vmax=1, cmap="viridis")
axes[0].set_title("Evolving field")
axes[0].set_xticks([]); axes[0].set_yticks([])

im1 = axes[1].imshow(frames_masked[0], origin="lower",
                     vmin=0, vmax=1, cmap="viridis")
axes[1].set_title(f"Masked (threshold={thresholds[0]:.2f})")
axes[1].set_xticks([]); axes[1].set_yticks([])

plt.tight_layout()

def update(frame):
    im0.set_data(frames_original[frame])
    im1.set_data(frames_masked[frame])
    axes[0].set_title(f"Field (step {frame})")
    axes[1].set_title(f"Masked (thr={thresholds[frame]:.2f})")
    return im0, im1

ani = FuncAnimation(fig, update, frames=n_steps, interval=120, blit=True)

ani.save("CA_Simulation/outputs/rain_animation.mp4",
         writer="ffmpeg", fps=10)
