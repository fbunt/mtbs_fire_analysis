import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

from CA_Simulation.CA_helpers import (
    correlated_gaussian,
    gauss_to_uniform,
    mean_reverting,
)

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
shape = (200, 200)
sigma_field = 8
n_steps = 100
rng = np.random.default_rng(42)

# Wind signals (pixels per step, integer-rounded shifts)
vx = mean_reverting(n_steps, 0.0, 0.025, 0.4, (-2.5, 2.5), rng)
vy = mean_reverting(n_steps, 0.0, 0.025, 0.4, (-2.5, 2.5), rng)

# Threshold series for masking
thresholds = mean_reverting(n_steps, 0.8, 0.05, 0.05, (0, 1.25), rng)

# AR(1) coefficient
rho = 0.9
sqrt_inv = np.sqrt(1 - rho ** 2)

# ---------------------------------------------------------------------
# Initial field (Gaussian space)
# ---------------------------------------------------------------------
field = correlated_gaussian(shape, sigma_field, rng)

frames_original, frames_masked = [], []

for t in range(n_steps):
    # Wind advection (wrap edges)
    dx, dy = int(np.round(vx[t])), int(np.round(vy[t]))
    shifted = np.roll(np.roll(field, dy, axis=0), dx, axis=1)

    # New spatially correlated noise
    noise = correlated_gaussian(shape, sigma_field, rng)

    # AR(1) update keeps variance stationary
    field = rho * shifted + sqrt_inv * noise

    # Map to uniform [0,1] for display / threshold logic
    uniform = gauss_to_uniform(field)

    frames_original.append(uniform)
    thresh = thresholds[t]
    frames_masked.append(np.where(uniform >= thresh, uniform, np.nan))

# ---------------------------------------------------------------------
# Plotting / Animation
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im0 = axes[0].imshow(frames_original[0], origin="lower", vmin=0, vmax=1, cmap="viridis")
axes[0].set_title("Evolving field")
axes[0].set_xticks([]); axes[0].set_yticks([])

im1 = axes[1].imshow(frames_masked[0], origin="lower", vmin=0, vmax=1, cmap="viridis")
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

# write animation to CA_Simulation/outputs

ani.save("CA_Simulation/outputs/rain_animation.mp4", writer="ffmpeg", fps=10)


