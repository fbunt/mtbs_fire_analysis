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
    Fuel
)

# ---------------------------------------------------------------------
# Parameters (same as before)
# ---------------------------------------------------------------------
shape        = (200, 200)
sigma_field  = 8
n_steps      = 100
rng          = np.random.default_rng(42)

fuel_tracker = Fuel(shape, sigma=sigma_field, rng=rng)

frames_original, frames_masked, thresholds = [], [], []

for _ in range(n_steps):
    fuel = fuel_tracker.update(0.02)  # update the fuel field
    frames_original.append(fuel.copy())  # store the original fuel field
         # keep the threshold for titles

print('finished simulation, starting plots')


fig, axes = plt.subplots(1, 1, figsize=(5, 5))
im0 = axes.imshow(frames_original[0], origin="lower",
                    vmin=0, vmax=1, cmap="viridis")
axes.set_title("Fuel field")
axes.set_xticks([]); axes.set_yticks([])

def update(frame):
    im0.set_data(frames_original[frame])
    axes.set_title(f"Fuel field (step {frame})")
    return im0,

ani = FuncAnimation(fig, update, frames=n_steps, interval=120, blit=True)

ani.save("CA_Simulation/outputs/fuel_animation.mp4",
         writer="ffmpeg", fps=10)

