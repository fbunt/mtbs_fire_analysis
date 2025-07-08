# fire_spread.py
"""
Fire‐spread kernel CA for a *single* ignition on a 2-D fuel grid.

Dependencies
------------
numpy, scipy.ndimage, matplotlib (for optional animation).
"""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
from scipy.ndimage import shift as nd_shift
import imageio.v3 as iio
from math import ceil


# -----------------------------------------------------------------------------
# Helper: circular Boolean kernel
# -----------------------------------------------------------------------------

def circular_kernel(radius: int) -> np.ndarray:
    """Return a (2R+1)×(2R+1) boolean kernel with an inscribed circle."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius

# ─── helper (put near circular_kernel) ───────────────────────────────
def gaussian_kernel(sigma: int) -> np.ndarray:
    """ℓ₁-normalised isotropic 2-D Gaussian with σ = radius."""
    k = 2
    radius = ceil(sigma * k)
    if k < 1:
        raise ValueError("radius must be at least 1.")
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    g /= g.sum()
    return g.astype(np.float32)

# -----------------------------------------------------------------------------
# Minimal protocol for the Fuel object (it just needs an .array attr)
# -----------------------------------------------------------------------------

class _FuelProtocol:  # runtime duck‑typing — avoids typing.Protocol
    array: np.ndarray


# -----------------------------------------------------------------------------
# Fire‑spread engine
# -----------------------------------------------------------------------------

class FireSpreadEngine:
    """Simulate one wildfire ignition until it extinguishes naturally."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        burn_spread_base: float,
        kernel_sigma: int = 3,
        burn_rate: float = 0.25,
        *,
        rng: Optional[np.random.Generator] = None,
        step_callback: Optional[callable] = None,
        ignition_cluster: bool = True,
    ) -> None:
        if burn_rate <= 0:
            raise ValueError("burn_rate must be positive.")

        self.burn_spread_base = float(burn_spread_base)
        self.kernel_sigma = kernel_sigma
        self.kernel_gauss    = gaussian_kernel(int(kernel_sigma))
        self.burn_rate = float(burn_rate)
        self.ignition_cluster = ignition_cluster
        self.rng = np.random.default_rng() if rng is None else rng
        self._callback = step_callback
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def burn(
        self,
        ignition_xy: Tuple[int, int],
        fuel: _FuelProtocol,
        rain_mask_iter: Iterable[np.ndarray],
        wind_iter: Iterable[Tuple[float, float]]
    ) -> np.ndarray:
        """Run the CA until the frontier dies. Returns a boolean burnt mask."""
        fuel_arr = fuel.array  # mutable reference
        nrows, ncols = fuel_arr.shape
        r0, c0 = ignition_xy
        if not (0 <= r0 < nrows and 0 <= c0 < ncols):
            raise ValueError("Ignition point outside domain.")

        # ── 0. State masks
        burning = np.zeros_like(fuel_arr, dtype=bool)
        burnt = np.zeros_like(fuel_arr, dtype=bool)

        # ── 1. Light the initial ignition cluster (MUCH simpler!)
        if self.ignition_cluster and self.kernel_sigma > 1:
            init_r = self.kernel_sigma // 2
            rr, cc = np.ogrid[:nrows, :ncols]
            ign_mask = (rr - r0) ** 2 + (cc - c0) ** 2 <= init_r ** 2
            burning |= ign_mask & (fuel_arr > 0)
        else:
            if fuel_arr[r0, c0] > 0:
                burning[r0, c0] = True

        if not burning.any():
            return burnt  # nothing to burn

        # ── 2. Animation buffers
        wind_hist: List[tuple[float, float]] = []

        # ── 3. Main CA loop
        step = 0
        rain_iter = iter(rain_mask_iter)
        while burning.any():
            rain = next(rain_iter)
            u,v = next(wind_iter, (0.0, 0.0))  # default wind if not provided
            wind_hist.append((u, v))

            # 3.1 snapshot before updates
            if self._callback:
                self._callback(step, burning, burnt, rain, fuel_arr, (u, v))
            # consume fuel on current burning frontier (clipped at zero)
            fuel_arr[burning] = np.maximum(fuel_arr[burning] - self.burn_rate, 0.0)
            burnt_now = burning & (fuel_arr <= 0)
            burning &= ~burnt_now
            burnt |= burnt_now

            # 3.3 determine new ignitions
            influence = convolve(burning.astype(np.float32), self.kernel_gauss, mode="constant")

            # 2. push that influence *down-wind* by (u, v)
            #    ndarray coords: axis-0 = y (rows), axis-1 = x (cols) → shift = (-v, -u)
            if abs(u) > 1e-6 or abs(v) > 1e-6:
                influence = nd_shift(influence,
                                    shift=(v, u),      # sub-pixel OK
                                    order=1,             # bilinear
                                    mode="constant",
                                    cval=0.0)            # pad with zeros
            candidates = (influence > 0) & ~burning & ~burnt & (fuel_arr > 0)
            if candidates.any():
                density = influence[candidates]
                p = (
                    fuel_arr[candidates]
                    * self.burn_spread_base
                    * density
                    * np.where(rain[candidates], 0.1, 1.0)
                )
                ignite = self.rng.random(p.size) < np.clip(p, 0.0, 1.0)
                if ignite.any():
                    burning.flat[np.flatnonzero(candidates)[ignite]] = True

            step += 1


        return burnt


_PX = 4           # scale-up factor for prettier videos


# ---------------------------------------------------------------------
# Snapshot builder  (fuel → green, rain → blue, burning → red, burnt → black)
# ---------------------------------------------------------------------
def rgba_snapshot(
    fuel_arr: np.ndarray,
    burning: np.ndarray,
    burnt: np.ndarray,
    rain: np.ndarray,
) -> np.ndarray:
    rgb = np.zeros(fuel_arr.shape + (3,), dtype=np.float32)
    rgb[..., 1] = fuel_arr                             # green = fuel level
    rgb[..., 2] = np.where(rain, 0.7, rgb[..., 2])     # blue = rain
    rgb[burnt]   = 0.0                                 # black = burnt
    rgb[burning] = (1.0, 0.0, 0.0)                     # red = burning
    alpha = np.ones(fuel_arr.shape, dtype=np.float32)
    return np.dstack((rgb, alpha))


# ---------------------------------------------------------------------
# MP4 writer  (identical to _save_animation in fire_spread.py)
# ---------------------------------------------------------------------
def save_animation(
    frames: List[np.ndarray],
    wind_hist: List[Tuple[float, float]],
    out_path: str | Path,
    dpi: int,
    fps: int,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    im = ax.imshow(frames[0], origin="lower")

    # white wind arrow (axes-fraction coords)
    arrow = ax.annotate(
        '', xy=(0.15, 0.92), xytext=(0.05, 0.92),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='-|>', color='white', lw=2)
    )
    label = ax.text(
        0.05, 0.87, '', transform=ax.transAxes,
        color='white', fontsize=8,
        bbox=dict(facecolor='black', alpha=0.4, pad=2)
    )
    ax.set_xticks([]); ax.set_yticks([])

    def _update(k: int):
        im.set_data(frames[k])
        u, v = wind_hist[k]
        speed = float(np.hypot(u, v))

        if speed > 1e-6:          # show arrow
            scale = 0.08
            dx, dy = (u / speed) * scale, (v / speed) * scale
            arrow.xy     = (0.05 + dx, 0.92 + dy)
            arrow.set_visible(True)
        else:
            arrow.set_visible(False)

        label.set_text(f'{speed:.1f} px/step')
        return im, arrow, label

    ani = FuncAnimation(fig, _update, frames=len(frames), interval=1000 / fps)
    ani.save(out_path, writer="ffmpeg", dpi=dpi, fps=fps)
    plt.close(fig)


class LegacyCollector:
    def __init__(self, enable: bool = True, dpi: int = 140, fps: int = 12):
        self.enable = enable
        self.dpi = dpi
        self.fps = fps
        self.frames: List[np.ndarray] = []
        self.wind_hist: List[Tuple[float, float]] = []

    # called once per CA step by the engine
    def __call__(self, step, burning, burnt, rain, fuel_arr, wind):
        if not self.enable:
            return
        self.frames.append(rgba_snapshot(fuel_arr, burning, burnt, rain))
        self.wind_hist.append(wind)

    # after the fire finishes
    def flush_to_mp4(self, outfile: str):
        if self.enable and self.frames:
            save_animation(
                self.frames,
                self.wind_hist,
                outfile,
                dpi=self.dpi,
                fps=self.fps,
            )
            self.frames.clear()
            self.wind_hist.clear()