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


# -----------------------------------------------------------------------------
# Helper: circular Boolean kernel
# -----------------------------------------------------------------------------

def circular_kernel(radius: int) -> np.ndarray:
    """Return a (2R+1)×(2R+1) boolean kernel with an inscribed circle."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius

# ─── helper (put near circular_kernel) ───────────────────────────────
def gaussian_kernel(radius: int) -> np.ndarray:
    """ℓ₁-normalised isotropic 2-D Gaussian with σ = radius."""
    size = 2 * radius + 1
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    g = np.exp(-(x**2 + y**2) / (2.0 * radius**2))
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
        kernel_radius: int = 10,
        burn_rate: float = 0.25,
        wind_stretch_coeff: float = 0.5,
        *,
        rng: Optional[np.random.Generator] = None,
        ignition_cluster: bool = True,
    ) -> None:
        if burn_rate <= 0:
            raise ValueError("burn_rate must be positive.")

        self.burn_spread_base = float(burn_spread_base)
        self.kernel_radius = int(kernel_radius)
        self.kernel_gauss    = gaussian_kernel(kernel_radius)
        self.burn_rate = float(burn_rate)
        self.ignition_cluster = ignition_cluster
        self.rng = np.random.default_rng() if rng is None else rng

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def burn(
        self,
        ignition_xy: Tuple[int, int],
        fuel: _FuelProtocol,
        rain_mask_iter: Iterable[np.ndarray],
        wind_iter: Iterable[Tuple[float, float]],
        *,
        record_frames: bool = False,
        out_mp4: Optional[str | Path] = None,
        dpi: int = 140,
        fps: int = 12,
        max_frames: Optional[int] = None,
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
        if self.ignition_cluster and self.kernel_radius > 1:
            init_r = self.kernel_radius // 2
            rr, cc = np.ogrid[:nrows, :ncols]
            ign_mask = (rr - r0) ** 2 + (cc - c0) ** 2 <= init_r ** 2
            burning |= ign_mask & (fuel_arr > 0)
        else:
            if fuel_arr[r0, c0] > 0:
                burning[r0, c0] = True

        if not burning.any():
            return burnt  # nothing to burn

        # ── 2. Animation buffers
        frames: List[np.ndarray] = []
        wind_hist: List[tuple[float, float]] = [] 
        save_frames = record_frames and (max_frames is None or max_frames > 0)

        # ── 3. Main CA loop
        step = 0
        rain_iter = iter(rain_mask_iter)
        while burning.any():
            rain = next(rain_iter)
            u,v = next(wind_iter, (0.0, 0.0))  # default wind if not provided
            wind_hist.append((u, v))

            # 3.1 snapshot before updates
            if save_frames and (max_frames is None or step < max_frames):
                frames.append(self._rgba_snapshot(fuel_arr, burning, burnt, rain))
            if max_frames is not None and step >= max_frames:
                save_frames = False

            # 3.2 consume fuel on current burning frontier
            fuel_arr[burning] -= self.burn_rate
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

        # ── 4. Save MP4 if requested
        if record_frames and frames and out_mp4 is not None:
            print(f"Calc done, saving animation to {out_mp4}")
            self._save_animation(frames, wind_hist, out_mp4, dpi, fps)

        return burnt

    # ------------------------------------------------------------------
    #  Animation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rgba_snapshot(
        fuel_arr: np.ndarray,
        burning: np.ndarray,
        burnt: np.ndarray,
        rain: np.ndarray,
    ) -> np.ndarray:
        rgb = np.zeros(fuel_arr.shape + (3,), dtype=np.float32)
        rgb[..., 1] = fuel_arr                            # green = fuel level
        rgb[..., 2] = np.where(rain, 0.7, rgb[..., 2])    # blue = rain
        rgb[burnt] = 0.0                                  # black = burnt
        rgb[burning] = (1.0, 0.0, 0.0)                    # red = burning
        alpha = np.ones(fuel_arr.shape, dtype=np.float32)
        return np.dstack((rgb, alpha))

    @staticmethod
    def _save_animation(
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
        # white arrow anchored in the top-left corner (axes-fraction coords)
        arrow = ax.annotate(
            '', xy=(0.15, 0.92), xytext=(0.05, 0.92),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='-|>', color='white', lw=2)
        )
        # speed label just below the arrow tail
        label = ax.text(
            0.05, 0.87, '', transform=ax.transAxes,
            color='white', fontsize=8,
            bbox=dict(facecolor='black', alpha=0.4, pad=2)
        )
        ax.set_xticks([])
        ax.set_yticks([])

        def _update(k: int):
            im.set_data(frames[k])
            u, v = wind_hist[k]
            speed = np.hypot(u, v)

            # normalise arrow to a fixed length in “axes fraction” space
            if speed > 1e-6:
                scale = 0.08                              # arrow length (0–1 axes units)
                dx, dy = (u / speed) * scale, (v / speed) * scale
                arrow.xy     = (0.05 + dx, 0.92 + dy)     # arrow head
                arrow.set_visible(True)
            else:                                         # calm ⇒ hide arrow
                arrow.set_visible(False)

            label.set_text(f'{speed:.1f} px/step')

            return im, arrow, label

        anim = FuncAnimation(fig, _update, frames=len(frames), interval=1000 / fps)
        anim.save(out_path, writer="ffmpeg", dpi=dpi, fps=fps)
        plt.close(fig)


def simulate_season(
        shape,
        ignition_density,
        burn_spread_base,
        burn_rate,
        kernel_radius,
        rng=None,
        plot=True,
        out_mp4="outputs/season.mp4",
):
    rng = np.random.default_rng() if rng is None else rng
    npix = shape[0] * shape[1]

    # 1. World state ---------------------------------------------------
    fuel   = Fuel(shape, sigma=10, rng=rng)
    state  = np.zeros(shape, np.uint8)          # 0/1/2 as agreed
    id_map = np.zeros(shape, np.int32)          # final fire IDs

    # static configs copied from run_single_fire3.py ------------------
    rain_template = build_rain_generator(shape, rng)
    wind_template = build_wind_iter(shape, rng)
    engine        = FireSpreadEngine(burn_spread_base,
                                     kernel_radius,
                                     burn_rate,
                                     rng=rng)

    # 2. How many fires? ----------------------------------------------
    n_fires = rounded_exponential(mean=ignition_density*npix, rng=rng)

    frames_all = [] if plot else None
    for fid in range(1, n_fires+1):

        # 2.1 choose ignition ----------------------------------------
        ignition = random_uniform_pixel(shape, rng)
        while state[ignition] == 2:             # skip already-burnt
            ignition = random_uniform_pixel(shape, rng)

        # 2.2 reset weather ------------------------------------------
        rain = rain_template.clone_and_reset()
        wind = wind_template.clone_and_reset()

        # 2.3 run fire -----------------------------------------------
        collector = FrameCollector(plot)
        burnt_mask = engine.burn(
            ignition_xy=ignition,
            fuel=fuel,
            rain_mask_iter=rain,
            wind_iter=wind,
            step_callback=collector  # new
        )

        # 2.4 update season-level state ------------------------------
        state[burnt_mask]  = 2
        id_map[burnt_mask] = fid
        if plot:
            frames_all.extend( collector.frames_with_label(fid) )

    # 3. Output --------------------------------------------------------
    if plot:
        assemble_mp4(frames_all, out_mp4)
        plot_final_map(id_map, state)

    return id_map
