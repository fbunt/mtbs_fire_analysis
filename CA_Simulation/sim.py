import numpy as np
from typing import Tuple, Dict, Optional

# -----------------------------------------------------------------------------
# External stochastic‑process helpers (already implemented elsewhere)
# -----------------------------------------------------------------------------
from CA_Simulation.CA_helpers import (
    correlated_gaussian,  # correlated_gaussian(cov: np.ndarray, size: Tuple[int, int]) -> np.ndarray
    gauss_to_uniform,     # gauss_to_uniform(z: np.ndarray) -> np.ndarray
    mean_reverting,       # mean_reverting(x0: float, mu: float, kappa: float, sigma: float, dt: float, n: int) -> np.ndarray
)

import numpy as np
from typing import Tuple, Optional

# -----------------------------------------------------------------------------
#  External stochastic‑process helpers (do NOT re‑implement here)
# -----------------------------------------------------------------------------
from CA_Simulation.CA_helpers import (
    correlated_gaussian,  # correlated_gaussian(shape, sigma, rng=None, wrap=True)
    gauss_to_uniform,     # gauss_to_uniform(z)
    mean_reverting,       # mean_reverting(n_steps, mu, alpha, sigma, clamp=None, rng=None)
)

__all__ = ["FireSimulation"]


class FireSimulation:
    """Continuous‑time wildfire CA with per‑fire rain reset and separate winds.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        Raster size ``(ny, nx)``.
    sigma_wind : float, default 10.0
        Spatial s.d. (pixels) for the Gaussian kernel that corrrelates the *wind
        direction* fields.  The same value is used for both fire‑level and
        rain‑level winds.
    sigma_rain : float, default 10.0
        Gaussian‑kernel width for the per‑fire *rain threshold* field.
    rng : numpy.random.Generator, optional
        RNG for full reproducibility.  If *None*, a new generator is created.
    """

    # ------------------------------------------------------------------ #
    #  Construction & persistent state
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        grid_shape: Tuple[int, int],
        *,
        sigma_wind: float = 10.0,
        sigma_rain: float = 10.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.grid_shape = grid_shape
        self.sigma_wind = sigma_wind
        self.sigma_rain = sigma_rain
        self.rng = np.random.default_rng() if rng is None else rng

        #  Fuel never resets during a year; burned resets each 1 January.
        self.fuel = np.ones(grid_shape, dtype=float)  # 1 ⇒ pristine fuel
        self.burned = np.zeros(grid_shape, dtype=bool)
        self.year = 0
        self.time = 0

        #  Per‑timestep mean‑reverting *scalar* wind speeds (not directions).
        #  We pre‑draw a very long series and advance an index for speed.
        self._wind_fire_ts = mean_reverting(
            n_steps=1_000_000, mu=5.0, alpha=0.10, sigma=1.0, rng=self.rng
        )
        self._wind_rain_ts = mean_reverting(
            n_steps=1_000_000, mu=8.0, alpha=0.05, sigma=1.2, rng=self.rng
        )
        self._wind_idx = 0

        #  Set up per‑fire state (rain threshold, wind directions, etc.).
        self._init_fire_state()

    # ------------------------------------------------------------------ #
    #  Lifecycle helpers
    # ------------------------------------------------------------------ #
    def reset_year(self) -> None:
        """Clear the *burned* mask but keep the residual fuel in place."""
        self.burned.fill(False)
        self.year += 1

    def new_fire(self) -> None:
        """Start a new ignition event, resetting rain & wind fields."""
        self._init_fire_state()

    # ................................................................. #
    def _init_fire_state(self) -> None:
        """Initialise all variables that live only for a single fire run."""
        ny, nx = self.grid_shape

        #  (1) Per‑fire rain‑extinguish threshold ϑ(x,y)
        self.rain_threshold = correlated_gaussian(
            (ny, nx), sigma=self.sigma_rain, rng=self.rng
        )

        #  (2) Direction fields for *two* wind layers (fire‑level & rain‑level)
        self.wind_dir_fire = correlated_gaussian(
            (ny, nx), sigma=self.sigma_wind, rng=self.rng
        ) * np.pi  # scale → [‑π, π]

        self.wind_dir_rain = correlated_gaussian(
            (ny, nx), sigma=self.sigma_wind, rng=self.rng
        ) * np.pi

        #  (3) Accumulated rain since ignition
        self.cum_rain = np.zeros((ny, nx), dtype=float)

    # ------------------------------------------------------------------ #
    #  Top‑level timestep
    # ------------------------------------------------------------------ #
    def step(self) -> None:
        """Advance the simulation by **one** discrete‑time tick."""
        ny, nx = self.grid_shape

        #  (1) Draw *scalar* speeds from the pre‑computed OU series
        v_fire = self._wind_fire_ts[self._wind_idx]
        v_rain = self._wind_rain_ts[self._wind_idx]
        self._wind_idx += 1

        #  (2) Update rainfall field & accumulate
        rain_now = gauss_to_uniform(
            correlated_gaussian((ny, nx), sigma=3.0, rng=self.rng)
        )
        dy_r = np.round(np.sin(self.wind_dir_rain) * v_rain).astype(int)
        dx_r = np.round(np.cos(self.wind_dir_rain) * v_rain).astype(int)
        rain_shifted = np.roll(rain_now, shift=(dy_r, dx_r), axis=(0, 1))
        self.cum_rain += rain_shifted

        #  (3) Spread the fire
        ignitions = self._spread_step(v_fire)
        self.burned |= ignitions
        self.fuel[ignitions] = np.maximum(0.0, self.fuel[ignitions] - 0.2)

        #  (4) Rain extinguishes where cumulative exceeds threshold
        self.burned[self.cum_rain > self.rain_threshold] = False

        self.time += 1

    # ------------------------------------------------------------------ #
    #  Fire‑spread kernel (very simple placeholder)
    # ------------------------------------------------------------------ #
    def _spread_step(self, v_fire: float) -> np.ndarray:
        """Return a boolean mask of *newly ignited* cells at this time step."""
        ny, nx = self.grid_shape
        dy = np.round(np.sin(self.wind_dir_fire) * v_fire).astype(int)
        dx = np.round(np.cos(self.wind_dir_fire) * v_fire).astype(int)

        src_y, src_x = np.where(self.burned)
        tgt_y = (src_y + dy[src_y, src_x]) % ny
        tgt_x = (src_x + dx[src_y, src_x]) % nx

        ignitable = (self.fuel[tgt_y, tgt_x] > 0.0) & (~self.burned[tgt_y, tgt_x])
        p = np.minimum(1.0, v_fire / 10.0)  # spread probability
        draws = self.rng.random(len(src_y)) < p

        mask = np.zeros((ny, nx), dtype=bool)
        mask[tgt_y[ignitable & draws], tgt_x[ignitable & draws]] = True
        return mask

    # ------------------------------------------------------------------ #
    #  Diagnostics helpers
    # ------------------------------------------------------------------ #
    def burned_fraction(self) -> float:
        """Return the fraction of the grid currently burning."""
        return float(self.burned.mean())



# -----------------------------------------------------------------------------
# Example usage (remove or wrap in __name__ guard in production)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ny, nx = 200, 200
    # cov = np.eye(ny * nx)  # placeholder; user should supply realistic cov

    # def dummy_wind(rng):
    #     return rng.normal(0, 1, size=(ny, nx))

    # def dummy_ignite(rng):
    #     return tuple(rng.integers(0, s) for s in (ny, nx))

    # def fuel_recovery(fuel, dt_days):
    #     return np.minimum(1.0, fuel + 0.1 * (dt_days / 365.0))

    sim = FireSimulation(
        grid_shape=(ny, nx)
    )

    sim.run(n_years=3, episodes_per_year=10)
    print(sim.yearly_stats)
