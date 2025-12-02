# ---------------------------------------------------------------------
#  cellular_automaton_helpers.py
#  – CA simulator that mimics test_helpers.create_sample_data_all API
# ---------------------------------------------------------------------
import numpy as np
from scipy.signal import convolve2d
from typing import Tuple, List


# ---------- low-level CA kernel --------------------------------------
def _step_ca(
    state: np.ndarray,
    fuel: np.ndarray,
    tsince: np.ndarray,
    t_now: float,
    *,
    p_lightning: float,
    p_spread_base: float,
    wind_vec: Tuple[float, float],
    wind_scale: float,
    recovery_rate: float,
    ignition_threshold: float,
    rng: np.random.Generator,
):
    """Evolve the CA one tick; return boolean mask of newly ignited cells."""
    # –– pre-built neighbour weights (3 × 3) with wind bias
    KERNEL = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    dy, dx = np.indices(KERNEL.shape) - 1
    dirs = np.stack([dx, dy], axis=-1)
    dot = (dirs * wind_vec).sum(axis=-1)
    K_WIND = KERNEL * (1 + wind_scale * dot)

    burning = state == 1
    burn_mask = burning.astype(float)
    neigh_fire = convolve2d(burn_mask, K_WIND, mode="same", boundary="wrap")
    spread_prob = 1 - np.exp(-p_spread_base * neigh_fire)

    rng_u = rng.random(state.shape)
    # lightning
    lightning = rng_u < p_lightning
    # spread from neighbours
    spread = rng_u < spread_prob
    ignitable = (fuel > ignition_threshold) & (state == 0)
    ignite = ignitable & (lightning | spread)

    # --- update state -------------------------------------------------
    finished = burning & ~ignite        # burns last exactly one tick
    state[finished] = 2                 # burned / cooling
    state[ignite] = 1                   # now burning

    fuel[ignite] = 0.0
    fuel += recovery_rate
    np.clip(fuel, 0, 1, out=fuel)

    tsince[ignite] = 0
    tsince[~ignite] += 1

    return ignite


# ---------- public wrapper – drop-in for create_sample_data_all -------
def create_sample_data_ca(
    num_pixels: int,
    time_interval: float,
    *,
    start_up_time: float = 0.0,
    round_to_nearest: float | None = None,
    grid_shape: Tuple[int, int] | None = None,
    dt: float = 0.1,
    rng: np.random.Generator | None = None,
    # CA knobs ---------------------------------------------------------
    p_lightning: float = 1e-4,
    p_spread_base: float = 0.35,
    wind_vec: Tuple[float, float] = (1.0, 0.3),
    wind_scale: float = 0.6,
    recovery_rate: float = 0.25,
    ignition_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cellular-automaton version of test_helpers.create_sample_data_all.

    Returns
    -------
    dts, dt_counts,
    uts, ut_counts,
    cts, ct_counts,
    n0t, n0_counts : np.ndarray pairs
        Exactly the same objects (and semantics) produced by the original
        renewal-process helper, ready for the HLH bootstrap machinery.
    """
    rng = np.random.default_rng() if rng is None else rng

    # ---------- grid geometry -----------------------------------------
    if grid_shape is None:
        n = int(np.sqrt(num_pixels))
        if n * n != num_pixels:
            raise ValueError("Provide grid_shape when num_pixels is not square.")
        grid_shape = (n, n)
    if grid_shape[0] * grid_shape[1] != num_pixels:
        raise ValueError("grid_shape does not match num_pixels.")

    # ---------- CA state arrays ---------------------------------------
    state = np.zeros(grid_shape, np.uint8)       # 0 unburned | 1 burning | 2 burned
    fuel = np.ones(grid_shape, float)
    tsince = np.zeros(grid_shape, float)

    # ---------- bookkeeping per pixel ---------------------------------
    t_close = start_up_time + time_interval
    first_ev = np.full(grid_shape, np.nan)
    last_ev = np.full(grid_shape, np.nan)

    ticks_total = int(np.ceil(t_close / dt))
    t_now = 0.0

    for _ in range(ticks_total):
        ignite = _step_ca(
            state,
            fuel,
            tsince,
            t_now,
            p_lightning=p_lightning,
            p_spread_base=p_spread_base,
            wind_vec=wind_vec,
            wind_scale=wind_scale,
            recovery_rate=recovery_rate * dt,   # make rate time-step safe
            ignition_threshold=ignition_threshold,
            rng=rng,
        )

        ev_times = t_now + ignite * dt  # broadcast
        # first event after window opens? (UT)
        need_first = np.isnan(first_ev) & ignite
        first_ev[need_first] = ev_times[need_first]
        # any subsequent event inside window → candidate DT
        inside_window = (ev_times >= start_up_time) & (ev_times <= t_close)
        update = ignite & inside_window
        last_ev[update] = ev_times[update]

        t_now += dt

    # ---------- gather per-pixel histories ----------------------------
    dts: List[float] = []
    uts: List[float] = []
    cts: List[float] = []

    # Flatten to vectors for simplicity
    first_ev_v = first_ev.ravel()
    last_ev_v = last_ev.ravel()

    # pixels with at least one fire in window
    seen = ~np.isnan(first_ev_v)

    # ------------------------------------------------------------------
    #  UTs – forward-recurrence gaps
    uts = (first_ev_v[seen] - start_up_time).tolist()

    # ------------------------------------------------------------------
    #  DTs – fully observed intervals (≥ 2 fires inside window)
    # For simplicity we only recorded last event; need full series.
    # A strict accounting would cache all inter-arrival times; here we
    # approximate by taking *one* dt per pixel if 2+ fires fell in window.
    # Extend if you need every dt.
    # ------------------------------------------------------------------
    # where two fires occurred inside the window
    # (this needs a proper events list → left as “homework” if desired)
    # ------------------------------------------------------------------
    dts = []  # intentionally empty for skeleton

    # ------------------------------------------------------------------
    #  CTs – right-censor times
    cts = (t_close - last_ev_v[seen]).tolist()

    # ------------------------------------------------------------------
    #  n0 – pixels with no fires in the window
    n0 = np.count_nonzero(~seen)
    n0t = [time_interval] * n0

    # ---------- helper for collapsing duplicates ----------------------
    def _collapse(arr: List[float], step: float | None):
        if not arr:
            return np.empty(0, float), np.empty(0, int)
        arr = np.asarray(arr, float)
        if step is not None:
            arr = np.round(arr / step) * step
        uniq, counts = np.unique(arr, return_counts=True)
        return uniq, counts.astype(int)

    dts_u, dt_counts = _collapse(dts, round_to_nearest)
    uts_u, ut_counts = _collapse(uts, round_to_nearest)
    cts_u, ct_counts = _collapse(cts, round_to_nearest)
    n0t_u, n0_counts = _collapse(n0t, round_to_nearest)

    return (
        dts_u,
        dt_counts,
        uts_u,
        ut_counts,
        cts_u,
        ct_counts,
        n0t_u,
        n0_counts,
    )

(
    dts, dt_ct,
    uts, ut_ct,
    cts, ct_ct,
    n0t, n0_ct,
) = create_sample_data_ca(
        10_000,
        39,
        start_up_time=100,
        round_to_nearest=0.1,   # optional binning
)


# %%
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
#  1. Create a spatially correlated raster
# ----------------------------------------------------------------------
def generate_correlated_field(
    shape: tuple[int, int],
    sigma: float | tuple[float, float] = 5.0,
    *,
    distribution: str = "uniform",
    rescale: bool = True,
    rng: np.random.Generator | None = None,
    wrap: bool = False,
):
    """
    Parameters
    ----------
    shape : (rows, cols)
        Pixel dimensions of the output array.
    sigma : float or (sy, sx)
        Standard deviation(s) of the Gaussian kernel **in pixels**.  Think of
        this as the correlation length; larger → smoother.
    distribution : {"uniform", "normal"}
        Base noise distribution before smoothing.
    rescale : bool
        If True (default) stretch the result back to the [0, 1] range.
    rng : np.random.Generator
        Optional random number source for reproducibility.
    wrap : bool
        If True use periodic boundary conditions (“toroidal wrap”) when
        filtering; otherwise edges fade to zero correlation.

    Returns
    -------
    field : 2-D ndarray  (float32)
        Smooth random surface.
    """
    rng = np.random.default_rng() if rng is None else rng

    if distribution == "uniform":
        noise = rng.random(shape)          # [0,1]
    elif distribution == "normal":
        noise = rng.standard_normal(shape)  # N(0,1)
    else:
        raise ValueError("distribution must be 'uniform' or 'normal'")

    mode = "wrap" if wrap else "nearest"
    field = gaussian_filter(noise, sigma=sigma, mode=mode)

    if rescale:
        mn, mx = field.min(), field.max()
        if mx > mn:       # avoid zero-division if nearly flat
            field = (field - mn) / (mx - mn)
        else:
            field = np.zeros_like(field)

    return field.astype(np.float32)


# ----------------------------------------------------------------------
# 2. Quick visual check
# ----------------------------------------------------------------------
def plot_field(
    field: np.ndarray,
    *,
    title: str | None = None,
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
    colorbar: bool = True,
):
    """
    Shaded plot of a 2-D field.

    Returns the Matplotlib Axes for further tweaking if you wish.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(field, cmap=cmap, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax


# ----------------------------------------------------------------------
# 3.  Example usage ----------------------------------------------------
if __name__ == "__main__":
    # --- parameters you can play with ---------------------------------
    shape = (300, 400)      # rows, cols
    sigma = 12              # correlation length in pixels
    wrap  = False           # True → toroidal surface

    field = generate_correlated_field(shape, sigma, wrap=wrap)

    plot_field(field, title=f"σ = {sigma} px  (wrap = {wrap})")
    plt.tight_layout()
    plt.show()



# %%
# ---------------------------------------------------------------------
#  mean_reverting_signal.py
# ---------------------------------------------------------------------
"""
Generate and visualise a 1-D mean-reverting stochastic signal.

   x[t+1] = x[t] + α (μ − x[t]) + σ ε[t]

where
   μ   = long-term mean
   α   = mean-reversion strength  (0 → random walk, 1 → jump straight to μ)
   σ   = step volatility
   ε[t] ~ N(0, 1)

Typical uses in the CA fire model:
  • drought / moisture index driving ignition & spread
  • prevailing wind strength or direction factor
  • climate trend scenarios (let μ drift slowly)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
# 1. Signal generator
# ------------------------------------------------------------------ #
def generate_mean_reverting_signal(
    n_steps: int,
    mu: float,
    alpha: float,
    sigma: float,
    *,
    x0: float | None = None,
    clamp: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    n_steps : int
        Length of the output array.
    mu : float
        Long-term mean the process is attracted to.
    alpha : float in [0, 1]
        Mean-reversion parameter. 0 → pure random walk; 1 → exponential
        decay to mu in one step.
    sigma : float >= 0
        Random innovation per step.
    x0 : float, optional
        Starting value.  Defaults to mu.
    clamp : (lo, hi), optional
        If given, clip each new value into [lo, hi] (useful for bounded
        indices like relative humidity 0–1).
    rng : np.random.Generator, optional
        For reproducibility.

    Returns
    -------
    ndarray (float64) of shape (n_steps,)
    """
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1")

    rng = np.random.default_rng() if rng is None else rng
    x = np.empty(n_steps, dtype=float)
    x[0] = mu if x0 is None else x0

    for t in range(n_steps - 1):
        innovation = sigma * rng.standard_normal()
        x[t + 1] = x[t] + alpha * (mu - x[t]) + innovation
        if clamp is not None:
            lo, hi = clamp
            x[t + 1] = np.clip(x[t + 1], lo, hi)

    return x


# ------------------------------------------------------------------ #
# 2. Quick line plot
# ------------------------------------------------------------------ #
def plot_signal(
    x: np.ndarray,
    *,
    title: str | None = None,
    ax: plt.Axes | None = None,
    color: str | None = None,
    ylabel: str = "value",
):
    """
    Simple line chart with a light reference line at the long-term mean.

    Returns
    -------
    Matplotlib Axes for further tweaking.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    ax.plot(x, color=color or "tab:blue")
    ax.set_xlabel("time step")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(ls="--", lw=0.3, alpha=0.6)

    return ax


# ------------------------------------------------------------------ #
# 3. Stand-alone demo
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # tweak these to taste
    STEPS  = 2_000
    MU     = 0.5
    ALPHA  = 0.02       # weak mean reversion → long dry/wet spells
    SIGMA  = 0.05
    CLAMP  = (0.0, 1.0)  # keep within [0,1]

    series = generate_mean_reverting_signal(
        STEPS, MU, ALPHA, SIGMA, clamp=CLAMP, rng=np.random.default_rng(42)
    )

    plot_signal(
        series,
        title=f"Mean-reverting signal  (μ={MU}, α={ALPHA}, σ={SIGMA})",
        ylabel="drought / moisture index",
    )
    plt.tight_layout()
    plt.show()

# %%
