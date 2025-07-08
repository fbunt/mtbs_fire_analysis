import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def correlated_gaussian(shape, sigma, rng=None, wrap=True):
    """Zero-mean, unit-variance field with Gaussian spatial correlation."""
    rng = np.random.default_rng() if rng is None else rng
    white = rng.standard_normal(shape)
    mode  = "wrap" if wrap else "nearest"
    field = gaussian_filter(white, sigma=sigma, mode=mode)
    field -= field.mean()
    field /= field.std(ddof=0)
    return field.astype(np.float32, copy=False) 

def mean_reverting(n_steps, mu, alpha, sigma, clamp=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.empty(n_steps)
    x[0] = mu
    for t in range(n_steps - 1):
        x[t+1] = x[t] + alpha * (mu - x[t]) + sigma * rng.standard_normal()
        if clamp is not None:
            x[t+1] = np.clip(x[t+1], *clamp)
    return x

# CA_helpers.py  ── replace the existing helper
def gauss_to_uniform(z):
    """
    Map N(0,1) → U(0,1) using the Gaussian CDF.
    Always returns float32 without an extra copy when possible.
    """
    u = 0.5 * (1.0 + erf(z.astype(np.float32, copy=False)))
    return u.astype(np.float32, copy=False)

def half_life_to_alpha(H):
    """Convert half-life (in steps) to α."""
    H = np.asarray(H, float)
    return 1.0 - 2.0**(-1.0 / H)

def meta_to_alpha_sigma(half_life_steps, mean_displacement, abs_disp=False):
    """
    Convert (H, D) → (α, σ) for the MeanRevertingProcess.

    Parameters
    ----------
    half_life_steps : float
    mean_displacement : float
        RMS displacement if abs_disp=False (default);  
        mean-absolute displacement if abs_disp=True.
    abs_disp : bool
        Set True when D is an absolute rather than RMS spread.
    """
    alpha = half_life_to_alpha(half_life_steps)
    if abs_disp:                       # D = E|y| ⇒ convert to rms first
        mean_displacement *= np.sqrt(np.pi / 2.0)
    sigma = mean_displacement * np.sqrt(alpha * (2.0 - alpha))
    return alpha, sigma

class GaussianFieldProcess:
    """
    Temporal AR(1) sequence of spatially correlated Gaussian fields.

    Every update produces:
        F_{t+1} = ρ · F_t + √(1 − ρ²) · ε_{t+1},
    where ε_{t+1} ~ correlated_gaussian(shape, sigma).

    Parameters
    ----------
    shape : tuple[int, int]
        Grid shape (ny, nx).
    sigma : float | tuple[float, float]
        Spatial std-dev passed straight to `correlated_gaussian`.
    rho : float
        Temporal correlation coefficient between consecutive fields (|ρ| ≤ 1).
    rng : np.random.Generator, optional
        Random-number generator; falls back to `np.random.default_rng()`.
    wrap : bool, default True
        Passed verbatim to `correlated_gaussian`.
    """

    def __init__(
        self,
        shape,
        sigma,
        rho: float,
        rng: np.random.Generator | None = None,
        wrap: bool = True,
    ):
        self.shape = shape
        self.sigma = sigma
        self.rho = float(rho)
        self.rng = np.random.default_rng() if rng is None else rng
        self.wrap = wrap

        # Initial state (ρ=0 ensures i.i.d. start-up)
        self.state = correlated_gaussian(shape, sigma, rng=self.rng, wrap=wrap)

    # Alias so you can call the instance directly
    def __call__(self):
            return self.update()

    def update(self, dx: int = 0, dy: int = 0) -> np.ndarray:
        """Advance one step with an optional (dx, dy) shift."""
        adv = (
            np.roll(np.roll(self.state, dy, axis=0), dx, axis=1)
            if (dx or dy)
            else self.state
        )
        eps = correlated_gaussian(self.shape, self.sigma, rng=self.rng, wrap=self.wrap)
        self.state = self.rho * adv + np.sqrt(1.0 - self.rho**2) * eps
        return self.state

    def reset(self) -> np.ndarray:
        """One update with ρ = 0 (memory erased, fresh innovation)."""
        eps = correlated_gaussian(self.shape, self.sigma, rng=self.rng, wrap=self.wrap)
        self.state = eps
        return self.state


class MeanRevertingProcess:
    """
    Discrete-time Ornstein–Uhlenbeck process with optional bounds.

        x_{t+1} = x_t + α (μ − x_t) + σ ξ_t,   ξ_t ~ N(0, 1)

    Parameters
    ----------
    mu : float
        Long-run mean.
    alpha : float
        Mean-reversion strength (0 → pure random walk).
    sigma : float
        Innovation standard deviation.
    x0 : float, optional
        Initial value; defaults to μ.
    clamp : tuple[float, float] | None
        Optional (min, max) to clip the series each step.
    rng : np.random.Generator, optional
        Random source; defaults to `np.random.default_rng()`.
    """

    def __init__(
        self,
        mu: float,
        alpha: float,
        sigma: float,
        x0: float | None = None,
        clamp: tuple[float, float] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.clamp = clamp
        self.rng = np.random.default_rng() if rng is None else rng

        self.x = self.mu if x0 is None else float(x0)

    @classmethod
    def from_half_life(cls, mu, half_life_steps, mean_disp,
                       *, abs_disp=False, **kwargs):
        alpha, sigma = meta_to_alpha_sigma(
            half_life_steps, mean_disp, abs_disp=abs_disp
        )
        return cls(mu=mu, alpha=alpha, sigma=sigma, **kwargs)

    def __call__(self):
        return self.update()
    
    def __iter__(self):          # NEW
        return self

    def __next__(self):          # NEW
        return self.update()

    def update(self) -> float:
        """Advance one step and return the new value."""
        self.x += self.alpha * (self.mu - self.x) + self.sigma * self.rng.standard_normal()
        if self.clamp is not None:
            self.x = float(np.clip(self.x, *self.clamp))
        return self.x

    def reset(self) -> float:
        """One update with α = 0 (no mean-reversion, fresh shock only)."""
        self.x += self.sigma * self.rng.standard_normal()
        if self.clamp is not None:
            self.x = float(np.clip(self.x, *self.clamp))
        return self.x


class RainGenerator:
    """
    Parameters
    ----------
    threshold_proc : MeanRevertingProcess
        Produces a scalar threshold in [0,1] each step.
    wind_u_proc, wind_v_proc : MeanRevertingProcess
        Zonal (x) and meridional (y) wind components in **pixels per step**.
        They are used to advect the rain field before thresholding.
    field_proc : GaussianFieldProcess
        Produces the correlated-Gaussian rain intensity field.
    """

    def __init__(
        self,
        threshold_proc: MeanRevertingProcess,
        wind_u_proc: MeanRevertingProcess,
        wind_v_proc: MeanRevertingProcess,
        field_proc: GaussianFieldProcess,
    ):
        self.thr = threshold_proc
        self.wind_u = wind_u_proc
        self.wind_v = wind_v_proc
        self.field = field_proc

    def update(self) -> np.ndarray:
        # Step the scalar drivers
        threshold = self.thr.update()
        dx = int(round(self.wind_u.update()))   # +x is eastward (cols)
        dy = int(round(self.wind_v.update()))   # +y is northward (rows)

        # Update the spatial field with advection
        g = self.field.update(dx=dx, dy=dy)

        # Uniform transform for thresholding & easy visualisation
        u = gauss_to_uniform(g)
        return (u >= threshold)
    
    def __iter__(self):          # NEW
        return self

    def __next__(self):          # NEW
        return self.update()

    def reset(self):
        """Reset **all** sub-processes (fresh, uncorrelated state)."""
        self.thr.reset()
        self.wind_u.reset()
        self.wind_v.reset()
        self.field.reset()

class Fuel:
    """
    Fuel(x, y) in [0, 1] that can be *mutated externally* (burning) and grows
    stochastically via spatially-correlated additions.

    Parameters
    ----------
    shape : tuple[int, int]
        Grid shape (ny, nx).
    sigma : float | tuple[float, float]
        Spatial correlation length passed straight to `correlated_gaussian`.
    rng : np.random.Generator, optional
        Random source; defaults to `np.random.default_rng()`.
    wrap : bool, default True
        Passed to `correlated_gaussian` (“wrap” keeps periodic edges).
    """

    def __init__(
        self,
        shape: tuple[int, int],
        sigma: float | tuple[float, float],
        rng: np.random.Generator | None = None,
        wrap: bool = True,
    ):
        self.shape = shape
        self.sigma = sigma
        self.rng = np.random.default_rng() if rng is None else rng
        self.wrap = wrap

        # Initial fuel: U(0,1) field with spatial correlation
        self.array = gauss_to_uniform(
            correlated_gaussian(shape, sigma, rng=self.rng, wrap=wrap)
        )

    # -----------------------------------------------------------------
    # External code can read/write `fuel.array` directly, e.g.
    # fuel.array[burn_mask] = 0.0
    # -----------------------------------------------------------------

    def update(self, magnitude: float) -> np.ndarray:
        """
        Add a non-negative, spatially correlated growth field.

        Parameters
        ----------
        magnitude : float
            Maximum possible increment at any pixel (scales the noise).

        Returns
        -------
        np.ndarray
            The updated fuel array (same object as `self.array`).
        """
        # Correlated noise → uniform [0,1] → scale to [0, magnitude]
        growth = gauss_to_uniform(
            correlated_gaussian(self.shape, self.sigma, rng=self.rng, wrap=self.wrap)
        )   * magnitude

        # Only positive growth; clamp resulting fuel to ≤ 1
        self.array += growth
        np.clip(self.array, 0.0, 1.0, out=self.array)
        return self.array

    def reset(self) -> np.ndarray:
        """Re-randomise the entire fuel field (fresh season, say)."""
        self.array[:] = gauss_to_uniform(
            correlated_gaussian(self.shape, self.sigma, rng=self.rng, wrap=self.wrap)
        ) 
        return self.array
