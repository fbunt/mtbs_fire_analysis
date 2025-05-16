import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import lambertw
from scipy.stats import weibull_min


def fit_dist(data, nll_func, survival_data=None, initial_params=None):
    """
    Fit a distribution to the data using maximum likelihood estimation.

    Parameters
    ----------
    data          : 1‑D array‑like
        Observed data.
    nll_func      : callable
        Function to compute the negative log likelihood.
    survival_data : 1‑D array‑like or None
        Optional survival data. If provided, the log likelihood is computed
        using both the observed and survival data.

    Returns
    -------
    tuple
        Fitted parameters (shape, scale).
    """
    if initial_params is None:
        # Initial guess for parameters
        initial_params = (1.0, 1.0)

    # Minimize the negative log likelihood
    result = minimize(
        nll_func,
        initial_params,
        args=(data, survival_data),
        method="L-BFGS-B",
        bounds=((1e-5, None), (1e-5, None)),
    )

    if result.success:
        return result.x
    else:
        raise RuntimeError("Parameter fitting failed.")


class WeibullDistribution:
    """
    Weibull distribution class.

    Parameters
    ----------
    shape : float
        Shape parameter of the Weibull distribution.
    scale : float
        Scale parameter of the Weibull distribution.
    """

    def __init__(self, shape=1, scale=1):
        self.shape = shape
        self.scale = scale

    @property
    def params(self):
        """Return the parameters of the distribution as dictionary."""
        return {"shape": self.shape, "scale": self.scale}

    def pdf(self, x):
        return weibull_min.pdf(x, self.shape, scale=self.scale)

    def cdf(self, x):
        return weibull_min.cdf(x, self.shape, scale=self.scale)

    def sf(self, x):
        return weibull_min.sf(x, self.shape, scale=self.scale)

    def hazard(self, x):
        """
        Compute the hazard function of the Weibull distribution.

        Parameters
        ----------
        x : 1‑D array‑like
            Points at which to evaluate the hazard function.

        Returns
        -------
        1‑D array
            Hazard function values.
        """
        return (self.shape / self.scale) * (x / self.scale) ** (self.shape - 1)

    def gen_neg_log_likelihood(self, params, data, survival_data=None):
        """
        Compute the log likelihood of the Weibull distribution given the data,
        survival data, and the parameters.

        Parameters
        ----------
        data   : 1‑D array‑like
            Observed data.
        survival_data : 1‑D array‑like or None
            Optional survival data. If provided, the log likelihood is computed
            using both the observed and survival data.

        Returns
        -------
        float
            Log likelihood value.
        """
        shape, scale = params
        if shape <= 0 or scale <= 0:
            return np.inf  # Invalid parameters

        # Log likelihood for observed data
        log_likelihood = np.sum(weibull_min.logpdf(data, shape, scale=scale))

        # Add log likelihood for survival data if provided
        if survival_data is not None:
            log_likelihood += np.sum(
                weibull_min.logsf(survival_data, shape, scale=scale)
            )

        return -log_likelihood

    def neg_log_likelihood(self, data, survival_data=None):
        """
        Compute the log likelihood of the Weibull distribution given the data,
        survival data, and the parameters.

        Parameters
        ----------
        data   : 1‑D array‑like
            Observed data.
        survival_data : 1‑D array‑like or None
            Optional survival data. If provided, the log likelihood is computed
            using both the observed and survival data.

        Returns
        -------
        float
            Log likelihood value.
        """
        return self.gen_neg_log_likelihood(
            (self.shape, self.scale), data, survival_data
        )

    def fit(self, data, survival_data=None):
        """
        Fit the Weibull distribution to the data using
        maximum likelihood estimation.

        Parameters
        ----------
        data          : 1‑D array‑like
            Observed data.
        survival_data : 1‑D array‑like or None
            Optional survival data. If provided, the log likelihood is computed
            using both the observed and survival data.

        Returns
        -------
        tuple
            Fitted parameters (shape, scale).
        """
        # Fit the Weibull distribution to the data
        params = fit_dist(
            data,
            self.gen_neg_log_likelihood,
            survival_data,
            initial_params=(self.shape, self.scale),
        )
        self.shape, self.scale = params


class HalfLifeHazardDistribution:
    """
    Hazard grows from 0 toward h_inf with exponential “half-life” approach.

    Parameters
    ----------
    hazard_inf : float
        Asymptotic hazard  h(∞)  (>0).
    half_life  : float
        Time τ½ where the gap  h_inf − h(t)  has halved  (>0).
    """

    # ---------- construction ----------------------------------------------
    def __init__(self, hazard_inf: float = 0.1, half_life: float = 10.0):
        if hazard_inf <= 0 or half_life <= 0:
            raise ValueError("hazard_inf and half_life must be positive.")
        self.hazard_inf = float(hazard_inf)
        self.lam = np.log(2.0) / float(half_life)  # λ = ln 2 / τ½

    @property
    def params(self):
        """Return the parameters of the distribution as dictionary."""
        return {"hazard_inf": self.hazard_inf, "half_life": self.half_life}

    @property
    def dist_type(self):
        """Return the distribution type."""
        return "HalfLifeHazard"

    # ---------- derived attributes ----------------------------------------
    @property
    def half_life(self) -> float:
        """Return τ½  (== ln 2 / λ)."""
        return np.log(2.0) / self.lam

    # ---------- core maths -------------------------------------------------
    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        return self.hazard_inf * (1.0 - np.exp(-self.lam * t))

    def cum_hazard(self, t):
        t = np.asarray(t, dtype=float)
        return self.hazard_inf * (t - (1.0 - np.exp(-self.lam * t)) / self.lam)

    def survival(self, t):
        return np.exp(-self.cum_hazard(t))

    def pdf(self, t):
        return self.hazard(t) * self.survival(t)

    def cdf(self, t):
        return 1.0 - self.survival(t)

    # ---------- log-densities (handy for fitting) --------------------------
    def log_pdf(self, t):
        t = np.maximum(t, 1e-12)  # avoid log(0) at t=0
        return np.log(self.hazard(t)) - self.cum_hazard(t)

    # ---------- pure NLL for optimisation ---------------------------------
    @staticmethod
    def _neg_log_likelihood(params, data, cens):
        """Pure function: NLL(h_inf, τ½ | data [, right-censored])."""
        h_inf, half_life = params
        if h_inf <= 0 or half_life <= 0:
            return np.inf  # outside domain

        lam = np.log(2.0) / half_life

        # helper closures (keep vectorised)
        def H(x):
            return h_inf * (x - (1.0 - np.exp(-lam * x)) / lam)

        log_pdf = np.log(h_inf * (1.0 - np.exp(-lam * data))) - H(data)
        nll = -np.sum(log_pdf)

        if cens is not None:
            nll += np.sum(H(cens))  # −log S(t) = H(t)

        return nll

    # ---------- log-likelihood (for testing) ------------------------------
    def neg_log_likelihood(self, data, cens=None):
        """
        Compute the log likelihood of the Half-Life Hazard distribution
        given the data and optional right-censored data.

        Parameters
        ----------
        data   : 1‑D array‑like
            Observed data.
        cens   : 1‑D array‑like or None
            Optional right-censored data. If provided, the log likelihood is
            computed using both the observed and censored data.

        Returns
        -------
        float
            Log likelihood value.
        """
        return self._neg_log_likelihood(
            (self.hazard_inf, self.half_life), data, cens
        )

    # ---------- MLE fit ----------------------------------------------------
    def fit(self, data, survival_data=None):
        """
        Maximum-likelihood fit.

        Parameters
        ----------
        data : 1-D array-like
            Observed failure times.
        survival_data : 1-D array-like, optional
            Right-censored times (still surviving at last observation).

        Updates
        -------
        self.hazard_inf , self.lam   to the MLEs.
        """
        data = np.asarray(data, dtype=float)
        cens = (
            None
            if survival_data is None
            else np.asarray(survival_data, dtype=float)
        )

        bounds = ((1e-12, None), (1e-12, None))  # keep feasibile
        init = (self.hazard_inf, self.half_life)

        res = minimize(
            self._neg_log_likelihood,
            x0=init,
            args=(data, cens),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter"}
            maxiter=100000,
            maxfun=100000,
        )

        if not res.success:
            raise RuntimeError(f"Parameter fitting failed: {res.message}")

        # ---- update object to MLEs ---------------------------------------
        self.hazard_inf, half_life_mle = res.x
        self.lam = np.log(2.0) / half_life_mle

        return {"hazard_inf": self.hazard_inf, "half_life": half_life_mle}

    # ---------- random variate generator (unchanged) ----------------------
    def rvs(self, size=1, rng=None):
        """Inverse-transform sampling using the closed-form Lambert-W."""
        rng = np.random.default_rng() if rng is None else rng
        u = rng.random(size)

        y = -np.log1p(-u) / self.hazard_inf  # −ln(1-U)/h_inf
        c = 1.0 + self.lam * y
        z = c + lambertw(-np.exp(-c), k=0).real  # principal branch
        return z / self.lam


def deficit(fit, data, ref: float = 0.0):
    """
    Compute the deficit of the fitted distribution.

    Parameters
    ----------
    fit : object
        Fitted distribution object.
    data : 1‑D array‑like
        Observed data.
    ref : float, optional
        Reference value for the deficit calculation.

    Returns
    -------
    float
        Deficit value.
    """
    ref_hazard = fit.hazard(ref)
    hazard = fit.hazard(data)
    return hazard - ref_hazard



def plot_fit(samples, dist_obj, output_name, bins=60, title=None):
    """
    Plot a sample distribution (histogram & ECDF) next to the fitted
    parametric PDF/CDF.

    Parameters
    ----------
    samples : 1‑D array‑like
        Observed data.
    pdf_fn  : callable
        Function pdf_fn(x, *params) returning the model PDF.
    cdf_fn  : callable
        Function cdf_fn(x, *params) returning the model CDF.
    params  : tuple/list
        Parameters to pass to pdf_fn / cdf_fn.
    bins    : int
        Number of histogram bins.
    title   : str or None
        Figure title.
    """
    samples = np.asarray(samples)
    x_plot = np.linspace(samples.min(), samples.max(), 500)

    # --- figure set‑up ------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    # -------- PDF panel -----------------------------------------------------
    axs[0].hist(
        samples,
        bins=bins,
        density=True,
        color="lightgrey",
        edgecolor="k",
        alpha=0.7,
        label="sample",
    )
    axs[0].plot(
        x_plot, dist_obj.pdf(x_plot), lw=2, color="crimson", label="fitted PDF"
    )
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("density")
    axs[0].set_title("Histogram vs fitted PDF")
    axs[0].legend()

    # -------- CDF panel -----------------------------------------------------
    sorted_x = np.sort(samples)
    ecdf = np.arange(1, len(sorted_x) + 1) / len(sorted_x)
    axs[1].step(
        sorted_x, ecdf, where="post", color="black", label="empirical CDF"
    )
    axs[1].plot(
        x_plot, dist_obj.cdf(x_plot), lw=2, color="crimson", label="fitted CDF"
    )
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("probability")
    axs[1].set_title("ECDF vs fitted CDF")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(output_name)
    plt.close(fig)
