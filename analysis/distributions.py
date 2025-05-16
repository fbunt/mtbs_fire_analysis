import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import lambertw
from scipy.stats import weibull_min

LN2 = np.log(2.0)

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
    Hazard grows from 0 toward h_inf with an exponential half-life approach.

    Parameters
    ----------
    hazard_inf : float   ( > 0 )
        Limit h(t→∞).
    half_life  : float   ( > 0 )
        Time at which the gap h_inf − h(t) has halved.
    """

    # ----- construction ----------------------------------------------------
    def __init__(self, hazard_inf: float = 0.1, half_life: float = 10.0):
        if hazard_inf <= 0 or half_life <= 0:
            raise ValueError("hazard_inf and half_life must be positive.")
        self.hazard_inf = float(hazard_inf)
        self.lam = LN2 / float(half_life)           # λ = ln 2 / τ½

    # ----- handy read-outs -------------------------------------------------
    @property
    def half_life(self) -> float:                   # τ½  (derived)
        return LN2 / self.lam

    @property
    def dist_type(self):
        """Return the distribution type."""
        return "HalfLifeHazard"

    @property
    def params(self):
        return {"hazard_inf": self.hazard_inf, "half_life": self.half_life}

    # ----- core functions --------------------------------------------------
    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        return self.hazard_inf * (-np.expm1(-self.lam * t))      # 1-e^{−λt}

    def cum_hazard(self, t):
        t = np.asarray(t, dtype=float)
        e = np.exp(-self.lam * t)
        return self.hazard_inf * (t - (1.0 - e) / self.lam)

    def survival(self, t):
        return np.exp(-self.cum_hazard(t))

    def pdf(self, t):
        return self.hazard(t) * self.survival(t)

    def cdf(self, t):
        return 1.0 - self.survival(t)

    # ---------------------------------------------------------------------
    #   Negative log-likelihood  **and**  its analytic gradient
    # ---------------------------------------------------------------------
    @staticmethod
    def _nll_and_grad(params, data, cens):
        """
        Returns (nll, grad) for the given parameters.

        Parameters
        ----------
        params : (h_inf, half_life)
        data   : failure times  (1-D ndarray)
        cens   : right-censored times or None
        """
        h_inf, tau = params
        if h_inf <= 0 or tau <= 0:
            # Outside the domain → infinite objective, zero gradient (harmless).
            return np.inf, np.array([0.0, 0.0])

        lam = LN2 / tau
        # --- helpers that we need several times --------------------------------
        def A(x):                                      #   t − (1−e^{−λt})/λ
            e = np.exp(-lam * x)
            return x - (1.0 - e) / lam

        t_fail = data
        t_all  = t_fail if cens is None else np.concatenate([t_fail, cens])

        # *NLL* -----------------------------------------------------------------
        g_fail = -np.expm1(-lam * t_fail)              # 1 − e^{−λt}
        nll  = -len(t_fail) * np.log(h_inf)            # −Σ log h_inf
        nll -= np.sum(np.log(g_fail))                  # −Σ log g(t)
        nll += h_inf * np.sum(A(t_all))                # +Σ H(t)

        # *Gradient wrt h_inf* ---------------------------------------------------
        grad_h = -len(t_fail) / h_inf + np.sum(A(t_all))

        # *Gradient wrt tau* -----------------------------------------------------
        # pieces for ∂/∂λ
        e_fail = np.exp(-lam * t_fail)
        dg_dlam_fail = t_fail * e_fail                 # d/dλ  (1−e^{−λt})
        term1 = -np.sum(dg_dlam_fail / g_fail)         # from −Σ log g(t)

        e_all = np.exp(-lam * t_all)
        dA_dlam_all = ((1.0 - e_all) - t_all * e_all * lam) / lam**2
        term2 = h_inf * np.sum(dA_dlam_all)            # from Σ H(t)

        dlam_dtau = -lam / tau                         # λ = ln2 / τ

        grad_tau = (term1 + term2) * dlam_dtau
        grad = np.array([grad_h, grad_tau])

        return nll, grad

    def neg_log_likelihood(self, data, cens=None):
        """
        Compute the negative log likelihood of the data given the parameters.

        Parameters
        ----------
        data : 1‑D array‑like
            Observed data.
        cens : 1‑D array‑like or None
            Optional right-censored data. If provided, the log likelihood is
            computed using both the observed and censored data.

        Returns
        -------
        float
            Negative log likelihood value.
        """
        return self._nll_and_grad((self.hazard_inf, self.half_life), data, cens)[0]

    # ---------------------------------------------------------------------
    #   Maximum-likelihood fit
    # ---------------------------------------------------------------------
    def fit(self, data, survival_data=None, verbose=False):
        """
        Fit the parameters by maximising the likelihood (L-BFGS-B with Jacobian).

        Raises RuntimeError if the optimiser fails.
        """
        data = np.asarray(data, dtype=float)
        cens = None if survival_data is None else np.asarray(survival_data, float)

        def _obj(p, d, c):
            nll, _ = self._nll_and_grad(p, d, c)
            return nll

        def _jac(p, d, c):
            _, g = self._nll_and_grad(p, d, c)
            return g

        bounds = ((1e-12, 0.5), (1e-12, 100))
        init   = (self.hazard_inf, self.half_life)

        res = minimize(
            fun   = _obj,
            x0    = init,
            args  = (data, cens),
            method= "L-BFGS-B",
            jac   = _jac,
            bounds= bounds,
            options={"disp": verbose},
        )

        if not res.success:
            raise RuntimeError(
                f"Parameter fitting failed: {res.message} (status {res.status})"
            )

        self.hazard_inf, tau_hat = res.x
        self.lam = LN2 / tau_hat
        return {"hazard_inf": self.hazard_inf, "half_life": tau_hat}

    # ---------------------------------------------------------------------
    #   Quick diagnostics when optimise() returns “ABNORMAL”
    # ---------------------------------------------------------------------
    @staticmethod
    def diagnose_fit(data, cens=None, init=(0.1, 10.0), grid=8):
        """
        Print a coarse NLL landscape around the initial guess to detect
        multi-modality / flat regions / numeric blow-ups.
        """
        data = np.asarray(data, float)
        cens = None if cens is None else np.asarray(cens, float)

        h0, t0 = init
        h_vals = np.logspace(np.log10(h0) - 1, np.log10(h0) + 1, grid)
        t_vals = np.logspace(np.log10(t0) - 1, np.log10(t0) + 1, grid)
        Z = np.empty((grid, grid))
        for i, h in enumerate(h_vals):
            for j, tau in enumerate(t_vals):
                Z[i, j], _ = HalfLifeHazardDistribution._nll_and_grad(
                    (h, tau), data, cens
                )
        print("\nCoarse NLL heat-map (rows = hazard_inf, cols = half_life):")
        with np.printoptions(precision=1, suppress=True):
            print(Z)

    # ---------------------------------------------------------------------
    #   Random variate generator
    # ---------------------------------------------------------------------
    def rvs(self, size=1, rng=None):
        """Inverse-transform sampling using the closed-form Lambert-W."""
        rng = np.random.default_rng() if rng is None else rng
        u   = rng.random(size)

        y = -np.log1p(-u) / self.hazard_inf
        c = 1.0 + self.lam * y
        z = c + lambertw(-np.exp(-c), k=0).real
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



def plot_fit(dist_obj, samples, cens=None,
            output_name="DistributionTestOutput",
            title=None):
    """
    Plot a sample distribution (histogram & ECDF) next to the fitted
    parametric PDF/CDF.

    Parameters
    ----------
    dist_obj : object
        Fitted distribution object.
    samples : 1‑D array‑like
        Observed data.
    cens    : 1‑D array‑like or None
        Optional right-censored data. If provided, a survival function
        is plotted as well as the pdf/cdf.
    params  : tuple/list
        Parameters to pass to pdf_fn / cdf_fn.
    bins    : int
        Number of histogram bins.
    title   : str or None
        Figure title.
    """
    samples = np.asarray(samples)
    if cens is not None:
        cens = np.asarray(cens)

    min_x = int(np.floor(samples.min()))
    max_x = int(np.ceil(samples.max()))
    x_plot = np.linspace(min_x, max_x, 500)
    bins = np.arange(min_x, max_x + 1, 1)
    
    # --- figure set‑up ------------------------------------------------------
    num_plots = 2 + (1 if cens is not None else 0)
    fig, axs = plt.subplots(1, num_plots, figsize=(10, 4))
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

    if cens is not None:
        # -------- Survival panel ---------------------------------------------
        axs[2].hist(
            cens,
            bins=bins,
            density=True,
            color="lightgrey",
            edgecolor="k",
            alpha=0.7,
            label="sample",
        )
        axs[2].plot(
            x_plot, dist_obj.survival(x_plot), lw=2, color="crimson",
            label="fitted survival"
        )
        axs[2].set_xlabel("t")
        axs[2].set_ylabel("density")
        axs[2].set_title("Histogram vs fitted survival")
        axs[2].legend()

    plt.tight_layout()
    plt.savefig(output_name)
    plt.close(fig)
