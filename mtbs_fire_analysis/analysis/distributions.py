import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, approx_fprime
from scipy.special import lambertw
from scipy.stats import weibull_min
from matplotlib.offsetbox import AnchoredText

LN2 = np.log(2.0)
_EPS = 1e-12          # numeric safety for logs/divisions

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
    """

    # ---------- construction ---------------------------------------------
    def __init__(self, hazard_inf: float = 0.1, half_life: float = 10.0):
        if hazard_inf <= 0 or half_life <= 0:
            raise ValueError("hazard_inf and half_life must be positive.")
        self.hazard_inf = float(hazard_inf)
        self.lam        = LN2 / float(half_life)          # λ = ln 2 / τ½

    # ---------- handy read-outs ------------------------------------------
    @property
    def dist_type(self) -> str:
        """Short name used by plotting / reporting helpers."""
        return "HalfLifeHazard"

    @property
    def half_life(self) -> float:
        return LN2 / self.lam

    @property
    def params(self):
        return {"hazard_inf": self.hazard_inf, "half_life": self.half_life}

    # ---------- core maths -----------------------------------------------
    def hazard(self, t):
        t = np.asarray(t, float)
        return self.hazard_inf * (-np.expm1(-self.lam * t))       # 1-e^{−λt}

    def cum_hazard(self, t):
        t = np.asarray(t, float)
        e = np.exp(-self.lam * t)
        return self.hazard_inf * (t - (1.0 - e) / self.lam)

    def survival(self, t):
        return np.exp(-self.cum_hazard(t))

    def pdf(self, t):
        return self.hazard(t) * self.survival(t)

    def cdf(self, t):
        return 1.0 - self.survival(t)

    # -------------------------------------------------------------------
    #  NLL + gradient in ORIGINAL params  p = (h_inf , tau)
    # -------------------------------------------------------------------
    @staticmethod
    def _nll_and_grad_orig(params, data, cens):
        h_inf, tau = params
        if h_inf <= 0 or tau <= 0:
            return np.inf, np.array([0.0, 0.0])

        lam = LN2 / tau
        t_fail = np.maximum(data, _EPS)                # guard log(0)

        # helper A(t) = t − (1−e^{−λt})/λ
        e_fail  = np.exp(-lam * t_fail)
        A_fail  = t_fail - (1.0 - e_fail) / lam

        if cens is not None and len(cens):
            t_cens = np.maximum(cens, _EPS)
            e_cens = np.exp(-lam * t_cens)
            A_cens = t_cens - (1.0 - e_cens) / lam
            A_all  = np.concatenate([A_fail, A_cens])
        else:
            A_all = A_fail

        g_fail   = 1.0 - e_fail                       # 1−e^{−λt}

        # ---------- NLL ---------------------------------------------------
        nll  = -len(t_fail) * np.log(h_inf)
        nll -= np.sum(np.log(np.maximum(g_fail, _EPS)))
        nll += h_inf * np.sum(A_all)

        # ---------- gradient wrt h_inf -----------------------------------
        grad_h = -len(t_fail) / h_inf + np.sum(A_all)

        # ---------- gradient wrt tau -------------------------------------
        dg_dlam_fail  = t_fail * e_fail
        term1 = -np.sum(dg_dlam_fail / np.maximum(g_fail, _EPS))

        # dA/dλ
        e_all = np.exp(-lam * np.maximum(np.concatenate([t_fail, cens]) if cens is not None else t_fail, _EPS))
        x_all = np.concatenate([t_fail, cens]) if cens is not None else t_fail
        dA_dlam_all = ((1.0 - e_all) - x_all * e_all * lam) / (lam**2)
        term2 = h_inf * np.sum(dA_dlam_all)

        grad_tau = (term1 + term2) * (-lam / tau)      # chain rule λ→τ
        return nll, np.array([grad_h, grad_tau])

    # -------------------------------------------------------------------
    #  NLL + gradient in LOG params  θ = (log h_inf , log τ)
    # -------------------------------------------------------------------
    @classmethod
    def _nll_and_grad_log(cls, theta, data, cens):
        h_inf, tau = np.exp(theta)
        nll, g_orig = cls._nll_and_grad_orig((h_inf, tau), data, cens)
        grad_log = g_orig * np.array([h_inf, tau])     # ∂p/∂θ = p
        return nll, grad_log

    # -------------------------------------------------------------------
    #  Public NLL helper (original scale)
    # -------------------------------------------------------------------
    def neg_log_likelihood(self, data, cens=None):
        return self._nll_and_grad_orig(
            (self.hazard_inf, self.half_life), data, cens
        )[0]

    # -------------------------------------------------------------------
    #  Gradient consistency check (finite diff vs analytic)
    # -------------------------------------------------------------------
    @classmethod
    def _check_grad_consistency(cls, theta0, data, cens, tol=1e-4):
        """
        Return True if analytic and finite-difference grads agree at theta0.
        """
        nll, g_analytic = cls._nll_and_grad_log(theta0, data, cens)

        def _obj(t):  # wrapper for approx_fprime
            return cls._nll_and_grad_log(t, data, cens)[0]

        g_fd = approx_fprime(theta0, _obj, np.sqrt(np.finfo(float).eps))
        ok = np.allclose(g_analytic, g_fd, rtol=tol, atol=tol)
        if not ok:
            print("\n[HalfLifeHazard] WARNING: analytic gradient mismatch "
                  f"(max diff {np.max(np.abs(g_analytic - g_fd)):.2e}). "
                  "Falling back to gradient-free optimisation.")
        return ok

    # -------------------------------------------------------------------
    #  Maximum-likelihood fit (robust)
    # -------------------------------------------------------------------
    def fit(self, data, survival_data=None, verbose=False):
        data = np.asarray(data, float)
        cens = None if survival_data is None else np.asarray(survival_data, float)

        theta0 = np.log([self.hazard_inf, self.half_life])
        log_bounds = ((np.log(1e-12), np.log(4.0)),
                      (np.log(1e-12), np.log(1000.0)))

        use_grad = self._check_grad_consistency(theta0, data, cens)

        def _obj(theta, d, c):
            return self._nll_and_grad_log(theta, d, c)[0]

        if use_grad:
            def _jac(theta, d, c):
                return self._nll_and_grad_log(theta, d, c)[1]
        else:
            _jac = None

        res = minimize(
            fun     = _obj,
            x0      = theta0,
            args    = (data, cens),
            jac     = _jac,
            method  = "L-BFGS-B",
            bounds  = log_bounds,
            options = {"disp": verbose, "maxls": 40},
        )

        if not res.success:
            raise RuntimeError(f"Parameter fitting failed: {res.message}")

        self.hazard_inf, tau_hat = np.exp(res.x)
        self.lam = LN2 / tau_hat
        return {"hazard_inf": self.hazard_inf, "half_life": tau_hat}

    # -------------------------------------------------------------------
    #  Random variate generator (unchanged)
    # -------------------------------------------------------------------
    def rvs(self, size=1, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        u   = rng.random(size)
        y   = -np.log1p(-u) / self.hazard_inf
        c   = 1.0 + self.lam * y
        z   = c + lambertw(-np.exp(-c), k=0).real
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
        x_plot, dist_obj.pdf(x_plot), lw=2, color="crimson", label="fit PDF"
    )
    axs[0].set_ylabel("density")
    axs[0].set_title("Histogram vs fitted PDF")
    axs[0].legend()

    # -------- CDF panel -----------------------------------------------------
    sorted_x = np.sort(samples)
    ecdf = np.arange(1, len(sorted_x) + 1) / len(sorted_x)
    axs[1].step(
        sorted_x, ecdf, where="post", color="black", label="sample CDF"
    )
    axs[1].plot(
        x_plot, dist_obj.cdf(x_plot), lw=2, color="crimson", label="fit CDF"
    )
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
            label="last survival",
        )
        axs[2].set_ylabel("density")
        ax2 = axs[2].twinx()

        ax2.plot(
            x_plot, dist_obj.survival(x_plot), lw=2, color="crimson",
            label="fit survival"
        )

        # Combine legends from both axes
        lines, labels = axs[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper center")
        ax2.set_ylabel("survival probability", color="crimson")
        axs[2].set_title("Histogram vs fitted survival")

    fig.supxlabel("Years since last fire")

    # Add a legend for the parameters

    labels = [f"{k}: {v}" for k, v in dist_obj.params.items()]
    extra = fig.legend(
    handles=[plt.Line2D([], [], lw=0)]*len(labels),  # dummy handles
    labels=labels,
    loc="upper center",  bbox_to_anchor=(0.5, 0.0),
    frameon=True, title="Parameter values",
    ncol=3, handlelength=0
    )
    plt.tight_layout()
    plt.savefig(output_name,bbox_extra_artists=[extra], bbox_inches='tight')
    plt.close(fig)
