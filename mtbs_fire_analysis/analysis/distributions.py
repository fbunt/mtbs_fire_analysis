import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, approx_fprime
from scipy.special import lambertw, gammainc, gammaln
from scipy.stats import weibull_min

LN2 = np.log(2.0)
_EPS = 1e-12          # numeric safety for logs/divisions

def fit_dist(nll_func, data, data_counts=None, survival_data=None,
             survival_counts=None, initial_params=None):
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
        args=(data, data_counts, survival_data, survival_counts),
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
        self.lam        = LN2 / float(half_life)       # λ = ln2 / τ½

    # ---------- metadata --------------------------------------------------
    @property
    def dist_type(self) -> str:
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

    # ---------- first moment --------------------------------------------
    def mean(self):
        """
        Return E[T] (finite for all h_inf>0, half_life>0).

        Numerically stable version:  compute in log-space

            E[T] = e^k / λ · k^(−k) · Γ(k) · P(k,k)
                 = 1/λ · exp( k − k·log(k) + log Γ(k) + log P(k,k) )

        where  k = h_inf / λ  and  P(k,k) = gammainc(k,k) (regularised).

        Using gammaln(k) avoids overflow in Γ(k);
        log P(k,k) is safe via np.log since 0 < P(k,k) ≤ 1.
        """

        k   = self.hazard_inf / self.lam
        # regularised lower incomplete gamma (0 < p <= 1)
        p   = gammainc(k, k)
        #  log E[T]  (all pieces in log-space)
        log_mean = (
            k                              #  k
            - k * np.log(k)                #  -k ln k
            + gammaln(k)                   #  ln Γ(k)
            + np.log(p)                    #  ln P(k,k)
            - np.log(self.lam)             #  -ln λ
        )
        return np.exp(log_mean)

    # -------------------------------------------------------------------
    #  NLL + gradient in ORIGINAL params  p = (h_inf , tau)
    # -------------------------------------------------------------------
    @staticmethod
    def _nll_and_grad_orig(params,
                           data,        data_counts=None,
                           cens=None,   cens_counts=None):
        """
        data_counts / cens_counts  = multiplicities for each time-point.
        If None, defaults to 1.0.
        """
        h_inf, tau = params
        if h_inf <= 0 or tau <= 0:
            return np.inf, np.array([0.0, 0.0])

        lam = LN2 / tau

        # ---------- failures ---------------------------------------------
        t_fail   = np.maximum(np.asarray(data, float), _EPS)
        w_fail   = np.ones_like(t_fail, float) if data_counts is None \
                                         else np.asarray(data_counts, float)
        e_fail   = np.exp(-lam * t_fail)
        g_fail   = 1.0 - e_fail                             # 1−e^{−λt}
        a_fail   = t_fail - (1.0 - e_fail) / lam

        # ---------- censored ---------------------------------------------
        if cens is not None and len(cens):
            t_cens = np.maximum(np.asarray(cens, float), _EPS)
            w_cens = np.ones_like(t_cens, float) if cens_counts is None \
                                            else np.asarray(cens_counts, float)
            e_cens = np.exp(-lam * t_cens)
            a_cens = t_cens - (1.0 - e_cens) / lam

            t_all  = np.concatenate([t_fail, t_cens])
            w_all  = np.concatenate([w_fail, w_cens])
            a_all  = np.concatenate([a_fail, a_cens])
        else:
            t_all, w_all, a_all = t_fail, w_fail, a_fail

        # ---------- NLL ---------------------------------------------------
        n_fail = np.sum(w_fail)

        nll  = -n_fail * np.log(h_inf)
        nll -= np.sum(w_fail * np.log(np.maximum(g_fail, _EPS)))
        nll += h_inf * np.sum(w_all * a_all)

        # ---------- gradient wrt h_inf -----------------------------------
        grad_h = -n_fail / h_inf + np.sum(w_all * a_all)

        # ---------- gradient wrt tau -------------------------------------
        dg_dlam_fail = t_fail * e_fail
        term1 = -np.sum(w_fail * dg_dlam_fail / np.maximum(g_fail, _EPS))

        # dA/dλ
        da_dlam_all = ((1.0 - np.exp(-lam * t_all)) - t_all
                       * np.exp(-lam * t_all) * lam) / lam**2
        term2 = h_inf * np.sum(w_all * da_dlam_all)

        grad_tau = (term1 + term2) * (-lam / tau)         # chain-rule
        return nll, np.array([grad_h, grad_tau])

    # -------------------------------------------------------------------
    #  NLL + gradient in LOG-params  θ = (log h_inf , log τ)
    # -------------------------------------------------------------------
    @classmethod
    def _nll_and_grad_log(cls, theta,
                          data, data_counts,
                          cens, cens_counts):
        h_inf, tau = np.exp(theta)
        nll, g_orig = cls._nll_and_grad_orig((h_inf, tau),
                                             data, data_counts,
                                             cens, cens_counts)
        grad_log = g_orig * np.array([h_inf, tau])
        return nll, grad_log

    # -------------------------------------------------------------------
    #  Public NLL helper (original scale)
    # -------------------------------------------------------------------
    def neg_log_likelihood(self,
                           data,        data_counts=None,
                           survival_data=None,   survival_counts=None):
        return self._nll_and_grad_orig((self.hazard_inf, self.half_life),
                                       data, data_counts,
                                       survival_data, survival_counts)[0]

    # -------------------------------------------------------------------
    #  Gradient consistency check
    # -------------------------------------------------------------------
    @classmethod
    def _check_grad_consistency(cls, theta0,
                                data, data_counts,
                                cens, cens_counts,
                                tol=1e-4):
        nll, g_a = cls._nll_and_grad_log(theta0,
                                         data, data_counts,
                                         cens, cens_counts)

        def _obj(t):
            return cls._nll_and_grad_log(t,
                                         data, data_counts,
                                         cens, cens_counts)[0]

        g_fd = approx_fprime(theta0, _obj, np.sqrt(np.finfo(float).eps))
        ok = np.allclose(g_a, g_fd, rtol=tol, atol=tol)
        if not ok:
            print("\n[HalfLife] WARNING: gradient mismatch "
                  f"(max diff {np.max(np.abs(g_a - g_fd)):.2e}); "
                  "switching to finite-diff optimisation.")
        return ok

    # -------------------------------------------------------------------
    #  Maximum-likelihood fit
    # -------------------------------------------------------------------
    def fit(self,
            data, data_counts=None,
            survival_data=None, survival_counts=None,
            verbose=False):
        """
        Parameters
        ----------
        data, data_counts               : failure times + multiplicities
        survival_data, survival_counts  : right-censor times + multiplicities
        """
        data = np.asarray(data, float)

        if survival_data is not None:
            survival_data = np.asarray(survival_data, float)

        theta0 = np.log([self.hazard_inf, self.half_life])
        log_bounds = ((np.log(1e-12), np.log(4.0)),
                      (np.log(1e-12), np.log(1e3)))

        use_grad = self._check_grad_consistency(theta0,
                                                data, data_counts,
                                                survival_data, survival_counts)

        def _obj(theta, d, dc, c, cc):
            return self._nll_and_grad_log(theta, d, dc, c, cc)[0]

        if use_grad:
            def _jac(theta, d, dc, c, cc):
                return self._nll_and_grad_log(theta, d, dc, c, cc)[1]
        else:
            _jac = None

        res = minimize(
            fun     = _obj,
            x0      = theta0,
            args    = (data, data_counts, survival_data, survival_counts),
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

    def reset_params(self, hazard_inf=None, half_life=None):
        """
        Reset the parameters of the distribution.

        Parameters
        ----------
        hazard_inf : float, optional
            New value for the hazard_inf parameter.
        half_life : float, optional
            New value for the half_life parameter.
        """
        if hazard_inf is not None:
            self.hazard_inf = hazard_inf
        if half_life is not None:
            self.lam = LN2 / half_life



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



def plot_fit_old(dist_obj, samples, cens=None,
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
    if cens is not None:
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    else:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
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
    # Vertical line at the mean
    mean = dist_obj.mean()
    axs[0].axvline(mean, color="blue", linestyle="--", label="Return Interval")
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
    # Vertical line at the mean
    axs[1].axvline(mean, color="blue", linestyle="--", label="Return Interval")

    axs[1].set_ylabel("probability")
    axs[1].set_title("ECDF vs fitted CDF")
    axs[1].legend()

    # -------- Hazard panel --------------------------------------------------
    axs[2].plot(
        x_plot, dist_obj.hazard(x_plot), lw=2, color="crimson", label="fit hazard"
    )
    axs[2].set_ylabel("hazard")
    axs[2].set_title("Fitted hazard function")
    # add assymptotic line at h_inf
    axs[2].axhline(dist_obj.hazard_inf, color="green", linestyle="--",
                   label="h_inf")
    axs[2].legend()

    if cens is not None:
        # -------- Survival panel ---------------------------------------------
        axs[3].hist(
            cens,
            bins=bins,
            density=True,
            color="lightgrey",
            edgecolor="k",
            alpha=0.7,
            label="last survival",
        )
        axs[3].set_ylabel("density")
        ax2 = axs[3].twinx()

        ax2.plot(
            x_plot, dist_obj.survival(x_plot), lw=2, color="crimson",
            label="fit survival"
        )

        # Combine legends from both axes
        lines, labels = axs[3].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper center")
        ax2.set_ylabel("survival probability", color="crimson")
        axs[3].set_title("Histogram vs fitted survival")

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


def _weighted_ecdf(x: np.ndarray, w: np.ndarray):
    """Return the x values and cumulative probabilities for a weighted ECDF."""
    # sort by x
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    total = cum_w[-1]
    return x_sorted, cum_w / total


def plot_fit(
    dist_obj,
    dts,
    dt_counts=None,
    survivals=None,
    survival_counts=None,
    output_name="DistributionTestOutput",
    title=None,
    max_dt=None,
):
    """Plot a sample distribution (histogram & ECDF) next to the fitted parametric functions.

    Parameters
    ----------
    dist_obj : object
        Fitted distribution object exposing pdf/cdf/hazard/survival/mean/params.
    dts : 1‑D array‑like
        Failure times ("dts").
    dt_counts : 1‑D array‑like or None, optional
        Multiplicities associated with *dts*. If **None**, each observation is
        assumed to occur once.
    survivals : 1‑D array‑like or None, optional
        Right‑censor times ("survival").
    survival_counts : 1‑D array‑like or None, optional
        Multiplicities associated with *survivals*.
    output_name : str, optional
        Path for the output image file.
    title : str or None, optional
        Figure title.
    max_dt : float or int or None, optional
        Extend the x‑axis (and fitted curves) to this value.  If *None*, the
        maximum of the supplied data is used.
    """

    dts = np.asarray(dts, float)
    if dts.ndim == 0:
        dts = dts[None]
    w_fail = (
        np.ones_like(dts, float) if dt_counts is None else np.asarray(dt_counts, float)
    )

    if survivals is not None and len(survivals):
        survivals = np.asarray(survivals, float)
        w_surv = (
            np.ones_like(survivals, float)
            if survival_counts is None
            else np.asarray(survival_counts, float)
        )
    else:
        survivals, w_surv = None, None

    # ------------------------------------------------------------------
    # Axis range & evaluation grid
    # ------------------------------------------------------------------
    min_x = int(np.floor(dts.min()))
    data_max = dts.max()
    if survivals is not None:
        data_max = max(data_max, survivals.max())
    max_x = max(max_dt, data_max) if max_dt is not None else data_max
    max_x = int(np.ceil(max_x))

    x_plot = np.linspace(min_x, max_x, 500)
    bins = np.arange(min_x, max_x + 1, 1)

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    n_rows = 4 if survivals is not None else 3
    fig, axs = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows))
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    # ------------------------------------------------------------------
    # PDF / Histogram panel
    # ------------------------------------------------------------------
    axs[0].hist(dts, bins=bins, weights=w_fail, density=True, color="lightgrey", edgecolor="k", alpha=0.7, label="FRIs")
    axs[0].plot(x_plot, dist_obj.pdf(x_plot), lw=2, color="crimson", label="fit PDF")
    mean_val = dist_obj.mean()
    axs[0].axvline(mean_val, color="blue", linestyle="--", label="Return Interval")
    axs[0].set_ylabel("density")
    axs[0].set_title("Histogram vs fitted PDF")
    axs[0].legend()

    # ------------------------------------------------------------------
    # CDF / ECDF panel
    # ------------------------------------------------------------------
    x_ecdf, y_ecdf = _weighted_ecdf(dts, w_fail)
    axs[1].step(x_ecdf, y_ecdf, where="post", color="black", label="FRI CDF")
    axs[1].plot(x_plot, dist_obj.cdf(x_plot), lw=2, color="crimson", label="fit CDF")
    axs[1].axvline(mean_val, color="blue", linestyle="--", label="Return Interval")
    axs[1].set_ylabel("probability")
    axs[1].set_title("ECDF vs fitted CDF")
    axs[1].legend()

    # ------------------------------------------------------------------
    # Hazard panel
    # ------------------------------------------------------------------
    axs[2].plot(x_plot, dist_obj.hazard(x_plot), lw=2, color="crimson", label="fit hazard")
    axs[2].set_ylabel("hazard")
    axs[2].set_title("Fitted hazard function")
    axs[2].legend()

    # ------------------------------------------------------------------
    # Survival panel (optional)
    # ------------------------------------------------------------------
    if survivals is not None:
        axs[3].hist(survivals, bins=bins, weights=w_surv, density=True, color="lightgrey", edgecolor="k", alpha=0.7, label="survivals")
        ax2 = axs[3].twinx()
        ax2.plot(x_plot, dist_obj.survival(x_plot), lw=2, color="crimson", label="fit survival")
        # merge legends
        lines, labels = axs[3].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper center")
        axs[3].set_ylabel("density")
        ax2.set_ylabel("survival probability", color="crimson")
        axs[3].set_title("Survivals vs fitted survival")

    # ------------------------------------------------------------------
    # Common x‑label & parameter box
    # ------------------------------------------------------------------
    fig.supxlabel("Years since last fire")

    param_labels = [f"{k}: {v:.4g}" for k, v in dist_obj.params.items()]
    extra = fig.legend(
        handles=[plt.Line2D([], [], lw=0)] * len(param_labels),
        labels=param_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
        title="Parameter values",
        ncol=len(param_labels),
        handlelength=0,
    )

    plt.tight_layout()
    plt.savefig(output_name, bbox_extra_artists=[extra], bbox_inches="tight")
    plt.close(fig)
