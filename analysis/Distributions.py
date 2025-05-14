import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, lognorm, norm, weibull_min
from scipy.optimize import minimize


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
    result = minimize(nll_func, initial_params,
                      args=(data, survival_data), method='L-BFGS-B',
                      bounds=((1e-5, None), (1e-5, None)))

    if result.success:
        return result.x
    else:
        raise RuntimeError("Parameter fitting failed.")



# Weibull distribution class, implements the PDF and CDF, and the log likelihood as above. Change all functions to take only named parameters, so wrap the scipy functions to achieve this. Fit function that updates the parameters.
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

    def pdf(self, x):
        return weibull_min.pdf(x, self.shape, scale=self.scale)

    def cdf(self, x):
        return weibull_min.cdf(x, self.shape, scale=self.scale)
    
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
            log_likelihood += np.sum(weibull_min.logsf(survival_data, shape, scale=scale))

        return -log_likelihood
        #return weibull_log_likelihood((self.shape, self.scale), data, survival_data)
    

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
        return self.gen_neg_log_likelihood((self.shape, self.scale), data, survival_data)
    
    def fit(self, data, survival_data=None):
        """
        Fit the Weibull distribution to the data using maximum likelihood estimation.

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
        params = fit_dist(data, self.gen_neg_log_likelihood, survival_data, initial_params=(self.shape, self.scale))
        self.shape, self.scale = params






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
    axs[0].hist(samples, bins=bins, density=True,
                color="lightgrey", edgecolor="k", alpha=0.7, label="sample")
    axs[0].plot(x_plot, dist_obj.pdf(x_plot), lw=2, color="crimson",
                label="fitted PDF")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("density")
    axs[0].set_title("Histogram vs fitted PDF")
    axs[0].legend()

    # -------- CDF panel -----------------------------------------------------
    sorted_x = np.sort(samples)
    ecdf = np.arange(1, len(sorted_x) + 1) / len(sorted_x)
    axs[1].step(sorted_x, ecdf, where="post",
                color="black", label="empirical CDF")
    axs[1].plot(x_plot, dist_obj.cdf(x_plot),
                lw=2, color="crimson", label="fitted CDF")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("probability")
    axs[1].set_title("ECDF vs fitted CDF")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(output_name)
    plt.close(fig)