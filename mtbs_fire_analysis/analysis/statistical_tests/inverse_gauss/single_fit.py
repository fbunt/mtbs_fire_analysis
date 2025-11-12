"""Single-fit smoke test for Inverse Gaussian lifetime model.

Generates synthetic samples from IG(mu_true, lam_true), fits the model via
BaseLifetime.fit, and prints parameter recovery plus mean and simple
likelihood components for sanity.
"""

from __future__ import annotations

import numpy as np

from mtbs_fire_analysis.analysis.scipy_dist import InverseGauss


def run(seed: int = 42):
    rng = np.random.default_rng(seed)
    mu_true, lam_true = 75.0, 1.5
    n = 500
    dist_true = InverseGauss(mu_true, lam_true)
    samples = dist_true.rvs(size=n, rng=rng)

    # Introduce a few right-censored observations (simulate censor window)
    censor_time = 8.0
    observed = samples[samples <= censor_time]
    censored = samples[samples > censor_time]

    model = InverseGauss(2.0, 1.0)  # crude initial guess
    model.fit(data=observed, survival_data=censored)

    print("Inverse Gaussian single-fit smoke test")
    print("True params  : mu={:.4f} lam={:.4f}".format(mu_true, lam_true))
    print("Fitted params: mu={:.4f} lam={:.4f}".format(model.mu, model.lam))
    print("Mean true / fitted:", dist_true.mean(), model.mean())
    print("N events / censored:", observed.size, censored.size)
    print(
        "Neg log-likelihood (fitted):",
        model.neg_log_likelihood(data=observed, survival_data=censored),
    )


if __name__ == "__main__":  # pragma: no cover
    run()
