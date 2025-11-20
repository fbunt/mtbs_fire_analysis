from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from mtbs_fire_analysis.analysis.statistical_tests.defaults import (
    HLH_DEF,
    IG_DEF,
    WEIBULL_DEF,
)

# Number of samples per distribution for the overview
N_SAMPLES = 200_000
PCTS = [0.1, 1, 5, 20, 80, 95, 99, 99.9]


def _summarize_samples(name: str, x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    # Basic stats
    mean = float(np.mean(x))
    med = float(np.median(x))
    std = float(np.std(x))
    sk = float(skew(x, bias=False))
    kurt = float(kurtosis(x, fisher=False, bias=False))
    # Percentiles
    q = np.percentile(x, PCTS, method="linear")
    pct_cols = {f"p{p:g}": float(v) for p, v in zip(PCTS, q, strict=True)}

    return {
        "dist": name,
        "mean": mean,
        "median": med,
        "std_dev": std,
        "skew": sk,
        "kurtosis": kurt,
        **pct_cols,
    }


essential_cols = (
    ["dist", "mean", "median", "std_dev", "skew", "kurtosis"]
    + [f"p{p:g}" for p in PCTS]
)


def main(seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Build models from defaults
    hlh = HLH_DEF.build()
    wb = WEIBULL_DEF.build()
    ig = IG_DEF.build()

    # Sample
    out = []
    out.append(_summarize_samples("HLH", hlh.rvs(size=N_SAMPLES, rng=rng)))
    out.append(
        _summarize_samples("Weibull", wb.rvs(size=N_SAMPLES, rng=rng))
    )
    out.append(
        _summarize_samples("InverseGauss", ig.rvs(size=N_SAMPLES, rng=rng))
    )

    overview = pd.DataFrame(out, columns=essential_cols)
    return overview


if __name__ == "__main__":  # pragma: no cover
    overview = main()
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print(overview.to_string(index=False))
