#!/usr/bin/env python3
"""
Plot default PDFs, CDFs, and hazard functions for our 3 distributions
(HLH, Weibull, InverseGauss) on a 3x3 grid and save the image.

Usage:
    python tools/plot_distributions.py \
            --out mtbs_fire_analysis/analysis/outputs/\
distribution_comparison.png
    python tools/plot_distributions.py --show
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from mtbs_fire_analysis.analysis.hlh_dist import HalfLifeHazardDistribution
from mtbs_fire_analysis.analysis.scipy_dist import InverseGauss, Weibull


def _default_models():
    return [
        ("HalfLifeHazard", HalfLifeHazardDistribution()),
        ("Weibull", Weibull()),
        ("InverseGauss", InverseGauss()),
    ]


essential_cols = ("PDF", "CDF", "Hazard")


def _x_grid(models, base_points: int = 800) -> np.ndarray:
    means: List[float] = []
    for _, m in models:
        try:
            means.append(float(m.mean()))
        except Exception:
            means.append(np.nan)
    means = [v for v in means if np.isfinite(v)]
    max_mean = max(means) if means else 1.0
    x_max = max(5.0 * max_mean, 5.0)
    return np.linspace(1e-9, x_max, base_points)


def plot_distribution_overview(
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path:
    models = _default_models()
    x = _x_grid(models)

    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(12, 9), constrained_layout=True
    )

    for j, title in enumerate(essential_cols):
        axes[0, j].set_title(title)

    for i, (name, model) in enumerate(models):
        params = getattr(model, "params", {}) or {}
        if name == "HalfLifeHazard":
            label = (
                "HLH (h_inf="
                f"{params.get('hazard_inf', np.nan):.3g}, "
                "tau="
                f"{params.get('half_life', np.nan):.3g})"
            )
        elif name == "Weibull":
            label = (
                "Weibull (k="
                f"{params.get('shape', np.nan):.3g}, "
                "lam="
                f"{params.get('scale', np.nan):.3g})"
            )
        else:
            label = (
                "InverseGauss (mu="
                f"{params.get('mu', np.nan):.3g}, "
                "lam="
                f"{params.get('lam', np.nan):.3g})"
            )
        axes[i, 0].set_ylabel(label)

        with np.errstate(
            over="ignore", under="ignore", invalid="ignore", divide="ignore"
        ):
            pdf = model.pdf(x)
            sf = model.sf(x)
            cdf = 1.0 - sf
            haz = model.hazard(x)

        ax = axes[i, 0]
        ax.plot(x, np.maximum(pdf, 0.0), color="C0", lw=1.8)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.plot(x, np.clip(cdf, 0.0, 1.0), color="C1", lw=1.8)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 2]
        finite_h = haz[np.isfinite(haz)]
        clip_hi = (
            float(np.nanpercentile(finite_h, 99.5))
            if finite_h.size
            else 1.0
        )
        haz_plot = np.clip(haz, 0.0, clip_hi)
        ax.plot(x, haz_plot, color="C2", lw=1.8)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("x (time units)")

    fig.suptitle(
        "Distribution comparison at constructor defaults",
        fontsize=14,
    )

    if save_path is None:
        out_dir = (
            Path(__file__).resolve().parents[1]
            / "mtbs_fire_analysis"
            / "analysis"
            / "outputs"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "distribution_comparison.png"
    save_path = Path(save_path)

    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate a 3x3 PDF/CDF/Hazard comparison for default "
            "distributions."
        )
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output image path (.png/.pdf). Defaults to "
            "analysis/outputs/distribution_comparison.png"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving.",
    )
    args = parser.parse_args()

    path = plot_distribution_overview(args.out, show=args.show)
    print(f"Saved distribution comparison to: {path}")
