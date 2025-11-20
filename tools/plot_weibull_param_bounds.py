#!/usr/bin/env python3
"""
Illustrate soft and hard bounds for a Weibull parameter (log x-axis).

- X axis: parameter value (natural units)
- Y axis: soft-bound penalty contribution (same units as added to NLL)
- Vertical lines at hard bounds
- Shaded region for soft box (practical bounds)

Usage:
  python tools/plot_weibull_param_bounds.py --param scale
    python tools/plot_weibull_param_bounds.py --param shape \
            --out /tmp/weibull_shape_bounds.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple  # noqa: F401  (reserved for future extension)

import matplotlib.pyplot as plt
import numpy as np

from mtbs_fire_analysis.analysis.fit_constraints import (
    get_bounds_natural_for,
    get_penalty_strength_for,
    get_soft_box_for,
)


def _penalty_for_param(value: float, lo: float, hi: float) -> float:
    """Quadratic penalty in LOG space outside the soft box for one param.

    We mirror the model change: compute distances in s=ln(value) relative to
    [ln(lo), ln(hi)] and normalise by the log-width for scale-free symmetry.
    """
    if not np.isfinite(value) or value <= 0.0:
        return 1e9
    s = np.log(value)
    s_lo, s_hi = np.log(lo), np.log(hi)
    w = max(s_hi - s_lo, 1e-12)
    if s < s_lo:
        d = (s_lo - s) / w
        return float(d * d)
    if s > s_hi:
        d = (s - s_hi) / w
        return float(d * d)
    return 0.0


def plot_weibull_param_bounds(
    param: str = "scale",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path:
    if param not in ("shape", "scale"):
        raise ValueError("param must be 'shape' or 'scale'")

    cls = "Weibull"
    sb = get_soft_box_for(cls)
    lo_soft, hi_soft = sb[param]
    strength = get_penalty_strength_for(cls) or 1e3

    # Hard bounds: natural units
    bounds_nat = get_bounds_natural_for(cls)
    idx = 0 if param == "shape" else 1
    lo_hard, hi_hard = bounds_nat[idx]

    # X grid in LOG domain spanning hard bounds with a small multiplicative
    # margin. Ensure strictly positive endpoints for log scale.
    x_lo = max(lo_hard * 0.8, 1e-12)
    x_hi = hi_hard * 1.2
    x = np.geomspace(x_lo, x_hi, 2000)

    # Soft penalty contribution (dim-only) scaled by strength
    y = np.array([_penalty_for_param(v, lo_soft, hi_soft) for v in x])
    y_scaled = strength * y
    # For readability, clip very large values so both sides are visible
    finite = y_scaled[np.isfinite(y_scaled)]
    pos = finite[finite > 0]
    clip_hi = float(np.nanpercentile(pos, 99.5)) if pos.size else 1.0
    y_plot = np.clip(y_scaled, 0.0, clip_hi)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Shade soft-box region (natural values on a log-scaled x-axis)
    ax.axvspan(
        lo_soft, hi_soft, color="tab:green", alpha=0.1, label="soft box"
    )

    # Hard bounds as vertical lines
    ax.axvline(lo_hard, color="tab:red", linestyle="--", label="hard bounds")
    ax.axvline(hi_hard, color="tab:red", linestyle="--")

    # Penalty curve (clipped for visibility)
    ax.plot(
        x,
        y_plot,
        color="tab:blue",
        lw=2,
        label="soft penalty term (clipped)",
    )

    ax.set_xscale("log")
    ax.set_xlabel(f"Weibull {param} (log x)")
    ax.set_ylabel("soft-bound loss (added to NLL)")
    ax.set_title(
        f"Weibull {param}: soft penalty vs. value\n"
        f"soft=({lo_soft:g}, {hi_soft:g}), hard=({lo_hard:g}, {hi_hard:g}), "
        f"strength={strength:g}, clip≤{clip_hi:.3g}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Save path default
    if save_path is None:
        out_dir = (
            Path(__file__).resolve().parents[1]
            / "mtbs_fire_analysis"
            / "analysis"
            / "outputs"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"weibull_{param}_bounds.png"
    save_path = Path(save_path)

    fig.tight_layout()
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
            "Plot soft penalty and hard bounds for a Weibull parameter."
        )
    )
    parser.add_argument(
        "--param",
        choices=["shape", "scale"],
        default="scale",
        help="Which Weibull parameter to plot (default: scale)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output image path (.png/.pdf). Defaults to "
            "analysis/outputs/weibull_<param>_bounds.png"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving.",
    )
    args = parser.parse_args()

    path = plot_weibull_param_bounds(args.param, args.out, show=args.show)
    print(f"Saved Weibull {args.param} bounds plot to: {path}")
