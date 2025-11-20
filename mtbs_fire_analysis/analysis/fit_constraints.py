"""
Centralised fit bounds, soft boxes, and health/penalty settings
================================================================

This module centralises the configuration used by all lifetime models
(HalfLifeHazardDistribution and SciPy-backed ones like Weibull, InverseGauss)
for:
    - hard optimiser bounds (we KEEP them authored in NATURAL units for clarity
        and convert to log-space when the optimiser needs them)
    - soft penalty "practical boxes" (in natural parameter space)
    - penalty scaling policy vs. data size
    - post-fit health thresholds (near-bounds/soft-box proximity)

Downstream scripts can import and tweak these at runtime:

        from mtbs_fire_analysis.analysis.fit_constraints import CONSTRAINTS
        CONSTRAINTS["HalfLifeHazardDistribution"]["soft_box"][
            "hazard_inf"
        ] = (1e-4, 0.5)

or replace the whole class section. Keep values reasonable to avoid optimiser
pathologies.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, Sequence, Tuple


# ----------------------------------------------------------------------------
# Global constraints registry keyed by class name
# ----------------------------------------------------------------------------
CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    # Half‑Life Hazard Distribution (HLH)
    "HalfLifeHazardDistribution": {
        # Author hard bounds in NATURAL units; optimiser sees log-converted
        # Order: (h_inf, half_life)
        "bounds_natural": [
            (1e-12, 10.0),   # hazard_inf
            (1e-5, 1e5),     # half_life (same time-units as data)
        ],
        # Soft practical box in natural parameter space
        "soft_box": {
            "hazard_inf": (1e-5, 5.0),
            "half_life": (0.05, 2000.0),  # in same units as data (years)
        },
        # Base strength of penalty (scaled by model policy below)
        "penalty_strength": 1e3,
    },

    # Weibull(k, lambda) wrapper
    "Weibull": {
        "bounds_natural": [
            (1e-3, 1e3),   # shape k
            (1e-3, 1e5),   # scale lambda
        ],
        "soft_box": {
            "shape": (0.3, 10.0),
            "scale": (1.0, 200.0),
        },
        "penalty_strength": 1e3,
    },

    # Inverse Gaussian (mu, lam)
    "InverseGauss": {
        "bounds_natural": [
            (1e-3, 1e4),  # mu
            (1e-3, 1e5),  # lam
        ],
        "soft_box": {
            "mu": (1, 1e3),
            "lam": (1e-2, 1e4),
        },
        "penalty_strength": 1e3,
    },
}


# ----------------------------------------------------------------------------
# Global penalty and health policies (can be tweaked by callers)
# ----------------------------------------------------------------------------
POLICY: Dict[str, Any] = {
    # Whether to scale soft penalty with effective data size.
    # Enabled by default per user preference; we scale by the SUM OF COUNTS
    # only (ignore raw lengths when counts are absent) and use sqrt scaling.
    "penalty_scale_with_counts": True,
    # Exponent when scaling with counts: scale = (N_eff) ** exponent
    # Using 0.5 (sqrt) to avoid overly aggressive penalties.
    "penalty_scale_exponent": 0.5,

    # Health/failure policy
    # Treat fits outside soft box as unhealthy
    "fail_if_outside_soft_box": True,
    # Also treat fits "too close" to a soft-box edge as unhealthy
    "fail_if_near_soft_edge": True,
    # Proximity threshold as fraction of box width (e.g., 0.05 → within 5%)
    "soft_edge_margin_fraction": 0.05,

    # Hard bounds proximity health: consider unhealthy if theta is within
    # this fraction of bound interval (in log-space) from either edge.
    "fail_if_near_hard_bounds": True,
    "hard_bounds_margin_fraction": 0.02,

    # After optimise, if unhealthy and this is True, raise an error.
    # Callers can override per-fit: fit(..., fail_on_unhealthy=False)
    "fail_on_unhealthy_default": False,
}


# ----------------------------------------------------------------------------
# Helper accessors
# ----------------------------------------------------------------------------

def get_bounds_log_for(class_name: str) -> Sequence[Tuple[float, float]]:
    """Return hard bounds in LOG space for the optimiser.

    We support two authoring styles in CONSTRAINTS for clarity:
      - bounds_natural: list of (lo, hi) in NATURAL units (preferred)
      - bounds_log:     list of (lo, hi) already in LOG space

    Only natural-log (base e) is used internally. Be careful not to assume
    base-10 logs when eyeballing values; e.g., ln(1e-12) ≈ −27.63.
    """
    cfg = CONSTRAINTS.get(class_name)
    if cfg is None:
        raise KeyError(f"Missing constraints for class {class_name}")
    if "bounds_log" in cfg:
        return cfg["bounds_log"]
    if "bounds_natural" in cfg:
        bl = []
        for lo, hi in cfg["bounds_natural"]:
            bl.append((math.log(lo), math.log(hi)))
        return bl
    raise KeyError(f"Missing bounds (natural/log) for class {class_name}")


def get_bounds_natural_for(class_name: str) -> Sequence[Tuple[float, float]]:
    """Return hard bounds in NATURAL units.

    If only log-bounds are authored, exponentiate to recover natural-scale
    ranges for display. This is mainly for inspector/printing tools.
    """
    cfg = CONSTRAINTS.get(class_name)
    if cfg is None:
        raise KeyError(f"Missing constraints for class {class_name}")
    if "bounds_natural" in cfg:
        return cfg["bounds_natural"]
    if "bounds_log" in cfg:
        bn = []
        for lo, hi in cfg["bounds_log"]:
            bn.append((float(np.exp(lo)), float(np.exp(hi))))
        return bn
    raise KeyError(f"Missing bounds (natural/log) for class {class_name}")


def get_soft_box_for(class_name: str) -> Dict[str, Tuple[float, float]]:
    cfg = CONSTRAINTS.get(class_name)
    if cfg is None or "soft_box" not in cfg:
        raise KeyError(f"Missing soft_box for class {class_name}")
    return cfg["soft_box"]


def get_penalty_strength_for(class_name: str) -> float:
    return float(CONSTRAINTS.get(class_name, {}).get("penalty_strength", 0.0))


def penalty_scale_factor(n_eff: float) -> float:
    if not POLICY.get("penalty_scale_with_counts", False):
        return 1.0
    exp = float(POLICY.get("penalty_scale_exponent", 1.0))
    n_eff = max(float(n_eff), 1.0)
    return float(n_eff ** exp)


def soft_box_distance(
    params: Dict[str, float], soft_box: Dict[str, Tuple[float, float]]
):
    """Return a tuple (inside_all, min_margin_frac).

    - inside_all: True if every param within [lo, hi]
    - min_margin_frac: for those within, the minimum fractional distance to
      the nearest edge across dims, where 0 at edge and 0.5 in the middle.
      If a param is outside, the margin is negative proportion to how far out
      relative to width (capped at -1.0).
    """
    inside = True
    min_margin = 1.0
    for name, (lo, hi) in soft_box.items():
        val = float(params.get(name, np.nan))
        if not np.isfinite(val):
            return False, -1.0
        width = max(hi - lo, 1e-12)
        if val < lo:
            inside = False
            # negative margin magnitude relative to width
            m = -min((lo - val) / width, 1.0)
        elif val > hi:
            inside = False
            m = -min((val - hi) / width, 1.0)
        else:
            # inside: distance to nearest edge normalised by width
            m = min((val - lo) / width, (hi - val) / width)
        min_margin = min(min_margin, m)
    return inside, float(min_margin)
