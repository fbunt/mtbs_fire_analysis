"""
Helpers to inspect and print centralised fit constraints and model params.

Provides programmatic and human-readable summaries for each registered
distribution: default params, soft-box bounds (natural scale), hard bounds
(log-parameter space, and converted back to natural scale), and quick health.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import registry
from .fit_constraints import POLICY, get_bounds_log_for, get_soft_box_for


def _natural_bounds_from_log(
    bounds_log: Sequence[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    return [(math.exp(lo), math.exp(hi)) for (lo, hi) in bounds_log]


def describe_constraints_for(
    name: str, *, init_params: Optional[Dict[str, Any]] = None
) -> Dict:
    """Return a structured description of constraints and params for a dist.

    Includes:
      - registry name and class name
      - params (from default-constructed model)
      - hard bounds (log space and natural space)
      - soft_box (natural space)
      - quick health issues
    """
    ctor = registry.REGISTRY[name]
    model = ctor(**(init_params or {}))
    cls_name = type(model).__name__
    params = dict(model.params)
    bounds_log = list(get_bounds_log_for(cls_name))
    bounds_nat = _natural_bounds_from_log(bounds_log)
    soft_box = dict(get_soft_box_for(cls_name))
    health = model.quick_health()
    return {
        "name": name,
        "class": cls_name,
        "params": params,
        "bounds_log": bounds_log,
        "bounds_natural": bounds_nat,
        "soft_box": soft_box,
        "health": health,
    }


def describe_all_constraints() -> List[Dict]:
    return [describe_constraints_for(n) for n in registry.list_distributions()]


def print_constraints(json_output: bool = False) -> None:
    """Print a readable summary of constraints for all registered dists."""
    items = describe_all_constraints()
    if json_output:
        print(json.dumps(items, indent=2, default=float))
        return
    # human-readable
    print("Policy:")
    print(
        "  penalty_scale_with_counts:",
        POLICY.get("penalty_scale_with_counts"),
    )
    print("  penalty_scale_exponent:", POLICY.get("penalty_scale_exponent"))
    print(
        "  fail_on_unhealthy_default:",
        POLICY.get("fail_on_unhealthy_default"),
    )
    print()
    for it in items:
        print(f"[{it['name']}] class={it['class']}")
        print("  params:")
        for k, v in it["params"].items():
            print(f"    - {k}: {v}")
        print("  soft_box (natural):")
        for k, (lo, hi) in it["soft_box"].items():
            print(f"    - {k}: [{lo}, {hi}]")
        # Print hard bounds with natural first, then log for easy comparison
        print("  hard bounds (natural):")
        for (lo, hi) in it["bounds_natural"]:
            print(f"    - [{lo}, {hi}]")
        print("  hard bounds (log space):")
        for (lo, hi) in it["bounds_log"]:
            print(f"    - [{lo}, {hi}]")
        health = it.get("health", {})
        ok = health.get("ok", False)
        issues = ", ".join(health.get("issues", []))
        print(f"  health: ok={ok}; issues=[{issues}]")
        print()
