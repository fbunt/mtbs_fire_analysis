#!/usr/bin/env python3
"""
Utility to print current distribution constraints and defaults.

Usage:
  - python -m mtbs_fire_analysis.tools.print_constraints [--json]
  - or run this file directly.
"""
from __future__ import annotations

import argparse

from mtbs_fire_analysis.analysis.constraints_inspect import print_constraints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print distribution constraints and defaults"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable text",
    )
    args = parser.parse_args()
    print_constraints(json_output=args.json)


if __name__ == "__main__":
    main()
