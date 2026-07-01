"""m02: static modal-NLCD census builder.

Stacks every per-year NLCD raster in ``[YEAR_START, YEAR_END]`` and reduces to
the per-pixel temporal mode (``rts.general.local_stats(stack, "mode")``),
writing the static census ``nlcd_mode_1984_2022.tif`` beside the per-year
stack. ``local_stats`` masks the output to null *iff every band is null* and
maps nulls to NaN internally before ``nanmode_jit`` (smallest-value-wins
tie-break), so nodata is skipped automatically — the same semantics the
per-interval kernel uses.

m02 reduces whatever per-year tree it is pointed at (``FIRE_NLCD_SUBDIR``), at
that tree's native **code-space** (raw NLCD class codes vs land-cover
**group** ids) and **resolution**, and writes the census to
``FIRE_NLCD_MODE_SUBDIR or FIRE_NLCD_SUBDIR``
(``paths.NLCD_MODE_RASTER_PATH``). The substrate overhaul drives it three
ways: the vanilla 30 m raw-code census
(``cleaned``/``cog``), the group-space census at 30 m (``grouped``) or coarse
(``grouped_<N>m``, from ``build_coarse_covariates``) — see
``D-2026-06-04-nlcd-interval-mode-group-space``.

Write-isolation guard (``D-2026-06-04-input-layer-rebuild-isolation``, the
2026-05-01 incident class — a rebuild run clobbering a shared canonical
raster). The census co-locates with the per-year stack m02 reduces, so the
output tree MUST be the SAME tree as the **source** (``FIRE_NLCD_MODE_SUBDIR``
is a *reader-side* census-location hook and must not redirect m02's BUILD).
Exact-tree equality blocks every way m02 could overwrite a canonical census it
did not produce: wrong code-space / wrong resolution (``cog`` -> ``cog_120m``),
same-shape different-product (``cleaned`` C1V0 <-> ``cog`` C1V1), or an
unrecognized custom source redirected onto a canonical tree. The guard
validates the ACTUAL bound write target (``NLCD_MODE_RASTER_PATH``) against the
bound source dir (``NLCD_PATH``), so it can never diverge from the real write;
``FIRE_ALLOW_PROD_WRITE=1`` overrides (you almost never want this). A vanilla
build (``cleaned``→``cleaned``) inherits the source tree, so Fred's upstream
users are unaffected — this is a no-op for them.

Re-run safety: ``--overwrite`` deletes + rebuilds. Without it, a build into an
**isolated** tree (``grouped`` / ``*_<N>m`` — what the overhaul produces)
aborts loud (exit 4) on an existing census rather than silently skipping (the
``protected_raster_save_with_cleanup`` skip-if-exists footgun); a build into a
bare canonical 30 m tree (``cleaned``/``cog``/``raw``) keeps the legacy silent
skip-if-exists, byte-identical to Fred's m02.

Per-cutoff mode (``--per-year-ends``, Phase-2 Flavor-B,
``D-2026-06-29-zero-fire-label-leak-flavor-b``). Instead of the single
all-years census, emit one *leak-free* census per requested window end
``nlcd_mode_1984_<Y_END>.tif`` (the zero-fire label for scored year ``Y`` reads
the census frozen at ``Y_END = Y-1``, so it never sees land cover from years
``>= Y``). Each cutoff is built with the SAME
``rts.general.local_stats(stack, "mode")`` reduction over ``[1984, Y_END]``, so
a per-cutoff census is byte-consistent with the all-years census at its shared
endpoint (``Y_END = 2022`` reproduces ``nlcd_mode_1984_2022.tif`` exactly). All
targets co-locate in the SAME guarded census tree as the all-years census, so
the write-isolation guard below validates them all in one shot (it keys on the
tree, not the filename). Without ``--overwrite``, an existing target aborts the
whole run up front (exit 4) before anything is written or deleted (no partial
set); with ``--overwrite`` each stale census is unlinked immediately before its
own rebuild, so a crash never widens the gap beyond the in-flight cutoff.

This guard is replicated locally (no import of ``fire_interval``): the
dependency only flows ``fire_interval -> mtbs_fire_analysis``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import raster_tools as rts

from mtbs_fire_analysis.pipeline.paths import (
    NLCD_MODE_RASTER_PATH,
    NLCD_PATH,
    NLCD_STACK_VRT_PATH,
    get_nlcd_mode_raster_path,
    get_nlcd_raster_path,
)
from mtbs_fire_analysis.utils import (
    protected_raster_save_with_cleanup,
    stack_rasters_as_vrt,
)

YEAR_START = 1984
YEAR_END = 2022  # inclusive — census observation window (W_0..W_1)

# A recognized NLCD per-year/census tree: a code-space prefix optionally
# carrying a ``_<N>m`` resolution suffix. ``cleaned``/``cog``/``raw`` are
# raw-code 30 m; ``grouped`` is group-space 30 m; the ``_<N>m`` suffix marks a
# coarse precompute at N metres. An unrecognized subdir (a custom deployment)
# is left unguarded — we can't reason about its space/resolution, and refusing
# would break back-compat for layouts we don't know about.
_TREE_RE = re.compile(
    r"^(?P<space>cleaned|cog|raw|grouped)(?:_(?P<res>\d+)m)?$"
)

# Bare raw-code 30 m trees where building the census is the LEGACY operation
# (Fred's ``cleaned``; this group's 30 m ``cog`` C1b base). For these, an
# existing census + no ``--overwrite`` keeps the legacy silent skip-if-exists
# (back-compat). Every other (isolated / coarse) output tree is one the
# overhaul produces → loud refuse instead, so a forgotten ``--overwrite`` on a
# multi-hour detached rebuild fails fast rather than reusing a stale census.
_LEGACY_BARE_TREES = frozenset({"cleaned", "cog", "raw"})

_PROD_OVERRIDE_VALUES = {"1", "true", "yes", "on"}


def _prod_override() -> bool:
    return (
        os.environ.get("FIRE_ALLOW_PROD_WRITE", "").lower()
        in _PROD_OVERRIDE_VALUES
    )


def _classify_tree(subdir: str) -> "tuple[str, int] | None":
    """``(code_space, resolution_m)`` for a recognized NLCD tree, else None.

    ``code_space`` is ``"group"`` for the ``grouped`` family, else ``"raw"``;
    resolution is the ``_<N>m`` suffix (30 if absent). None signals an
    unrecognized custom layout the guard leaves alone.
    """
    m = _TREE_RE.match(subdir)
    if not m:
        return None
    space = "group" if m.group("space") == "grouped" else "raw"
    res = int(m.group("res")) if m.group("res") else 30
    return space, res


def _assert_census_tree_consistency(
    src_subdir: str,
    out_subdir: str,
    pixel_m_env: "str | None",
) -> None:
    """Raise unless the census output tree is the SAME tree as the per-year
    source m02 reduces (the pure, directly-testable core of the guard).

    Exact-tree equality is the tightest correct invariant: the census
    co-locates with the per-year stack, so it blocks wrong code-space / wrong
    -resolution targets (``cog`` -> ``cog_120m`` / ``grouped_120m``), same
    -shape DIFFERENT-PRODUCT trees a ``(space, resolution)`` classifier would
    collapse (``cleaned`` C1V0 <-> ``cog`` C1V1), AND a redirect from an
    unrecognized custom source onto a canonical tree — every way m02's BUILD
    could overwrite a shared canonical census it did not produce.
    """
    if out_subdir != src_subdir:
        raise RuntimeError(
            f"m02: census output tree {out_subdir!r} != per-year source tree "
            f"{src_subdir!r}. m02 reduces the source per-year stack and the "
            f"census co-locates with it; FIRE_NLCD_MODE_SUBDIR (the "
            f"reader-side census-location hook) must not redirect m02's BUILD "
            f"to a different tree, or it would overwrite a shared canonical "
            f"census with the wrong product / resolution "
            f"(D-2026-06-04-input-layer-rebuild-isolation). Unset "
            f"FIRE_NLCD_MODE_SUBDIR to inherit the source, or "
            f"FIRE_ALLOW_PROD_WRITE=1 to override."
        )
    # m02 never reprojects: a coarse FIRE_PIXEL_M that disagrees with the
    # source tree's native resolution would silently produce a census at the
    # source's resolution, not FIRE_PIXEL_M's (the bare-`grouped` footgun).
    src_kind = _classify_tree(src_subdir)
    if pixel_m_env and src_kind is not None:
        try:
            px = int(pixel_m_env)
        except ValueError:
            return
        if px != src_kind[1]:
            raise RuntimeError(
                f"m02: FIRE_PIXEL_M={px} but the per-year source tree "
                f"{src_subdir!r} is native {src_kind[1]} m. m02 does not "
                f"reproject — it reduces the source stack as-is — so the "
                f"census would be {src_kind[1]} m, not {px} m. Point "
                f"FIRE_NLCD_SUBDIR at the {px} m source tree (e.g. "
                f"grouped_{px}m), or unset FIRE_PIXEL_M."
            )


def _guard_census_write() -> None:
    """Validate the ACTUAL bound write target against the bound source dir.

    Reads ``NLCD_MODE_RASTER_PATH`` / ``NLCD_PATH`` (frozen at import from the
    process env, i.e. the real write target — not a live-env re-derivation
    that could diverge from it). ``FIRE_ALLOW_PROD_WRITE=1`` bypasses.
    """
    if _prod_override():
        return
    src_subdir = NLCD_PATH.name
    out_subdir = NLCD_MODE_RASTER_PATH.parent.name
    _assert_census_tree_consistency(
        src_subdir, out_subdir, os.environ.get("FIRE_PIXEL_M")
    )


def _handle_existing(out_path, out_subdir, overwrite) -> "int | None":
    """Exists / ``--overwrite`` / exit-4 policy for ONE census target.

    Returns ``4`` to abort (an isolated/coarse tree with an existing census and
    no ``--overwrite`` — fail fast rather than silently reuse a stale raster on
    a detached rebuild), else ``None`` to proceed to the build. With
    ``--overwrite`` the existing census is unlinked here. A bare canonical 30 m
    tree (``cleaned``/``cog``/``raw``) returns ``None`` WITHOUT unlinking, so
    the build's ``skip_if_exists=True`` saver keeps the legacy silent skip
    (byte-identical to Fred's m02).
    """
    if out_path.exists():
        if overwrite:
            print(f"[m02] --overwrite: removing existing {out_path}")
            out_path.unlink()
        elif out_subdir not in _LEGACY_BARE_TREES:
            print(
                f"[m02] ERROR: census exists and --overwrite was not set:\n"
                f"  {out_path}\nPass --overwrite to rebuild. Aborting."
            )
            return 4
        # else: bare canonical 30 m tree — legacy silent skip-if-exists.
    return None


def _build_one_census(year_end, out_path, vrt_path) -> None:
    """Stack ``[YEAR_START, year_end]``, reduce to the per-pixel temporal mode,
    save to ``out_path``.

    Uses the canonical ``rts.general.local_stats(stack, "mode")`` kernel — the
    SAME reduction the all-years census uses (nanmode, smallest-value-wins
    tie-break, null iff every band null) — so a per-cutoff census is
    byte-consistent with the all-years census at its shared endpoint.
    ``skip_if_exists=True`` (the saver default) preserves the bare-canonical
    legacy skip; every isolated-tree exists-case has already been unlinked
    (``--overwrite``) or returned 4 before reaching here.
    """
    nlcd_paths = [
        get_nlcd_raster_path(y) for y in range(YEAR_START, year_end + 1)
    ]
    stack_rasters_as_vrt(nlcd_paths, vrt_path)
    nlcd_stack = rts.Raster(vrt_path).astype("int16")
    mode = rts.general.local_stats(nlcd_stack, "mode")
    protected_raster_save_with_cleanup(mode, out_path)


def _run_per_year(year_ends, overwrite) -> int:
    """Emit one leak-free per-cutoff census for each requested window end.

    Every target resolves through ``get_nlcd_mode_raster_path`` into the SAME
    guarded census tree as the all-years census (already validated by
    ``_guard_census_write`` before this runs — the guard keys on the tree, so
    one check covers every per-Y file). Without ``--overwrite``, a pre-flight
    pass refuses (exit 4) if ANY requested target already exists in an isolated
    tree, before writing or deleting anything (no partial set). With
    ``--overwrite``, each stale census is unlinked immediately before its OWN
    rebuild (never all up front), so a crash mid-run loses at most the
    in-flight cutoff, not the already-emitted set. Each cutoff builds through
    its own VRT path (``nlcd_1984_<Y>.vrt``); the ``Y=2022`` case coincides
    with the all-years VRT (identical ``[1984, 2022]`` content — harmless).
    """
    out_of_range = [y for y in year_ends if not (YEAR_START <= y <= YEAR_END)]
    if out_of_range:
        print(
            f"[m02] ERROR: --per-year-ends out of range {out_of_range}; each "
            f"window end must be in [{YEAR_START}, {YEAR_END}] (the per-year "
            f"NLCD data span). Aborting."
        )
        return 2

    targets = [(y, get_nlcd_mode_raster_path(y)) for y in year_ends]

    # Pre-flight (no side effects): without --overwrite, refuse the whole run
    # if any target already exists in an isolated tree — so we never emit a
    # partial set and never touch an existing census. (A bare-canonical
    # existing census is left for the per-target legacy skip-if-exists below.)
    if not overwrite:
        blocked = [
            p
            for _y, p in targets
            if p.exists() and p.parent.name not in _LEGACY_BARE_TREES
        ]
        if blocked:
            print(
                f"[m02] {len(blocked)} existing census(es) block this run; "
                f"pass --overwrite to rebuild. Aborting (nothing written)."
            )
            return 4

    for y, out_path in targets:
        # Unlink this stale census (if --overwrite) only now, right before its
        # own rebuild — so a failure never leaves a wider gap than one cutoff.
        if _handle_existing(out_path, out_path.parent.name, overwrite) == 4:
            return 4  # defensive: the pre-flight above already caught this
        vrt_path = NLCD_PATH / f"nlcd_1984_{y}.vrt"
        print(f"[m02] per-cutoff census 1984..{y}: {out_path}")
        _build_one_census(y, out_path, vrt_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete + rebuild the census even if it exists. Without it, a "
        "build into an isolated tree (grouped / *_<N>m) aborts loud (exit 4) "
        "on an existing census; a build into a bare canonical 30 m tree "
        "(cleaned/cog/raw) keeps the legacy silent skip-if-exists.",
    )
    parser.add_argument(
        "--per-year-ends",
        type=int,
        nargs="+",
        metavar="Y_END",
        help="Phase-2 Flavor-B (D-2026-06-29): emit one leak-free per-cutoff "
        "census nlcd_mode_1984_<Y_END>.tif for each Y_END instead of the "
        "single all-years census. The zero-fire label for scored year Y reads "
        "Y_END=Y-1. Each Y_END must lie in [1984, 2022]; Y_END=2022 "
        "reproduces the all-years census. Same reduction, same guarded tree.",
    )
    args = parser.parse_args()

    _guard_census_write()

    if args.per_year_ends:
        return _run_per_year(sorted(set(args.per_year_ends)), args.overwrite)

    # --- all-years census (default; behaviour unchanged) ---
    out_path = NLCD_MODE_RASTER_PATH
    print(f"[m02] census output: {out_path}")
    rc = _handle_existing(out_path, out_path.parent.name, args.overwrite)
    if rc is not None:
        return rc
    _build_one_census(YEAR_END, out_path, NLCD_STACK_VRT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
