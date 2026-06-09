"""Resolution-aware raster reads onto the analysis geobox.

This module lives in ``mtbs_fire_analysis`` (not ``fire_interval``) on purpose:
the pipeline producers that need a coarse-resolution covariate read — notably
``m10_data_extract`` — live here, and the dependency only flows
``fire_interval -> mtbs_fire_analysis`` (never the reverse), so the read
primitive must sit at or below this package for m10 to use it. The grid
selectors it asserts against (``grid_for_pixel_m`` / ``geobox_for_pixel_m``)
already live in ``mtbs_fire_analysis.defaults``.

Two entry points:

* ``read_onto_geobox`` — reduce a source raster onto the analysis ``geobox``
  flat-from-base (``mode`` for categorical, ``average`` for continuous), with a
  fail-loud alignment gate. The lazy/dask result is returned for the totals
  side to clip-then-compute.
* ``sample_on_geobox`` — read a source at analysis-grid integer indices
  (e.g. ``GridGeohasher.geohash_to_ij``) *resolution-correctly*: it reduces the
  source onto ``geobox`` FIRST, so the positional index is valid by
  construction regardless of the source's native pixel size. This is the fix
  for the m10 2026-06-09 bug, where 120 m geohash indices were read straight
  into a 30 m raster (a 4x position misalignment that dropped 57% of burned
  pixels to off-CONUS nodata and mis-valued the rest). It returns a coverage
  fraction so callers can fail loud on a silent grid/resolution mismatch.

Project-level policy (the coverage threshold, FAIL/WARN/INFO severity, the
blessed analysis-resolution set) stays in ``fire_interval`` — this module is
generic geometry. The alignment gate is generic enough to eventually graduate
to ``raster_tools`` itself.

``fire_interval.cog`` re-exports ``read_onto_geobox`` /
``_assert_geobox_alignment`` for back-compat with existing importers.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import rasterio

if TYPE_CHECKING:
    import raster_tools as rts
    from odc.geo.geobox import GeoBox

ResamplingName = Literal["average", "mode", "nearest", "bilinear"]


def read_onto_geobox(
    path: Path,
    geobox: GeoBox,
    *,
    resample_method: ResamplingName,
    require_nodata: bool = True,
    require_native_resolution: bool = False,
) -> rts.Raster:
    """Read a raster onto the canonical analysis ``geobox`` via a
    flat-from-base reproject (PATH A), with a fail-loud alignment gate.

    This is the coarse-input read primitive for 120 m / 480 m analysis.
    ``rts.Raster(path).reproject(geobox, resample_method=...)`` reads the 30 m
    base and computes the reduction **flat-from-base on the fly** — it does NOT
    consume whatever cascade overviews the COG carries. (Verified empirically:
    a single base->coarse ``mode`` equals the flat-from-base mode, not the
    cascade mode; the two differ ~6-12 % at category boundaries, which is the
    whole reason this read exists rather than serving a baked overview.) Use
    ``"mode"`` for categorical inputs (NLCD-grouped, eco, hexel) and
    ``"average"`` for continuous (DEM, solar, climate).

    Returns the **lazy** ``rts.Raster`` (dask-backed) so the caller can chain a
    clip-to-eco before the single terminal compute. DO NOT materialise here
    (``.values`` / ``np.asarray``) — that would build the full coarse-CONUS
    grid (16x/256x the base footprint) in memory; the windowed clip must slice
    the lazy graph first. See ``docs/plans/COARSENING_READ_PATH.md`` section 8.

    ``geobox`` must come from ``geobox_for_pixel_m(N)`` so its shape is the
    floor shape (a partial edge cell is dropped, not padded); the gate asserts
    the reproject landed on exactly that grid, catching the GDAL
    ceil-overview-vs-selector-floor off-by-one (+1 row at factor 4, +1 on both
    axes at factor 16).

    Args:
        path: source raster (any resolution; canonically the 30 m base COG).
        geobox: target analysis grid from ``geobox_for_pixel_m(N)``.
        resample_method: ``"mode"`` (categorical) or ``"average"``
            (continuous); ``"nearest"`` / ``"bilinear"`` also accepted.
        require_nodata: when True (default), raise if ``path`` carries no
            nodata tag. The ``mode``/``average`` nodata-exclusion that keeps
            the coarse counts four-types-safe is **contingent** on the source
            nodata tag (grouped NLCD = 250, continuous = -99999): an untagged
            sentinel silently *wins* the mode of a mostly-nodata cell and
            corrupts the zero-fire / type-1 counts. Pass ``False`` only for a
            source that genuinely contains no sentinel values.
        require_native_resolution: when True, raise unless ``path``'s native
            pixel size already equals ``geobox``'s. Use ONLY for a ``nearest``
            point-sample whose contract is that the source is already at the
            analysis resolution (the fire-history burned/first-burn-year mask —
            point-sampled at build time, not reduced at read time). A finer
            source point-sampled onto a coarse cell silently undercounts the
            burned/zero split while ``burned + zero = total`` still holds (both
            halves come from the same mis-sampled array), so no downstream
            conservation gate can catch it — the guard is the only place the
            error is visible. Leave False for the ``mode``/``average`` reads,
            which legitimately reduce the 30 m base onto a coarse geobox.

    Returns:
        The lazy reprojected ``rts.Raster`` on ``geobox``.

    Raises:
        ValueError: source lacks a nodata tag (and ``require_nodata``); source
            native pixel size != geobox (and ``require_native_resolution``);
            or the reproject output does not land on ``geobox`` (shape /
            transform).
    """
    if require_native_resolution:
        _assert_source_is_native_resolution(path, geobox)
    if require_nodata:
        with rasterio.open(str(path)) as ds:
            if ds.nodata is None:
                raise ValueError(
                    f"read_onto_geobox: source {path} carries no nodata tag; "
                    f"the flat-from-base {resample_method!r} nodata-exclusion "
                    "is contingent on it (an untagged sentinel wins the mode "
                    "of a mostly-nodata cell and corrupts the zero-fire / "
                    "type-1 counts). Stamp the COG's nodata (grouped "
                    "NLCD=250, continuous=-99999) before reading it onto a "
                    "coarse geobox, or pass require_nodata=False for a source "
                    "with no sentinel values."
                )

    import raster_tools as rts

    src = rts.Raster(str(path))
    out = src.reproject(geobox, resample_method=resample_method)
    _assert_geobox_alignment(out, geobox, path)
    return out


def sample_on_geobox(
    path: Path,
    geobox: "GeoBox | None",
    ij: "tuple",
    *,
    resample_method: ResamplingName,
    require_nodata: bool = True,
) -> "tuple":
    """Read ``path`` at analysis-grid integer indices ``ij``, correctly.

    ``ij`` is ``(rows, cols)`` — integer positions in ``geobox``'s grid (e.g.
    from ``GridGeohasher.geohash_to_ij``). When ``geobox`` is set, the source
    is reduced onto it via ``read_onto_geobox`` FIRST, so the positional index
    is valid by construction *regardless of the source's native resolution*.
    This is the fix for the m10 2026-06-09 bug: 120 m geohash indices read
    straight into a 30 m raster land ~4x toward the origin (dropping 57% of
    burned pixels to off-CONUS nodata, mis-valuing the rest); reducing onto the
    geobox first puts the array on the same grid ``ij`` was computed for.

    ``geobox=None`` is the native (30 m) path: index the source directly —
    byte-identical to the legacy ``_add_raster`` read, so 30 m is unchanged.

    Returns ``(values, coverage)``: ``values`` is one entry per index;
    ``coverage`` is the fraction of sampled cells that are NOT the source
    nodata — the signal the caller checks to fail loud on a silent
    grid/resolution mismatch (the missing fire-data-engineering coverage
    Check; the alignment gate inside ``read_onto_geobox`` is the other half).
    """
    import numpy as np
    import raster_tools as rts
    import xarray as xr

    rows = np.asarray(ij[0])
    cols = np.asarray(ij[1])
    n = int(rows.size)

    if geobox is None:
        out = rts.Raster(str(path))
    else:
        out = read_onto_geobox(
            path,
            geobox,
            resample_method=resample_method,
            require_nodata=require_nodata,
        )

    vals = out.xdata.isel(
        band=xr.DataArray(np.zeros(n, dtype=int), dims="z"),
        y=xr.DataArray(rows, dims="z"),
        x=xr.DataArray(cols, dims="z"),
    ).to_numpy()

    nv = out.null_value
    coverage = 1.0 if (nv is None or n == 0) else float((vals != nv).mean())
    return vals, coverage


def _assert_source_is_native_resolution(path: Path, geobox: GeoBox) -> None:
    """Fail loud unless ``path``'s native pixel size equals ``geobox``'s.

    For a ``nearest`` mask read the contract is that the source is ALREADY at
    the analysis resolution (the fire-history mask is point-sampled at build
    time, not reduced at read time), so the reproject is an identity resample.
    Unlike the ``mode`` / ``average`` reads — which legitimately reduce the
    30 m base onto a coarse geobox — a resolution mismatch here is silent
    corruption: ``nearest`` point-samples one fine pixel per coarse cell, the
    alignment gate still passes (the output lands on the geobox floor shape),
    and the burned/zero split is undercounted while ``burned + zero = total``
    still holds (both halves come from the same mis-sampled array). Guard at
    the read, where the error is visible — the downstream conservation gate
    structurally cannot see it. Nightly review 2026-06-09 [risk].
    """
    gb_res_x = abs(geobox.transform.a)
    gb_res_y = abs(geobox.transform.e)
    with rasterio.open(str(path)) as ds:
        src_res_x = abs(ds.transform.a)
        src_res_y = abs(ds.transform.e)
    tol_x = 1e-6 * gb_res_x
    tol_y = 1e-6 * gb_res_y
    if abs(src_res_x - gb_res_x) > tol_x or abs(src_res_y - gb_res_y) > tol_y:
        raise ValueError(
            f"read_onto_geobox: source {path} native pixel size "
            f"({src_res_x}x{src_res_y} m) != analysis geobox pixel size "
            f"({gb_res_x}x{gb_res_y} m), but require_native_resolution=True. "
            "This read is contracted native (a point-sampled fire-history "
            "mask): a 'nearest' point-sample of a finer source onto a coarse "
            "cell undercounts the burned/zero split while burned+zero=total "
            "still holds, so no downstream gate catches it. Build the mask at "
            "the analysis resolution (derived_<N>m/) before reading it onto a "
            "coarse geobox."
        )


def _assert_geobox_alignment(
    out: rts.Raster, geobox: GeoBox, path: Path
) -> None:
    """Fail loud unless the reproject landed on exactly ``geobox``.

    ``rts.reproject`` honours the passed geobox by construction, so this is a
    guard against a wrong-geobox caller / future regression and against the
    GDAL ceil-vs-selector-floor off-by-one (an overview-shaped read would be
    +1 row at factor 4, +1 on both axes at factor 16 — exactly where silent
    geospatial misalignment hides). Shape + transform are load-bearing
    (raise); CRS is advisory only, because production rasters carry different
    authority codes (ESRI:102039 vs EPSG:5070) for the same CONUS Albers — a
    transform+shape match means the cells are co-located regardless.
    """
    out_gb = out.geobox
    if tuple(out_gb.shape) != tuple(geobox.shape):
        raise ValueError(
            f"read_onto_geobox: output shape {tuple(out_gb.shape)} != target "
            f"floor shape {tuple(geobox.shape)} for {path}; a ceil-rounded / "
            "padded reproject would be +1 row (factor 4) or +1 on both axes "
            "(factor 16). geobox must come from geobox_for_pixel_m(N)."
        )
    if not out_gb.transform.almost_equals(geobox.transform, precision=1e-6):
        raise ValueError(
            f"read_onto_geobox: output transform {out_gb.transform} != target "
            f"{geobox.transform} for {path} (pixel size f*30 or base origin "
            "drift)."
        )
    if out_gb.crs != geobox.crs:
        warnings.warn(
            f"read_onto_geobox: output CRS ({out_gb.crs}) != target "
            f"({geobox.crs}) for {path}; transform+shape match so cells are "
            "co-located (expected authority-code difference, not an error).",
            stacklevel=2,
        )
