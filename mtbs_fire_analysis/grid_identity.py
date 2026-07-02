"""Geohash grid-identity stamping + cross-grid join guard.

The geohash LINEAR index (``geohasher.py``:
``ravel_multi_index((row, col), grid_shape) = row*W + col``) depends on the
active grid's ``shape`` (the stride ``W``) and ``affine``. The
substrate-overhaul divisible-by-256 pad (``FIRE_DIVISIBLE_GRID``) changes the
grid width, so a geohash-keyed table is only join-compatible with another
table built on the SAME grid. A join across grids mis-matches **silently**
(wrong/empty rows, no error) -- a higher-severity failure than the raw-array
shape-mismatch the all-layers barrier was designed for.

This module:

* stamps the active grid identity in a JSON sidecar next to every
  geohash-keyed parquet output (``write_grid_sidecar``), and
* provides a fail-loud guard for the (exactly two) cross-grid join sites
  (``assert_grids_match`` / ``assert_matches_active``).

Semantics (substrate-overhaul Phase-3 §2b, signed off §7 Q-geohash):

* **Strict on mismatch** -- two present-and-different grid identities raise
  ``GridIdentityMismatchError``.
* **Lenient on absence** -- a missing sidecar (legacy / pre-stamp data, or an
  upstream/Fred dataset) WARNs and proceeds; strict-on-absent would break
  every existing run + every upstream user. Tightened to require-present at
  the Phase-3 exit gate.

Back-compat: purely additive -- a new module, new sidecar files, and asserts
that no-op on absent stamps. With ``FIRE_DIVISIBLE_GRID`` default-OFF the
sidecar records the legacy grid and nothing about existing data changes.

See ``docs/plans/SUBSTRATE_OVERHAUL_PHASE3_EXECUTION.md`` §1, §2b.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from mtbs_fire_analysis.defaults import grid_for_pixel_m
from mtbs_fire_analysis.geohasher import GridGeohasher

SCHEMA = "grid_identity/v1"
#: Sidecar filename inside a partitioned-parquet output directory.
SIDECAR_DIR_NAME = "_grid_identity.json"
#: Sidecar suffix beside a single-file parquet output.
SIDECAR_FILE_SUFFIX = ".grid.json"


class GridIdentityMismatchError(RuntimeError):
    """Two geohash-keyed sources were built on different (incompatible) grids.

    Joining them on ``geohash`` would mis-match silently; regenerate both on
    the same grid (substrate-overhaul §1).
    """


def active_grid_payload() -> dict:
    """Identity payload for the grid the default ``GridGeohasher`` hashes with.

    This is exactly what the m10/m11 ``GridGeohasher()`` callers stamp and
    hash with, so it is the correct anchor for the "stored frame vs values
    geohashed fresh in this process" guard (``assert_matches_active``).
    """
    hasher = GridGeohasher()
    return {
        "schema": SCHEMA,
        "grid_id": hasher.grid_id,
        **hasher.grid_descriptor,
    }


def _sidecar_for(out_path: "str | Path", *, beside: bool = False) -> Path:
    """Sidecar path for a geohash-keyed output (directory or single file).

    ``beside=True`` forces the ``<path>.grid.json`` file-suffix form even for a
    directory output, placing the sidecar BESIDE the dir rather than inside it.
    Use this when a ``_grid_identity.json`` *inside* the partitioned dir would
    break a bare ``pl.scan_parquet(dir)`` / ``dd.read_parquet(dir)`` reader of
    that dir (polars errors on "different file extensions").
    ``read_grid_sidecar`` resolves both forms, so beside is transparent.
    """
    path = Path(out_path)
    if path.is_dir() and not beside:
        return path / SIDECAR_DIR_NAME
    return Path(f"{path}{SIDECAR_FILE_SUFFIX}")


def _padding_state_of(geohasher):
    """Whether ``geohasher``'s grid is the padded (``True``) or unpadded
    (``False``) grid at its own resolution, or ``None`` for a custom grid.

    Derived from the geohasher's actual ``grid_shape`` (not the ambient env),
    so the recorded flag can never drift from the grid that was hashed with --
    the authoritative identity is ``grid_id`` / ``grid_shape`` regardless.
    """
    pixel_m = int(round(abs(geohasher.affine.a)))
    shape = tuple(geohasher.grid_shape)
    try:
        if shape == grid_for_pixel_m(pixel_m, padding_enabled=True)[1]:
            return True
        if shape == grid_for_pixel_m(pixel_m, padding_enabled=False)[1]:
            return False
    except ValueError:
        pass
    return None


def write_grid_sidecar(
    out_path, geohasher, *, extra=None, beside=False
) -> Path:
    """Stamp ``geohasher``'s grid identity next to a geohash-keyed output.

    Call this AFTER the parquet write -- a partitioned-dir output needs the
    directory to exist for the ``_grid_identity.json`` placement; a single
    file gets ``<file>.grid.json`` beside it.

    Args:
        out_path: the parquet output (dask partitioned directory or file).
        geohasher: the ``GridGeohasher`` whose grid produced the geohashes.
        extra: optional extra fields to fold into the sidecar payload.
        beside: force the ``<path>.grid.json`` beside-form even for a dir
            output (default False = ``_grid_identity.json`` inside the dir).
            Use for a dir whose bare ``pl.scan_parquet(dir)`` readers would
            break on an in-dir non-parquet file (e.g. the m20 combined frame,
            read bare by a00 / validate / a36). ``read_grid_sidecar`` resolves
            both forms transparently.

    Returns:
        The sidecar path written, or ``None`` if the stamp write failed
        (best-effort: a stamp failure WARNs and is swallowed, never aborts an
        otherwise-successful pipeline run -- the stamp is advisory and the
        join guard is already lenient on an absent sidecar).
    """
    payload = {
        "schema": SCHEMA,
        "grid_id": geohasher.grid_id,
        **geohasher.grid_descriptor,
        "divisible_grid": _padding_state_of(geohasher),
    }
    if extra:
        payload.update(extra)
    sidecar = _sidecar_for(out_path, beside=beside)
    try:
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True))
    except OSError as exc:
        warnings.warn(
            f"grid-identity stamp write failed for {out_path}: {exc}. The "
            "parquet is intact; the downstream join guard will WARN (treat "
            "as unstamped) rather than verify same-grid. Re-stamp by "
            "re-running the writer (substrate-overhaul §2b).",
            stacklevel=2,
        )
        return None
    return sidecar


def read_grid_sidecar(path):
    """Return the stored grid-identity payload, or ``None`` if unstamped.

    ``None`` (legacy / pre-stamp / upstream data, or an unreadable sidecar)
    is the back-compat signal: the guards WARN-and-proceed rather than fail.
    Accepts the same ``path`` shape used to write (directory or single file).
    """
    path = Path(path)
    for cand in (
        path / SIDECAR_DIR_NAME,
        Path(f"{path}{SIDECAR_FILE_SUFFIX}"),
    ):
        if cand.is_file():
            try:
                return json.loads(cand.read_text())
            except (OSError, ValueError) as exc:
                # A sidecar that EXISTS but cannot be parsed is a real problem
                # (corrupt / partial write), not the benign "no stamp" case --
                # warn loudly so it is not silently degraded to "unstamped".
                warnings.warn(
                    f"grid-identity sidecar {cand} exists but is unreadable "
                    f"({exc}); treating as absent (guard will WARN, not "
                    "verify). Re-stamp by re-running the writer.",
                    stacklevel=2,
                )
                return None
    return None


def _fmt(payload) -> str:
    if payload is None:
        return "None"
    return (
        f"grid_id={payload.get('grid_id')} shape={payload.get('grid_shape')}"
    )


def assert_grids_match(
    left, right, *, left_label, right_label, context=""
) -> None:
    """Fail loud iff two PRESENT grid identities differ.

    ``left`` / ``right`` are ``read_grid_sidecar`` payloads (or ``None``).
    A ``None`` side (absent sidecar) WARNs and returns -- never raises
    (lenient on absence). Two present-but-different ``grid_id`` raise
    ``GridIdentityMismatchError`` (strict on mismatch, §7 Q-geohash).
    """
    ctx = f" ({context})" if context else ""
    if left is None or right is None:
        missing = ", ".join(
            lbl
            for lbl, p in ((left_label, left), (right_label, right))
            if p is None
        )
        warnings.warn(
            f"geohash join {left_label} ⋈ {right_label}{ctx}: no "
            f"grid-identity sidecar on [{missing}] -- cannot verify same-grid "
            "(legacy / unstamped data); proceeding. Regenerate with the "
            "stamping writers to enable the guard (substrate-overhaul §2b).",
            stacklevel=2,
        )
        return
    if left["grid_id"] != right["grid_id"]:
        raise GridIdentityMismatchError(
            f"geohash join {left_label} ⋈ {right_label}{ctx}: DIFFERENT "
            f"grids -- {left_label} {_fmt(left)} vs {right_label} "
            f"{_fmt(right)}. A geohash join across grids mis-matches SILENTLY "
            "(wrong/empty rows). Regenerate both on the same grid "
            "(substrate-overhaul §1)."
        )


def assert_matches_active(payload, *, label, context="") -> None:
    """Guard a stored geohash frame against the active process grid.

    Use where a stored frame is joined with values geohashed fresh in this
    process (e.g. m11 joins the loaded frame with raster points hashed by a
    new ``GridGeohasher()``). ``payload`` is from ``read_grid_sidecar``.
    """
    assert_grids_match(
        payload,
        active_grid_payload(),
        left_label=label,
        right_label="active-process-grid",
        context=context,
    )
