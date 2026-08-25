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

Semantics (substrate-overhaul Phase-3 §2b, signed off §7 Q-geohash;
require-present tightening 2026-08-25, phd-research register row
``D-2026-06-24-geohash-grid-version-guard``):

* **Strict on mismatch** -- two present-and-different grid identities raise
  ``GridIdentityMismatchError``.
* **Strict on absence** -- a missing sidecar, or one too malformed to compare
  on, raises ``GridIdentityMissingError``. An unverifiable join is exactly the
  silent mis-match this guard exists to stop, so it is not a benign case.
* **Named escape hatch** -- ``FIRE_GRID_ALLOW_UNSTAMPED=1`` restores the
  warn-and-proceed behaviour for a deliberate legacy / upstream read. It never
  relaxes a present-but-mismatched pair, and never changes the v1/v2 dialect
  comparison below.

Back-compat: reading legacy / pre-stamp / upstream data now requires either
re-running the stamping writers (m10/m10b/m11) or setting
``FIRE_GRID_ALLOW_UNSTAMPED=1`` deliberately. With ``FIRE_DIVISIBLE_GRID``
default-OFF the sidecar records the legacy grid and nothing about the grid
itself changes.

**Schema v2 (2026-08-04) adds the CRS term.** Shape and affine are datum-blind,
and the planned WGS84 re-anchor moves only the datum -- so a v1 stamp cannot
tell a NAD83 table from a WGS84 one, and a cross-datum geohash join would pass
this guard silently. ``grid_id`` therefore now hashes a declared ``crs_id``
alongside the geometry, which means every stamp written before this change
stores an id a v2 process recomputes differently.

Rather than make that a flag day, the guard **negotiates dialects**: a v1 stamp
is compared on the projection it does guarantee (shape + affine), which is
exactly the comparison this guard made before the term existed. Nothing is
weakened relative to a v1 stamp's own guarantee. What it cannot do is separate
two datums, so the mixed case WARNs -- this is a migration window, not a steady
state. ``geohash_table_check(..., require_crs_aware=True)`` is the mechanical
gate: the datum re-anchor must not begin while any v1 stamp survives.

See ``docs/plans/SUBSTRATE_OVERHAUL_PHASE3_EXECUTION.md`` §1, §2b and
phd-research ``docs/plans/G3_CRS_AWARE_GRID_IDENTITY.md`` §5.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

from affine import Affine

from mtbs_fire_analysis.defaults import grid_for_pixel_m, grid_id_from
from mtbs_fire_analysis.geohasher import GridGeohasher

#: The pre-2026-08 dialect: ``grid_shape`` + ``affine``, and therefore
#: **datum-blind**. Still readable, and still the only projection on which a
#: v1 stamp and a v2 stamp can be compared.
SCHEMA_V1 = "grid_identity/v1"
#: Adds the declared CRS term (``crs_id``), so a datum-only re-anchor produces
#: a different ``grid_id`` instead of an identical one.
SCHEMA_V2 = "grid_identity/v2"
#: What new stamps are written as.
SCHEMA = SCHEMA_V2
#: Sidecar filename inside a partitioned-parquet output directory.
SIDECAR_DIR_NAME = "_grid_identity.json"
#: Sidecar suffix beside a single-file parquet output.
SIDECAR_FILE_SUFFIX = ".grid.json"
#: Set to ``"1"`` to allow a join whose grids cannot be verified (an absent or
#: malformed stamp) -- a deliberate legacy / upstream read. Never relaxes a
#: present-but-mismatched pair.
ALLOW_UNSTAMPED_ENV = "FIRE_GRID_ALLOW_UNSTAMPED"


def _unstamped_reads_allowed() -> bool:
    """Whether the escape hatch is set RIGHT NOW.

    Read at call time, never captured at import: a caller that sets the var
    around one deliberate legacy read (and a test that monkeypatches it) must
    still take effect on an already-imported module.
    """
    return os.environ.get(ALLOW_UNSTAMPED_ENV) == "1"


class GridIdentityMismatchError(RuntimeError):
    """Two geohash-keyed sources were built on different (incompatible) grids.

    Joining them on ``geohash`` would mis-match silently; regenerate both on
    the same grid (substrate-overhaul §1).
    """


class GridIdentityMissingError(GridIdentityMismatchError):
    """A geohash join partner has no readable grid-identity sidecar.

    Raised when a side is unstamped (or carries a stamp too malformed to
    compare on) and unstamped reads were not explicitly allowed via
    ``FIRE_GRID_ALLOW_UNSTAMPED=1``. Subclasses the mismatch error, so any
    caller that catches "these grids are not verified compatible" catches
    both the different-grids and the cannot-tell cases.
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
        otherwise-successful pipeline run -- the parquet is the expensive
        artifact, and the unstamped output it leaves behind is caught loudly
        at the next join rather than silently trusted).
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
            "parquet is intact but reads as UNSTAMPED, so the downstream "
            f"join guard will REFUSE it unless {ALLOW_UNSTAMPED_ENV}=1. "
            "Re-stamp by re-running the writer (substrate-overhaul §2b).",
            stacklevel=2,
        )
        return None
    return sidecar


def read_grid_sidecar(path):
    """Return the stored grid-identity payload, or ``None`` if unstamped.

    ``None`` (legacy / pre-stamp / upstream data, or an unreadable sidecar)
    means the guards cannot verify the grid, so they REFUSE the join unless
    ``FIRE_GRID_ALLOW_UNSTAMPED=1`` is set for a deliberate legacy read.
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
                    f"({exc}); treating as absent, so the join guard will "
                    f"REFUSE unless {ALLOW_UNSTAMPED_ENV}=1. Re-stamp by "
                    "re-running the writer.",
                    stacklevel=2,
                )
                return None
    return None


def is_crs_aware(payload) -> bool:
    """Whether ``payload`` carries the CRS term (i.e. is a v2 stamp).

    Keyed on the ``crs_id`` field rather than the ``schema`` string, because
    the field is what actually determines comparability -- and because
    ``schema`` was written-but-never-read before 2026-08, so an old file's
    string is not evidence of anything. A payload that is not a dict at all
    (a JSON list/str/int on disk) is not a stamp in either dialect, so it
    reads as ``False`` here and falls into the malformed branch of
    ``assert_grids_match`` instead of raising ``TypeError`` from the ``in``.
    """
    return isinstance(payload, dict) and "crs_id" in payload


def crs_blind_id(payload):
    """``payload``'s id computed WITHOUT the CRS term, or ``None``.

    For a v1 stamp this is just its stored ``grid_id``. For a v2 stamp it is
    recomputed from the shape and affine the stamp carries -- which is why
    both dialects remain comparable without re-stamping anything. Anything
    that is not a stamp dict yields ``None`` (i.e. "malformed").
    """
    if not isinstance(payload, dict):
        return None
    if not is_crs_aware(payload):
        return payload.get("grid_id")
    shape, affine = payload.get("grid_shape"), payload.get("affine")
    if shape is None or affine is None:
        return None
    return grid_id_from(shape, Affine(*affine[:6]))


def _stored_grid_id(payload):
    """``payload``'s stored ``grid_id``, or ``None`` if it carries none.

    The same-dialect counterpart of ``crs_blind_id``: both sides are the same
    dialect, so their stored ids are directly comparable -- but only if they
    exist. A missing key (or a payload that is not a stamp dict) is malformed,
    never a value to compare on.
    """
    if not isinstance(payload, dict):
        return None
    return payload.get("grid_id")


def _fmt(payload) -> str:
    if payload is None:
        return "None"
    crs = payload.get("crs_id", "<datum-blind v1>")
    return (
        f"grid_id={payload.get('grid_id')} shape={payload.get('grid_shape')} "
        f"crs={crs}"
    )


def assert_grids_match(
    left, right, *, left_label, right_label, context=""
) -> None:
    """Fail loud unless two grid identities are verified compatible.

    ``left`` / ``right`` are ``read_grid_sidecar`` payloads (or ``None``).
    Two present-but-different ``grid_id`` raise ``GridIdentityMismatchError``
    (strict on mismatch, §7 Q-geohash). A ``None`` side (absent sidecar)
    raises ``GridIdentityMissingError`` -- the join cannot be verified at all
    -- unless ``FIRE_GRID_ALLOW_UNSTAMPED=1``, which downgrades that case
    (only) to the pre-tightening WARN-and-proceed. A present-but-uncomparable
    stamp (no ``grid_id``, or no shape+affine to compare a v1 against a v2 on)
    takes that same strict path: it looks like provenance while carrying none.
    """
    ctx = f" ({context})" if context else ""
    if left is None or right is None:
        missing = ", ".join(
            lbl
            for lbl, p in ((left_label, left), (right_label, right))
            if p is None
        )
        if not _unstamped_reads_allowed():
            raise GridIdentityMissingError(
                f"geohash join {left_label} ⋈ {right_label}{ctx}: no "
                f"grid-identity sidecar on [{missing}] -- cannot verify "
                "same-grid. A geohash is only meaningful within its own grid, "
                "so an unverifiable join mis-matches SILENTLY (wrong/empty "
                "rows, no error). Re-run the stamping writers (m10/m10b/m11, "
                "substrate-overhaul §2b) to stamp the data, or set "
                f"{ALLOW_UNSTAMPED_ENV}=1 for a deliberate legacy/upstream "
                "read."
            )
        warnings.warn(
            f"geohash join {left_label} ⋈ {right_label}{ctx}: no "
            f"grid-identity sidecar on [{missing}] -- cannot verify same-grid "
            "(legacy / unstamped data); proceeding. Regenerate with the "
            "stamping writers to enable the guard (substrate-overhaul §2b). "
            f"Proceeding only because {ALLOW_UNSTAMPED_ENV}=1.",
            stacklevel=2,
        )
        return
    # Dialect skew: one side predates the CRS term. Compare on the projection
    # they share -- shape + affine -- which is exactly the comparison this
    # guard made before the term existed, so nothing is weakened relative to
    # the v1 stamp's own guarantee. What it CANNOT do is separate two datums,
    # so it warns: this is the migration window, not a steady state, and F4b
    # must not begin until no v1 stamps remain (G3 §5).
    if is_crs_aware(left) != is_crs_aware(right):
        lid, rid = crs_blind_id(left), crs_blind_id(right)
        legacy = left_label if not is_crs_aware(left) else right_label
        if lid is None or rid is None:
            # A present-but-malformed stamp is worse than an absent one: it
            # looks like provenance while carrying none. Same strictness, same
            # escape hatch.
            if not _unstamped_reads_allowed():
                raise GridIdentityMissingError(
                    f"geohash join {left_label} ⋈ {right_label}{ctx}: a stamp "
                    "is malformed (no shape/affine to compare on) -- cannot "
                    "verify same-grid. A geohash is only meaningful within "
                    "its own grid, so an unverifiable join mis-matches "
                    "SILENTLY (wrong/empty rows, no error). Re-run the "
                    "stamping writers (m10/m10b/m11, substrate-overhaul §2b) "
                    f"to re-stamp the data, or set {ALLOW_UNSTAMPED_ENV}=1 "
                    "for a deliberate legacy/upstream read."
                )
            warnings.warn(
                f"geohash join {left_label} ⋈ {right_label}{ctx}: a stamp is "
                "malformed (no shape/affine to compare on); proceeding "
                "unverified. Re-stamp by re-running the writer. Proceeding "
                f"only because {ALLOW_UNSTAMPED_ENV}=1.",
                stacklevel=2,
            )
            return
        if lid != rid:
            raise GridIdentityMismatchError(
                f"geohash join {left_label} ⋈ {right_label}{ctx}: DIFFERENT "
                f"grids -- {left_label} {_fmt(left)} vs {right_label} "
                f"{_fmt(right)}. A geohash join across grids mis-matches "
                "SILENTLY (wrong/empty rows). Regenerate both on the same "
                "grid (substrate-overhaul §1)."
            )
        warnings.warn(
            f"geohash join {left_label} ⋈ {right_label}{ctx}: [{legacy}] "
            f"carries a {SCHEMA_V1} stamp with no CRS term, so the grids were "
            "verified on shape+affine only -- a cross-DATUM join would not be "
            f"caught. Re-stamp it to {SCHEMA_V2} to enable that check.",
            stacklevel=2,
        )
        return

    lid, rid = _stored_grid_id(left), _stored_grid_id(right)
    if lid is None or rid is None:
        # Same-dialect twin of the malformed branch above. Without it a
        # payload with no ``grid_id`` raised a bare KeyError out of the join
        # site, and TWO of them compared equal (None == None) and passed
        # SILENTLY -- the one fail-open left in the module. Same strictness,
        # same escape hatch.
        bad = ", ".join(
            lbl
            for lbl, gid in ((left_label, lid), (right_label, rid))
            if gid is None
        )
        if not _unstamped_reads_allowed():
            raise GridIdentityMissingError(
                f"geohash join {left_label} ⋈ {right_label}{ctx}: a stamp is "
                f"malformed (no grid_id to compare on) on [{bad}] -- cannot "
                "verify same-grid. A geohash is only meaningful within its "
                "own grid, so an unverifiable join mis-matches SILENTLY "
                "(wrong/empty rows, no error). Re-run the stamping writers "
                "(m10/m10b/m11, substrate-overhaul §2b) to re-stamp the "
                f"data, or set {ALLOW_UNSTAMPED_ENV}=1 for a deliberate "
                "legacy/upstream read."
            )
        warnings.warn(
            f"geohash join {left_label} ⋈ {right_label}{ctx}: a stamp is "
            f"malformed (no grid_id to compare on) on [{bad}]; proceeding "
            "unverified. Re-stamp by re-running the writer. Proceeding only "
            f"because {ALLOW_UNSTAMPED_ENV}=1.",
            stacklevel=2,
        )
        return

    if lid != rid:
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
