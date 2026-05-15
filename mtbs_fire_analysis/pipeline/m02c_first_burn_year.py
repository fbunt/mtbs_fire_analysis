"""Build the int16 "first-burn-year" CONUS raster (stratification-redesign
Stage 2c).

Pixel value is the first year in `[MTBS_PERIM_YEAR_START,
MTBS_PERIM_YEAR_END]` for which `dse_{year}.tif` has a non-nodata value
at that pixel; nodata (`-1`) where the pixel never burned in the window.
This is the per-max-date generalisation of `ever_burned_mask.tif` per
``docs/plans/STRATIFICATION_REDESIGN.md`` §8 Stage 2c +
`DECISION_REGISTER.md` D-2026-05-14: the legacy fixed-window mask is a
strict special case (``derived_mask(W_1) == ever_burned_mask``,
enforced by `I_derived_mask_consistency`).

Outputs (m01-class shared-derived write per
``SOLAR_COVARIATE_PLAN.md`` §2 "Deliberate exception: m01-class derived
inputs", matching the m02b convention):

- ``first_burn_year.tif`` (int16, nodata=-1, zstd-compressed) under
  ``${FIRE_DATA_ROOT}/data/mtbs_perims/derived/``.
- ``first_burn_year_LATEST.json`` sidecar: timestamp, year range,
  per-input (path, size, mtime), sha256 of newline-joined input paths,
  script sha256.

Reuses the m02b ``dse_stack_1984_2022.vrt`` artefact (rebuilt
unconditionally — gdalbuildvrt overwrites).

Run::

    uv run --extra pipeline python \\
        -m mtbs_fire_analysis.pipeline.m02c_first_burn_year
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import dask.array as da
import numpy as np
import raster_tools as rts

from mtbs_fire_analysis.pipeline.paths import (
    FIRST_BURN_YEAR_LATEST_JSON_PATH,
    FIRST_BURN_YEAR_PATH,
    FIRST_BURN_YEAR_STACK_VRT_PATH,
    MTBS_PERIM_YEAR_END,
    MTBS_PERIM_YEAR_START,
    PERIMS_RASTERS_PATH,
)
from mtbs_fire_analysis.utils import (
    protected_raster_save_with_cleanup,
    stack_rasters_as_vrt,
)

# Re-exported under the local-LOC-friendly aliases used in the build code.
YEAR_START = MTBS_PERIM_YEAR_START
YEAR_END = MTBS_PERIM_YEAR_END  # inclusive

# Output nodata sentinel: -1 is unambiguously outside the valid year
# range [1984, 2022] and works with int16 storage.
FIRST_BURN_NODATA = np.int16(-1)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _input_provenance(dse_paths: list[Path]) -> list[dict]:
    out = []
    for p in dse_paths:
        st = p.stat()
        out.append(
            {
                "path": str(p),
                "size_bytes": st.st_size,
                "mtime_iso": datetime.fromtimestamp(
                    st.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    return out


def _build_first_burn_year(dse_paths: list[Path]) -> rts.Raster:
    """Compute the per-pixel first burn year as a dask-backed Raster.

    Strategy: build the VRT stack (one band per year), translate each
    band into "year if burned, INT16_MAX if not" via the per-band null
    mask, then min-reduce along the band axis. Pixels with no burn in
    any year land at INT16_MAX and are mapped to FIRST_BURN_NODATA in a
    final pass.
    """
    FIRST_BURN_YEAR_STACK_VRT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stack_rasters_as_vrt(dse_paths, FIRST_BURN_YEAR_STACK_VRT_PATH)
    stack = rts.Raster(FIRST_BURN_YEAR_STACK_VRT_PATH)

    # Per-band True iff pixel is null (i.e. NOT burned that year). Using
    # to_null_mask() lets the build stay null-sentinel-agnostic the same
    # way m02b does.
    null_mask = stack.to_null_mask().data  # dask bool (n_bands, H, W)

    n_bands = null_mask.shape[0]
    expected = YEAR_END - YEAR_START + 1
    if n_bands != expected:
        raise ValueError(
            f"VRT stack band count {n_bands} != expected "
            f"{expected} ({YEAR_START}..{YEAR_END}); input list "
            f"length: {len(dse_paths)}"
        )

    int16_sentinel = np.int16(np.iinfo(np.int16).max)
    years = np.arange(
        YEAR_START, YEAR_END + 1, dtype=np.int16
    )[:, None, None]
    # year_or_sentinel[band, h, w] = year(band) where burned else sentinel.
    year_or_sentinel = da.where(null_mask, int16_sentinel, years).astype(
        np.int16
    )
    first_burn = year_or_sentinel.min(axis=0)
    first_burn = da.where(
        first_burn == int16_sentinel, FIRST_BURN_NODATA, first_burn
    ).astype(np.int16)
    return rts.data_to_raster_like(
        first_burn, like=stack, nv=FIRST_BURN_NODATA
    )


def _write_sidecar(dse_paths: list[Path], script_path: Path) -> None:
    paths_concat = "\n".join(str(p) for p in dse_paths).encode("utf-8")
    sidecar = {
        "_about": (
            "Provenance for first_burn_year.tif: int16 CONUS raster "
            "where pixel value is the first year (in "
            f"{YEAR_START}..{YEAR_END}) for which dse_{{year}}.tif has "
            f"a non-nodata value; pixel == {int(FIRST_BURN_NODATA)} "
            "(nodata) where no year burned. Built per "
            "STRATIFICATION_REDESIGN.md §8 Stage 2c + D-2026-05-14."
        ),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "year_start": YEAR_START,
        "year_end": YEAR_END,
        "nodata_value": int(FIRST_BURN_NODATA),
        "n_dse_inputs": len(dse_paths),
        "dse_inputs": _input_provenance(dse_paths),
        "dse_paths_sha256": hashlib.sha256(paths_concat).hexdigest(),
        "script_path": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "output_first_burn_year_path": str(FIRST_BURN_YEAR_PATH),
        "stack_vrt_path": str(FIRST_BURN_YEAR_STACK_VRT_PATH),
        "stack_vrt_note": (
            "Rebuilt unconditionally by every m02b/m02c run via "
            "gdalbuildvrt; not a long-lived artefact and has no "
            "independent provenance. Trust this sidecar's "
            "dse_inputs[] for what the raster was built from."
        ),
    }
    FIRST_BURN_YEAR_LATEST_JSON_PATH.write_text(
        json.dumps(sidecar, indent=2) + "\n"
    )


def main():
    dse_paths = [
        PERIMS_RASTERS_PATH / f"dse_{y}.tif"
        for y in range(YEAR_START, YEAR_END + 1)
    ]
    missing = [p for p in dse_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} dse_*.tif inputs (first 3): "
            f"{[str(p) for p in missing[:3]]}"
        )

    FIRST_BURN_YEAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    raster = _build_first_burn_year(dse_paths)
    protected_raster_save_with_cleanup(raster, FIRST_BURN_YEAR_PATH)
    _write_sidecar(dse_paths, Path(__file__).resolve())
    print(f"wrote: {FIRST_BURN_YEAR_PATH}")
    print(f"sidecar: {FIRST_BURN_YEAR_LATEST_JSON_PATH}")


if __name__ == "__main__":
    main()
