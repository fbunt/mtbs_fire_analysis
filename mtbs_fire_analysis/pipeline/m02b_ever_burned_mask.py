"""Build the binary "ever-burned" CONUS mask (stratification-redesign Stage 1).

A pixel is `1` in the output iff any `dse_{year}.tif` (1984..2022)
has a non-nodata value at that pixel (i.e. any MTBS perimeter
intersected it in the observation window). See
``docs/plans/STRATIFICATION_REDESIGN.md`` §5.2 + §8 Stage 1.

Outputs (m01-class shared-derived write per ``SOLAR_COVARIATE_PLAN.md``
§2 "Deliberate exception: m01-class derived inputs"):

- ``ever_burned_mask.tif`` (uint8 0/1, zstd-compressed) under
  ``${FIRE_DATA_ROOT}/data/mtbs_perims/derived/``.
- ``ever_burned_mask_LATEST.json`` sidecar: timestamp, year range,
  per-input (path, size, mtime), sha256 of newline-joined input paths,
  script sha256.

Run::

    uv run --extra pipeline python \\
        -m mtbs_fire_analysis.pipeline.m02b_ever_burned_mask
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import raster_tools as rts

from mtbs_fire_analysis.pipeline.paths import (
    EVER_BURNED_MASK_LATEST_JSON_PATH,
    EVER_BURNED_MASK_PATH,
    EVER_BURNED_STACK_VRT_PATH,
    PERIMS_RASTERS_PATH,
)
from mtbs_fire_analysis.utils import (
    protected_raster_save_with_cleanup,
    stack_rasters_as_vrt,
)

YEAR_START = 1984
YEAR_END = 2022  # inclusive


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


def _build_mask(dse_paths: list[Path]) -> rts.Raster:
    EVER_BURNED_STACK_VRT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stack_rasters_as_vrt(dse_paths, EVER_BURNED_STACK_VRT_PATH)
    stack = rts.Raster(EVER_BURNED_STACK_VRT_PATH)
    # ``max`` returns nodata only when ALL bands are nodata. Inverting the
    # resulting null-mask gives the binary "any year burned" raster.
    max_dse = rts.general.local_stats(stack, "max")
    return (~max_dse.to_null_mask()).astype("uint8")


def _write_sidecar(dse_paths: list[Path], script_path: Path) -> None:
    paths_concat = "\n".join(str(p) for p in dse_paths).encode("utf-8")
    sidecar = {
        "_about": (
            "Provenance for ever_burned_mask.tif: the binary CONUS raster "
            "where pixel == 1 iff any dse_{year}.tif in "
            f"{YEAR_START}..{YEAR_END} has a non-nodata value. Built per "
            "STRATIFICATION_REDESIGN.md §5.2 + §8 Stage 1."
        ),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "year_start": YEAR_START,
        "year_end": YEAR_END,
        "n_dse_inputs": len(dse_paths),
        "dse_inputs": _input_provenance(dse_paths),
        "dse_paths_sha256": hashlib.sha256(paths_concat).hexdigest(),
        "script_path": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "output_mask_path": str(EVER_BURNED_MASK_PATH),
        "stack_vrt_path": str(EVER_BURNED_STACK_VRT_PATH),
    }
    EVER_BURNED_MASK_LATEST_JSON_PATH.write_text(
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

    EVER_BURNED_MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    mask = _build_mask(dse_paths)
    protected_raster_save_with_cleanup(mask, EVER_BURNED_MASK_PATH)
    _write_sidecar(dse_paths, Path(__file__).resolve())
    print(f"wrote: {EVER_BURNED_MASK_PATH}")
    print(f"sidecar: {EVER_BURNED_MASK_LATEST_JSON_PATH}")


if __name__ == "__main__":
    main()
