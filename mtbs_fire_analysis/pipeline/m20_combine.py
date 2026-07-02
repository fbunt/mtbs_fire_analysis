import argparse
from pathlib import Path

import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

from mtbs_fire_analysis.geohasher import GridGeohasher
from mtbs_fire_analysis.grid_identity import (
    SIDECAR_FILE_SUFFIX,
    assert_matches_active,
    read_grid_sidecar,
    write_grid_sidecar,
)

# Try to keep dask from leaking memory
dask.config.set(
    {
        "distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0,
        "distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_": 0,
    }
)


def _path(path):
    path = Path(path)
    assert path.exists()
    return path


DESC = """Combine dataframes to form a large dataframe of burn records."""


def _get_parser():
    p = argparse.ArgumentParser(description=DESC)
    p.add_argument(
        "-j",
        "--num_workers",
        type=int,
        default=5,
        help=(
            "Number of workers for Dask cluster when combining dataframes. "
            "Default is 5."
        ),
    )
    p.add_argument("files", type=_path, nargs="+", help="Files to combine")
    p.add_argument("out_path", type=Path, help="Output path")
    return p


def main(files, out_path, num_workers):
    assert not out_path.exists()

    # Fail loud if a stale, OLD-grid per-year frame slipped into the combine.
    # m10 is skip-if-exists with no --overwrite, so a pre-flip frame can
    # survive a grid re-encode; each per-year frame is grid-stamped by m10, so
    # any input whose grid differs from the active process grid is a mixed-grid
    # combine and must abort. Track whether EVERY input carried a stamp: an
    # unstamped input (legacy / pre-stamp data) WARNs and proceeds, but must
    # NOT let us forge a same-grid stamp on the output below.
    all_inputs_stamped = True
    for f in files:
        payload = read_grid_sidecar(f)
        if payload is None:
            all_inputs_stamped = False
        assert_matches_active(
            payload,
            label=f"m20 input {Path(f).name}",
            context="m20_combine: all inputs must share the active grid",
        )

    with (
        LocalCluster(n_workers=num_workers) as cluster,
        Client(cluster) as _,
    ):
        ddfs = [dd.read_parquet(path) for path in files]
        ddf = dd.concat(ddfs)
        ddf.to_parquet(out_path)

    # Stamp the combined frame's grid identity so the downstream a00
    # (m10-combined join m10b) geohash guard can verify same-grid instead of
    # degrading to a WARN. Write it BESIDE the partitioned dir, NOT inside:
    # a `_grid_identity.json` inside the dir breaks bare pl.scan_parquet(dir)
    # readers of the combined frame (a00, validate, a36).
    #
    # Two safety conditions (both from the pre-commit adversarial verify):
    #  - Unlink any stale beside sidecar from a prior run FIRST, so a swallowed
    #    stamp-write failure (write_grid_sidecar -> None) can never leave a
    #    forged claim reading as valid on freshly-combined data.
    #  - Stamp ONLY when EVERY input was present-and-matching. If any input was
    #    unstamped we cannot attest the combined grid; leave the output
    #    unstamped so a00 keeps WARNing (the honest lenient-on-absence signal)
    #    rather than trusting an active-grid stamp we cannot back up.
    beside_sidecar = Path(f"{out_path}{SIDECAR_FILE_SUFFIX}")
    beside_sidecar.unlink(missing_ok=True)
    if all_inputs_stamped:
        write_grid_sidecar(out_path, GridGeohasher(), beside=True)


if __name__ == "__main__":
    args = _get_parser().parse_args()
    main(args.files, args.out_path, args.num_workers)
