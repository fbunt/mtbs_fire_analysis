"""Wave-0 D9 gate harness (platform spec section 10; wave-0 plan S13).

Runs the consumer's ``m10`` over the gate years reading every input through the
jlab client against the imported platform store, then diffs each year's output
against a frozen golden fixture with the platform's dataset-agnostic comparator
(``jlab.comparator``). The dataset binding -- sort keys, exclusions, tolerance,
the fixture path, the checksum assertions -- comes entirely from the profile's
``[verification.d9]`` table (``jlab.profiles.verification_for``); this harness
names no column.

Run it as::

    uv run python -m mtbs_fire_analysis.gate_d9 \\
        --scratch-root <dir> [--years 1984 ...] [--skip-run] \\
        [--skip-checksum] [--report <path.json>] [--profile fire|wave0-gate] \\
        [--verification d9|d9_full]

Steps, each logged and recorded in the JSON report:

  0. Load the profile and the ``--verification`` entry (``d9`` gate years, or
     ``d9_full`` all years), resolve the fixture under ``JLAB_ROOT``, and
     ``verify_fixture_hashes`` -- a mismatch aborts (the oracle rotted).
  1. Point ``FIRE_DATA_ROOT`` at the scratch tree, set ``FIRE_CATALOG=1`` and
     the client's PAM proxy, confine caches to the scratch tree, and pin
     ``PROJ_NETWORK=OFF`` (network grid fetches made lon/lat ``inf`` in a dask
     worker) -- ALL before importing ``paths``/``m10`` (paths.py reads env at
     import).
  2. Resolve the catalog-backed inputs and assert each path exists (a miss
     fails here, not as an opaque m10 crash): the named inputs ``m10`` imports,
     AND every input ROLE the profile's ``[inputs]`` table binds (static once,
     temporal per gate year).
  3. Run the registry's ``checksum_assertions`` (GDAL band checksums). Each
     asserts the resolved ``(collection, year)`` item against a golden byte
     located through the client from the ``golden`` sub-table (deprecated
     included), or beside the resolved path via the wave-0 ``golden_member``.
  4. Unless ``--skip-run``: run ``m10 --clear-cache`` per year in scratch as a
     subprocess and assert each output was freshly produced.
  5. Diff each year against the fixture; a mismatch confined to a column the
     registry declares in ``expected_diff`` and staying within its
     ``max_fraction`` is a measured, accepted difference (reported with its
     count/fraction/reason, never hidden), not a failure. Exit 0 iff every year
     is ``ok`` on that basis.
  6. ``--skip-run`` diffs whatever already sits under the scratch ``data_tmp``.

The ``-a``/``--all-columns`` decision (step 4): the golden output schema does
NOT carry the eight extra MTBS columns (``Asmnt_Type``, ``days_since_epoch``,
``dNBR_offst``, ``dNBR_stdDv``, ``IncGreen_T``, ``Low_T``, ``Mod_T``,
``High_T``). In ``m10._build_dataframe_and_save`` those are dropped exactly
when ``drop_extra`` is true, and ``drop_extra = not all_columns``. So the
golden was produced with ``all_columns`` false; to reproduce its column set the
gate runs ``m10`` WITHOUT ``-a``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from jlab import comparator, profiles
from jlab.client import ensure_pam_proxy

REPO_ROOT = Path(__file__).resolve().parents[1]

# The four WUI flavors the sweep resolves per year (matching m10's own set).
_WUI_FLAVORS = ("flag", "class", "bool", "prox")


class GateAbort(Exception):  # noqa: N818 -- "Abort" reads better than "Error"
    """A gate step failed and the run must stop. ``step`` names the step and
    ``detail`` is recorded verbatim in the report; ``extra`` holds any
    structured context (e.g. the failing input name/year)."""

    def __init__(self, step: str, detail: str, **extra):
        super().__init__(f"[{step}] {detail}")
        self.step = step
        self.detail = detail
        self.extra = extra


def _log(msg: str) -> None:
    print(f"[gate-d9] {msg}", flush=True)


# ---------------------------------------------------------------------------
# step 0: profile + fixture + oracle hash check


def _load_profile_and_fixture(profile_id: str, verification_name: str = "d9"):
    """Load the profile and its ``verification_name`` verification (``d9`` or
    ``d9_full``), resolve the fixture dir under ``JLAB_ROOT`` (env, or the repo
    ``.env`` via python-dotenv), and read the fixture ``manifest.json``.
    Returns ``(profile, verification, fixture_dir, manifest)``. Raises
    ``GateAbort`` when unconfigured."""
    load_dotenv(REPO_ROOT / ".env")  # override=False: an exported env var wins
    profile = profiles.load_profile(profile_id)
    verification = profiles.verification_for(profile, verification_name)
    jlab_root = os.environ.get("JLAB_ROOT")
    if not jlab_root:
        raise GateAbort(
            "fixture",
            "JLAB_ROOT is not set (env or the repo .env); cannot resolve the "
            "golden fixture",
        )
    fixture_dir = Path(jlab_root) / verification["fixture"]
    manifest_path = fixture_dir / "manifest.json"
    if not manifest_path.is_file():
        raise GateAbort("fixture", f"no fixture manifest at {manifest_path}")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    return profile, verification, fixture_dir, manifest


# ---------------------------------------------------------------------------
# step 1: environment


def _setup_environment(scratch_root: Path, profile_id: str) -> dict:
    """Create the scratch layout and set the process env so paths.py serves
    catalog inputs into a scratch ``FIRE_DATA_ROOT`` with caches confined to
    the scratch tree. MUST run before importing paths/m10. Returns the recorded
    env subset."""
    scratch_root = Path(scratch_root).resolve()
    (scratch_root / "data").mkdir(parents=True, exist_ok=True)
    (scratch_root / "data_tmp").mkdir(parents=True, exist_ok=True)
    (scratch_root / "logs").mkdir(parents=True, exist_ok=True)
    (scratch_root / "cache").mkdir(parents=True, exist_ok=True)

    os.environ["FIRE_DATA_ROOT"] = str(scratch_root)
    os.environ["FIRE_CATALOG"] = "1"
    os.environ["FIRE_CATALOG_PROFILE"] = profile_id
    # Confine PAM side files and the client blob cache to the scratch tree so a
    # gate run never writes outside it (the store stays read-only).
    os.environ["XDG_CACHE_HOME"] = str(scratch_root / "cache")
    # Keep PROJ offline. With PROJ_NETWORK=ON (the default in some shells) the
    # NAD83->WGS84 step behind m10's lon/lat has ~48 grid-based candidate
    # pipelines fetched from cdn.proj.org; a fetch that fails inside a dask
    # worker yields inf coordinates, and one that succeeds shifts lon/lat by
    # ~1e-5 deg against the offline pipeline. The gate must not depend on the
    # network, so pin the single offline pipeline (the lon/lat atol absorbs the
    # ~1e-5 deg difference against a golden produced with the network on).
    os.environ["PROJ_NETWORK"] = "OFF"
    pam = ensure_pam_proxy()

    return {
        "FIRE_DATA_ROOT": os.environ["FIRE_DATA_ROOT"],
        "FIRE_CATALOG": os.environ["FIRE_CATALOG"],
        "FIRE_CATALOG_PROFILE": os.environ["FIRE_CATALOG_PROFILE"],
        "XDG_CACHE_HOME": os.environ["XDG_CACHE_HOME"],
        "PROJ_NETWORK": os.environ["PROJ_NETWORK"],
        "GDAL_PAM_PROXY_DIR": os.environ.get("GDAL_PAM_PROXY_DIR"),
        "GDAL_PAM_PROXY_DIR_created": str(pam) if pam else None,
        "JLAB_ROOT": os.environ.get("JLAB_ROOT"),
        "JLAB_API_URL": os.environ.get("JLAB_API_URL"),
    }


def _import_paths():
    """Import the consumer's paths module (env must already be set). A resolve
    failure at import (an input the store cannot serve) becomes a resolution
    ``GateAbort`` rather than an opaque import traceback."""
    try:
        from mtbs_fire_analysis.pipeline import paths
    except Exception as e:  # noqa: BLE001 -- reported as a resolution failure
        raise GateAbort(
            "resolution",
            f"importing paths.py failed while resolving catalog inputs: {e!r}",
        ) from e
    return paths


# ---------------------------------------------------------------------------
# step 2: resolution sweep


def _iter_sweep_targets(paths_mod, years):
    """Yield ``(name, year, path)`` for every catalog-backed input to check:
    the nine input constants once, then per year the MTBS/NLCD
    (self, pre, post)/WUI rasters and the perims ``dse_{year}.tif``."""
    consts = [
        ("ASPECT_PATH", paths_mod.ASPECT_PATH),
        ("ECO_REGIONS_PATH", paths_mod.ECO_REGIONS_PATH),
        ("ELEVATION_PATH", paths_mod.ELEVATION_PATH),
        ("HEX_GRID_PATH", paths_mod.HEX_GRID_PATH),
        ("NLCD_MODE_RASTER_PATH", paths_mod.NLCD_MODE_RASTER_PATH),
        ("PERIMS_PATH", paths_mod.PERIMS_PATH),
        ("SLOPE_PATH", paths_mod.SLOPE_PATH),
        ("STATES_PATH", paths_mod.STATES_PATH),
        ("PERIMS_RASTERS_PATH", paths_mod.PERIMS_RASTERS_PATH),
    ]
    for name, p in consts:
        yield name, None, Path(p)
    for y in years:
        yield (
            "get_mtbs_raster_path",
            y,
            Path(paths_mod.get_mtbs_raster_path(y, "CONUS")),
        )
        yield (
            "get_nlcd_raster_path",
            y,
            Path(paths_mod.get_nlcd_raster_path(y)),
        )
        yield (
            "get_nlcd_raster_path(pre)",
            y,
            Path(paths_mod.get_nlcd_raster_path(max(y - 1, 1984))),
        )
        yield (
            "get_nlcd_raster_path(post)",
            y,
            Path(paths_mod.get_nlcd_raster_path(y + 1)),
        )
        for flavor in _WUI_FLAVORS:
            yield (
                f"get_wui_flavor_path[{flavor}]",
                y,
                Path(paths_mod.get_wui_flavor_path(y, flavor)),
            )
        yield "dse", y, Path(paths_mod.PERIMS_RASTERS_PATH) / f"dse_{y}.tif"


def _resolution_sweep(paths_mod, years):
    """Resolve every catalog-backed input for every gate year and assert each
    path exists. Returns the list of resolved records; raises ``GateAbort`` on
    the first path that does not exist (carrying the failing name/year)."""
    resolved = []
    for name, year, path in _iter_sweep_targets(paths_mod, years):
        exists = path.exists()
        size = path.stat().st_size if (exists and path.is_file()) else None
        resolved.append(
            {
                "name": name,
                "year": year,
                "path": str(path),
                "size": size,
                "exists": exists,
            }
        )
        if not exists:
            raise GateAbort(
                "resolution",
                f"{name} (year={year}) resolved to a path that does not "
                f"exist: {path}",
                name=name,
                year=year,
                resolved=resolved,
            )
    return resolved


def _role_sweep(profile, years):
    """Resolve every ``[inputs]`` role the profile declares through the adapter
    and assert each path exists. This is the profile-contract half of step 2:
    the name-based sweep above checks what ``m10`` imports; this checks that
    every role the profile binds resolves under the active profile.

    A role is treated as STATIC when ``resolve(role)`` with no year yields a
    single Item; otherwise (zero or many) it is TEMPORAL and resolved per gate
    year -- no cadence is hard-coded. Returns the list of records
    ``{role, collection, year, path, exists}``; raises ``GateAbort`` on the
    first path that does not exist (carrying the failing role/year)."""
    from mtbs_fire_analysis.pipeline import catalog_paths

    inputs = profile.get("inputs", {}) if profile else {}
    records = []
    for role in inputs:
        collection = profiles.input_collection(profile, role)
        try:
            targets = [(None, Path(catalog_paths.resolve(role)))]
        except catalog_paths.CatalogResolveError:
            targets = [
                (y, Path(catalog_paths.resolve(role, year=y))) for y in years
            ]
        for year, path in targets:
            exists = path.exists()
            rec = {
                "role": role,
                "collection": collection,
                "year": year,
                "path": str(path),
                "exists": exists,
            }
            records.append(rec)
            if not exists:
                raise GateAbort(
                    "resolution",
                    f"input role {role!r} (collection {collection!r}, "
                    f"year={year}) resolved to a path that does not exist: "
                    f"{path}",
                    role=role,
                    collection=collection,
                    year=year,
                    resolved=records,
                )
    return records


# ---------------------------------------------------------------------------
# step 3: checksum assertions


def _localize_one(cli, collection, *, year=None, include_deprecated=None):
    """Resolve ``(collection, year)`` through the client to exactly one Item
    and localize it. Raises ``GateAbort`` when zero or many Items match."""
    items = cli.find(
        collection, year=year, include_deprecated=include_deprecated
    )
    if len(items) != 1:
        ids = [it.get("id") for it in items]
        raise GateAbort(
            "checksum",
            f"expected exactly one item for {collection!r} year={year!r}, "
            f"got {len(ids)}: {ids}",
            collection=collection,
            year=year,
        )
    return Path(cli.localize(items[0]))


def _golden_path(cli, entry, resolved):
    """Locate the golden byte for a checksum assertion. Returns
    ``(golden_path, form)``. Prefers the wave-1 ``golden`` sub-table (looked up
    through the client with deprecated Items included); falls back to the
    wave-0 ``golden_member`` form (a file beside the resolved path)."""
    golden_spec = entry.get("golden")
    if golden_spec is not None:
        g_col = golden_spec["collection"]
        g_year = golden_spec.get("year", entry["year"])
        g_member = golden_spec["member"]
        g_local = _localize_one(
            cli, g_col, year=g_year, include_deprecated=True
        )
        if g_local.name == g_member:
            golden = g_local
        else:
            art_dir = g_local if g_local.is_dir() else g_local.parent
            golden = art_dir / g_member
        return golden, "golden"
    member = entry["golden_member"]
    return resolved.parent / member, "golden_member"


def _run_checksum_assertions(verification):
    """Run the registry's band-checksum assertions: the item the profile
    resolves for ``(collection, year)`` must have the same GDAL band checksum
    as the golden byte. The golden is located from the ``golden`` sub-table
    (looked up through the client) or the wave-0 ``golden_member`` form (a file
    beside the resolved path). Returns the list of records; raises
    ``GateAbort`` on a mismatch."""
    from mtbs_fire_analysis.pipeline import catalog_paths

    cli = catalog_paths.client()
    records = []
    for entry in verification.get("checksum_assertions", []):
        collection = entry["collection"]
        year = entry["year"]
        resolved = _localize_one(cli, collection, year=year)
        golden, form = _golden_path(cli, entry, resolved)
        if not golden.is_file():
            raise GateAbort(
                "checksum",
                f"golden byte {golden} (form {form}) not found for "
                f"{collection} year={year}",
                collection=collection,
                year=year,
            )
        cs_resolved = comparator.band_checksum(resolved)
        cs_golden = comparator.band_checksum(golden)
        rec = {
            "collection": collection,
            "year": year,
            "resolved": str(resolved),
            "golden_member": str(golden),
            "golden_form": form,
            "checksum_resolved": cs_resolved,
            "checksum_golden": cs_golden,
            "match": cs_resolved == cs_golden,
        }
        records.append(rec)
        _log(
            f"checksum {collection} year={year} ({form}): "
            f"resolved={cs_resolved} golden={cs_golden} -> "
            f"{'OK' if rec['match'] else 'MISMATCH'}"
        )
        if not rec["match"]:
            raise GateAbort(
                "checksum",
                f"band checksum {cs_resolved} != golden {cs_golden} for "
                f"{collection} year={year} (the policy floor may be wrong)",
                **rec,
            )
    return records


# ---------------------------------------------------------------------------
# step 4: run m10 + fresh-output assertions


def _year_output_dir(scratch_root: Path, verification, year: int) -> Path:
    pattern = verification["output_dir_pattern"]
    return Path(scratch_root) / "data_tmp" / pattern.format(year=year)


def _has_parquet(out_dir: Path) -> bool:
    return out_dir.is_dir() and any(out_dir.glob("*.parquet"))


def _clean_output_dir(out_dir: Path) -> None:
    """Remove any pre-existing per-year output under the harness's own scratch
    data_tmp so m10 rebuilds it fresh this run. The gate owns this scratch
    tree, so clearing a stale dir (a reused scratch, or a prior
    partial/interrupted run) is safe and keeps the gate re-runnable; freshness
    is still enforced afterwards by ``_assert_produced`` (mtime >= t0). m10
    itself also ``--clear-cache``s the dir, so this is belt-and-suspenders and
    independent of m10's internals."""
    if out_dir.exists():
        _log(f"removing stale output before rebuild: {out_dir}")
        shutil.rmtree(out_dir)


def _assert_produced(out_dir: Path, t0: float) -> None:
    """The per-year output must exist, hold >=1 parquet part, and be newer than
    ``t0`` (freshly produced this run; outputs are not byte-deterministic so
    bytes are never compared here)."""
    if not out_dir.is_dir():
        raise GateAbort(
            "run", f"m10 produced no output directory at {out_dir}"
        )
    parts = sorted(out_dir.glob("*.parquet"))
    if not parts:
        raise GateAbort(
            "run", f"m10 output {out_dir} holds no *.parquet parts"
        )
    mtime = out_dir.stat().st_mtime
    if mtime < t0:
        raise GateAbort(
            "run",
            f"m10 output {out_dir} mtime {mtime} predates the run start {t0}; "
            "it was not freshly produced",
        )


def _run_m10_subprocess(year: int, log_path: Path) -> int:
    """Run ``m10 --clear-cache`` for one year as a subprocess (env inherited,
    cwd = repo root), streaming its stdout+stderr to ``log_path``. Returns the
    exit code."""
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "mtbs_fire_analysis.pipeline.m10_data_extract",
        "--min_year",
        str(year),
        "--max_year",
        str(year),
        "--clear-cache",
    ]
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode


def _run_m10_years(scratch_root: Path, verification, years):
    """Run m10 for each year with the fresh-output guarantees. Returns a
    per-year record dict; raises ``GateAbort`` on any failure."""
    logs_dir = Path(scratch_root) / "logs"
    records = {}
    for year in years:
        out_dir = _year_output_dir(scratch_root, verification, year)
        _clean_output_dir(out_dir)
        t0 = time.time()
        log_path = logs_dir / f"m10_{year}.log"
        _log(f"running m10 for {year} (log: {log_path})")
        rc = _run_m10_subprocess(year, log_path)
        if rc != 0:
            raise GateAbort(
                "run",
                f"m10 for {year} exited {rc}; see {log_path}",
                year=year,
                returncode=rc,
                log=str(log_path),
            )
        _assert_produced(out_dir, t0)
        parts = sorted(out_dir.glob("*.parquet"))
        records[str(year)] = {
            "t0": t0,
            "output_dir": str(out_dir),
            "parts": len(parts),
            "mtime": out_dir.stat().st_mtime,
            "log": str(log_path),
        }
        _log(f"m10 {year}: {len(parts)} parts produced")
    return records


# ---------------------------------------------------------------------------
# step 5: diff


def _diff_year(scratch_root: Path, fixture_dir: Path, verification, year: int):
    """Compare the scratch output for ``year`` against the fixture golden with
    the registry's sort keys / exclusions / tolerances. Returns a
    ``CompareReport``."""
    pattern = verification["output_dir_pattern"]
    actual_dir = _year_output_dir(scratch_root, verification, year)
    expected_dir = Path(fixture_dir) / "outputs" / pattern.format(year=year)
    actual = comparator.read_parts(actual_dir)
    expected = comparator.read_parts(expected_dir)
    return comparator.compare_tables(
        actual,
        expected,
        sort_keys=verification["sort_keys"],
        exclude=verification.get("exclude", ()),
        atol=verification.get("atol") or None,
    )


def _diff_all(
    scratch_root: Path, fixture_dir: Path, verification, years, skip_run
):
    """Diff each gate year that has scratch output. Returns
    ``(ok, per_year_dict, message)``. With no outputs present the message is
    the explicit smoke-run outcome and ``ok`` is False.

    The registry's optional ``expected_diff`` table (``{column: {reason,
    max_fraction}}``) records measured, accepted per-column differences. A year
    whose only mismatches are expected columns within their bound is OK; the
    per-year record still carries the raw comparator ``per_column_mismatch``
    plus the ``expected`` and ``unexpected`` column splits
    (count/fraction/reason), and ``strict_ok`` (the comparator's exact verdict)
    beside the effective ``ok``."""
    present = [
        y
        for y in years
        if _has_parquet(_year_output_dir(scratch_root, verification, y))
    ]
    if not present:
        data_tmp = Path(scratch_root) / "data_tmp"
        return (
            False,
            {},
            f"no outputs to diff under {data_tmp}; run the gate without "
            "--skip-run to produce them first",
        )
    expected_diff = verification.get("expected_diff", {})
    diff_years = present if skip_run else years
    per_year = {}
    ok = True
    for year in diff_years:
        out_dir = _year_output_dir(scratch_root, verification, year)
        if not _has_parquet(out_dir):
            per_year[str(year)] = {
                "ok": False,
                "error": f"no output at {out_dir}",
            }
            ok = False
            _log(f"diff {year}: FAIL (no output)")
            continue
        report = _diff_year(scratch_root, fixture_dir, verification, year)
        classified = comparator.classify_mismatches(
            report, expected_diff, report.rows_expected
        )
        year_expected = {c: r for c, r in classified.items() if r["expected"]}
        year_unexpected = {
            c: r for c, r in classified.items() if not r["expected"]
        }
        year_ok = (
            report.rows_actual == report.rows_expected
            and not report.missing_columns
            and not report.extra_columns
            and not year_unexpected
        )
        rec = report.to_dict()
        rec["strict_ok"] = report.ok
        rec["ok"] = year_ok
        rec["expected"] = year_expected
        rec["unexpected"] = year_unexpected
        per_year[str(year)] = rec
        ok = ok and year_ok
        if year_ok and year_expected:
            detail = ", ".join(
                f"{c}={r['count']} rows, {r['fraction'] * 100:.2f}%, "
                f"{r['reason']}"
                for c, r in year_expected.items()
            )
            _log(f"diff {year}: OK (expected: {detail})")
        else:
            _log(f"diff {year}: {report.summary()}")
        # Drop the year's tables before the next (one pair in RAM at a time).
        del report
    return (
        ok,
        per_year,
        "diffed years: " + ", ".join(str(y) for y in diff_years),
    )


# ---------------------------------------------------------------------------
# orchestration


def run_gate(args) -> int:
    """Run the gate end to end. Writes the JSON report (if ``--report``) and
    returns the process exit code (0 iff every step passed and every year
    diffed clean)."""
    scratch_root = Path(args.scratch_root).resolve()
    report: dict = {
        "profile": args.profile,
        "verification": args.verification,
        "scratch_root": str(scratch_root),
        "skip_run": bool(args.skip_run),
        "skip_checksum": bool(args.skip_checksum),
        "steps": {},
        "ok": False,
        "exit_code": 1,
    }

    try:
        # step 0
        _log(
            f"loading profile {args.profile!r} and its "
            f"{args.verification!r} verification"
        )
        profile, verification, fixture_dir, manifest = (
            _load_profile_and_fixture(args.profile, args.verification)
        )
        years = list(args.years) if args.years else list(verification["years"])
        report["gate_years"] = years
        report["fixture_dir"] = str(fixture_dir)
        _log(f"verifying fixture hashes under {fixture_dir}")
        mismatches = comparator.verify_fixture_hashes(manifest, fixture_dir)
        report["steps"]["fixture_hashes"] = {
            "ok": not mismatches,
            "checked": len(manifest.get("outputs", {})),
            "mismatches": mismatches,
        }
        if mismatches:
            raise GateAbort(
                "fixture",
                f"the frozen golden has rotted: {len(mismatches)} output(s) "
                f"no longer match ({mismatches[:3]}...)",
            )
        _log(
            f"fixture ok: {len(manifest.get('outputs', {}))} outputs verified"
        )

        # step 1
        env = _setup_environment(scratch_root, args.profile)
        report["steps"]["environment"] = env
        _log(f"environment set; FIRE_DATA_ROOT={env['FIRE_DATA_ROOT']}")
        paths_mod = _import_paths()

        # step 2
        _log("resolving all catalog-backed inputs for every gate year")
        resolved = _resolution_sweep(paths_mod, years)
        role_resolved = _role_sweep(profile, years)
        report["steps"]["resolution"] = {
            "ok": True,
            "resolved": resolved,
            "roles": role_resolved,
        }
        _log(
            f"resolution ok: {len(resolved)} named inputs + "
            f"{len(role_resolved)} profile roles resolved and present"
        )

        # step 3
        if args.skip_checksum:
            report["steps"]["checksums"] = {"skipped": True}
            _log("checksum assertions skipped (--skip-checksum)")
        else:
            _log("running band-checksum assertions")
            records = _run_checksum_assertions(verification)
            report["steps"]["checksums"] = {"ok": True, "assertions": records}

        # step 4
        if args.skip_run:
            report["steps"]["run"] = {"skipped": True}
            _log("m10 run skipped (--skip-run); diffing existing outputs")
        else:
            _log(f"running m10 over {years}")
            run_records = _run_m10_years(scratch_root, verification, years)
            report["steps"]["run"] = {"skipped": False, "years": run_records}

        # step 5
        ok, per_year, message = _diff_all(
            scratch_root, fixture_dir, verification, years, args.skip_run
        )
        report["steps"]["diff"] = {
            "ok": ok,
            "years": per_year,
            "message": message,
        }
        _log(f"diff: {message}")
        report["ok"] = ok
        report["exit_code"] = 0 if ok else 1

    except GateAbort as abort:
        report["steps"].setdefault(abort.step, {})
        report["steps"][abort.step].update(
            {"ok": False, "error": abort.detail}
        )
        if abort.extra:
            report["steps"][abort.step]["context"] = _jsonable(abort.extra)
        report["ok"] = False
        report["exit_code"] = 1
        report["abort"] = {"step": abort.step, "detail": abort.detail}
        _log(f"ABORT {abort}")

    finally:
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as fh:
                json.dump(report, fh, indent=2, default=str)
            _log(f"report written to {report_path}")

    _log(f"exit {report['exit_code']} (ok={report['ok']})")
    return report["exit_code"]


def _jsonable(obj):
    """Best-effort JSON-safe copy (paths -> str), for abort context."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return json.loads(json.dumps(obj, default=str))


def _get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--scratch-root",
        required=True,
        help="Scratch FIRE_DATA_ROOT for the gate (data/, data_tmp/, logs/, "
        "cache/ are created under it).",
    )
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Gate years (default: the profile verification's years).",
    )
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip m10; diff whatever already sits under scratch data_tmp.",
    )
    p.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip the band-checksum assertions (each is minutes on a full "
        "CONUS raster).",
    )
    p.add_argument(
        "--report",
        default=None,
        help="Write the JSON report to this path.",
    )
    p.add_argument(
        "--profile",
        default="fire",
        help=(
            "Profile id carrying the [verification.*] tables "
            "(fire = the wave-1 products; wave0-gate = the wave-0 imports)."
        ),
    )
    p.add_argument(
        "--verification",
        default="d9",
        choices=["d9", "d9_full"],
        help="Which registered golden-diff verification to run: 'd9' (the "
        "gate years, default) or 'd9_full' (all years 1984-2022).",
    )
    return p


def main(argv=None) -> int:
    args = _get_parser().parse_args(argv)
    return run_gate(args)


if __name__ == "__main__":
    sys.exit(main())
