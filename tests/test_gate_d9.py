"""Unit tests for the D9 gate harness (mtbs_fire_analysis.gate_d9).

All pieces are faked/mocked: no m10 run, no store reads or writes. The real
comparator (jlab.comparator) diffs tiny synthetic parquet under tmp_path so the
exit-code paths are exercised end to end."""

from __future__ import annotations

import os
import types

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mtbs_fire_analysis import gate_d9


@pytest.fixture(autouse=True)
def _env_guard():
    """Restore os.environ around each test (gate_d9 mutates it directly)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _verification():
    return {
        "kind": "golden-diff",
        "fixture": ".fixtures/x",
        "years": [1984],
        "sort_keys": ["k"],
        "exclude": ["__null_dask_index__"],
        "atol": {"lon": 1e-4},
        "output_dir_pattern": "mtbs_CONUS_{year}",
        "checksum_assertions": [],
    }


def _write_parquet(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _fake_paths(tmp_path, years):
    """A stand-in for the consumer paths module: every catalog-backed name
    resolves to an existing file/dir under tmp_path."""
    base = tmp_path / "store"
    base.mkdir(parents=True, exist_ok=True)

    def _f(name):
        p = base / name
        p.write_bytes(b"x")
        return p

    perims_dir = base / "perims_rasters"
    perims_dir.mkdir(exist_ok=True)
    for y in years:
        (perims_dir / f"dse_{y}.tif").write_bytes(b"x")

    ns = types.SimpleNamespace(
        ASPECT_PATH=_f("aspect.tif"),
        ECO_REGIONS_PATH=_f("eco.gpkg"),
        ELEVATION_PATH=_f("dem.tif"),
        HEX_GRID_PATH=_f("hex.gpkg"),
        NLCD_MODE_RASTER_PATH=_f("nlcd_mode.tif"),
        PERIMS_PATH=_f("perims.gpkg"),
        SLOPE_PATH=_f("slope.tif"),
        STATES_PATH=_f("states.shp"),
        PERIMS_RASTERS_PATH=perims_dir,
        get_mtbs_raster_path=lambda y, aoi: _f(f"mtbs_{aoi}_{y}.tif"),
        get_nlcd_raster_path=lambda y: _f(f"nlcd_{y}.tif"),
        get_wui_flavor_path=lambda y, flavor: _f(f"wui_{flavor}_{y}.tif"),
    )
    return ns


def _args(**kw):
    argv = ["--scratch-root", str(kw.pop("scratch_root"))]
    if kw.get("skip_run"):
        argv.append("--skip-run")
    if kw.get("skip_checksum"):
        argv.append("--skip-checksum")
    if kw.get("report"):
        argv += ["--report", str(kw["report"])]
    return gate_d9._get_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# step 1: scratch layout + env set before import


def test_setup_environment_creates_layout_and_sets_env(tmp_path):
    env = gate_d9._setup_environment(tmp_path / "scratch", "wave0-gate")
    scratch = tmp_path / "scratch"
    for sub in ("data", "data_tmp", "logs", "cache"):
        assert (scratch / sub).is_dir()
    assert os.environ["FIRE_DATA_ROOT"] == str(scratch.resolve())
    assert os.environ["FIRE_CATALOG"] == "1"
    assert os.environ["FIRE_CATALOG_PROFILE"] == "wave0-gate"
    assert os.environ["XDG_CACHE_HOME"] == str((scratch / "cache").resolve())
    assert os.environ.get("GDAL_PAM_PROXY_DIR")
    # PROJ must stay offline: network grid fetches made lon/lat inf in a dask
    # worker during the first full gate run.
    assert os.environ["PROJ_NETWORK"] == "OFF"
    assert env["PROJ_NETWORK"] == "OFF"
    assert env["FIRE_DATA_ROOT"] == str(scratch.resolve())


def test_env_is_set_before_paths_import(tmp_path, monkeypatch):
    # A fake _import_paths asserts the env was already set when it runs.
    seen = {}

    def fake_import_paths():
        seen["FIRE_DATA_ROOT"] = os.environ.get("FIRE_DATA_ROOT")
        return _fake_paths(tmp_path, [1984])

    monkeypatch.setattr(
        gate_d9,
        "_load_profile_and_fixture",
        lambda pid: ({}, _verification(), tmp_path / "fix", {"outputs": {}}),
    )
    monkeypatch.setattr(gate_d9, "_import_paths", fake_import_paths)

    scratch = tmp_path / "scratch"
    rc = gate_d9.run_gate(
        _args(scratch_root=scratch, skip_run=True, skip_checksum=True)
    )
    assert seen["FIRE_DATA_ROOT"] == str(scratch.resolve())
    assert rc == 1  # empty data_tmp -> no outputs to diff


# ---------------------------------------------------------------------------
# step 2: resolution-sweep failure aborts with name/year


def test_resolution_sweep_ok(tmp_path):
    paths = _fake_paths(tmp_path, [1984, 2005])
    resolved = gate_d9._resolution_sweep(paths, [1984, 2005])
    names = {r["name"] for r in resolved}
    assert "ASPECT_PATH" in names and "get_mtbs_raster_path" in names
    assert "dse" in names
    assert all(r["exists"] for r in resolved)


def test_resolution_sweep_missing_aborts_with_name_year(tmp_path):
    paths = _fake_paths(tmp_path, [1984])
    # break MTBS for the year: return a path that does not exist
    paths.get_mtbs_raster_path = lambda y, aoi: tmp_path / f"missing_{y}.tif"
    with pytest.raises(gate_d9.GateAbort) as e:
        gate_d9._resolution_sweep(paths, [1984])
    assert e.value.step == "resolution"
    assert e.value.extra["name"] == "get_mtbs_raster_path"
    assert e.value.extra["year"] == 1984


# ---------------------------------------------------------------------------
# step 4: fresh-output assertion logic


def test_clean_output_dir_removes_preexisting(tmp_path):
    # A stale per-year output (a reused scratch or a prior interrupted run) is
    # removed so m10 rebuilds fresh -- the gate must be re-runnable.
    out = tmp_path / "data_tmp" / "mtbs_CONUS_1984"
    out.mkdir(parents=True)
    (out / "part.0.parquet").write_bytes(b"stale")
    gate_d9._clean_output_dir(out)
    assert not out.exists()
    # a non-existent dir is a no-op
    gate_d9._clean_output_dir(tmp_path / "data_tmp" / "mtbs_CONUS_2005")


def test_assert_produced_requires_parquet_and_fresh_mtime(tmp_path):
    out = tmp_path / "mtbs_CONUS_1984"
    out.mkdir()
    # no parquet -> abort
    with pytest.raises(gate_d9.GateAbort):
        gate_d9._assert_produced(out, t0=0.0)
    # a parquet, but mtime older than t0 -> abort (not freshly produced)
    _write_parquet(out / "part.0.parquet", pa.table({"k": [1]}))
    import time

    with pytest.raises(gate_d9.GateAbort):
        gate_d9._assert_produced(out, t0=time.time() + 60)
    # a parquet newer than t0 -> ok
    gate_d9._assert_produced(out, t0=0.0)


def test_run_m10_years_clears_stale_and_rebuilds(tmp_path, monkeypatch):
    # A stale output from a prior/interrupted run must not abort the gate: it
    # is cleared and m10 rebuilds it fresh (re-runnable on a reused scratch).
    scratch = tmp_path / "scratch"
    out = scratch / "data_tmp" / "mtbs_CONUS_1984"
    out.mkdir(parents=True)
    (out / "part.0.parquet").write_bytes(b"stale")
    (out / "stale-only.parquet").write_bytes(b"stale")
    (scratch / "logs").mkdir(parents=True)

    def fake_subprocess(year, log_path):
        # simulate m10: the dir was cleared, produce a fresh single part. A
        # real m10 run takes minutes; stamp the output clearly after t0 so the
        # freshness check is not hostage to sub-millisecond clock granularity.
        import time

        _write_parquet(out / "part.0.parquet", pa.table({"k": [1]}))
        future = time.time() + 5
        os.utime(out, (future, future))
        return 0

    monkeypatch.setattr(gate_d9, "_run_m10_subprocess", fake_subprocess)
    records = gate_d9._run_m10_years(scratch, _verification(), [1984])
    assert records["1984"]["parts"] == 1
    # the stale-only part is gone (dir was wiped before the rebuild)
    assert not (out / "stale-only.parquet").exists()
    assert (out / "part.0.parquet").exists()


# ---------------------------------------------------------------------------
# step 5 + exit codes: smoke (no outputs) and a clean diff


def _run_with_fakes(tmp_path, monkeypatch, *, seed_outputs, report=None):
    verification = _verification()
    fixture_dir = tmp_path / "fixture"
    # golden output for 1984
    golden = pa.table(
        {"k": [1, 2, 3], "v": [10, 20, 30], "lon": [1.0, 2.0, 3.0]}
    )
    _write_parquet(
        fixture_dir / "outputs" / "mtbs_CONUS_1984" / "part.0.parquet", golden
    )

    monkeypatch.setattr(
        gate_d9,
        "_load_profile_and_fixture",
        lambda pid: ({}, verification, fixture_dir, {"outputs": {}}),
    )
    monkeypatch.setattr(
        gate_d9, "_import_paths", lambda: _fake_paths(tmp_path, [1984])
    )

    scratch = tmp_path / "scratch"
    if seed_outputs:
        # identical output under the scratch data_tmp (lon within atol)
        actual = pa.table(
            {"k": [3, 1, 2], "v": [30, 10, 20], "lon": [3.00001, 1.0, 2.0]}
        )
        _write_parquet(
            scratch / "data_tmp" / "mtbs_CONUS_1984" / "part.0.parquet", actual
        )
    return gate_d9.run_gate(
        _args(
            scratch_root=scratch,
            skip_run=True,
            skip_checksum=True,
            report=report,
        )
    )


def test_smoke_no_outputs_exits_nonzero_with_message(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    rc = _run_with_fakes(
        tmp_path, monkeypatch, seed_outputs=False, report=report_path
    )
    assert rc == 1
    import json

    report = json.loads(report_path.read_text())
    # report shape
    assert report["exit_code"] == 1 and report["ok"] is False
    assert report["gate_years"] == [1984]
    assert report["steps"]["fixture_hashes"]["ok"] is True
    assert report["steps"]["resolution"]["ok"] is True
    assert report["steps"]["checksums"] == {"skipped": True}
    assert report["steps"]["run"] == {"skipped": True}
    diff = report["steps"]["diff"]
    assert diff["ok"] is False
    assert "no outputs to diff" in diff["message"]


def test_clean_diff_exits_zero(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    rc = _run_with_fakes(
        tmp_path, monkeypatch, seed_outputs=True, report=report_path
    )
    assert rc == 0
    import json

    report = json.loads(report_path.read_text())
    assert report["ok"] is True and report["exit_code"] == 0
    year = report["steps"]["diff"]["years"]["1984"]
    assert year["ok"] is True
    assert year["per_column_mismatch"] == {"k": 0, "v": 0, "lon": 0}


def test_dirty_diff_exits_nonzero(tmp_path, monkeypatch):
    verification = _verification()
    fixture_dir = tmp_path / "fixture"
    golden = pa.table({"k": [1, 2], "v": [10, 20], "lon": [1.0, 2.0]})
    _write_parquet(
        fixture_dir / "outputs" / "mtbs_CONUS_1984" / "part.0.parquet", golden
    )
    monkeypatch.setattr(
        gate_d9,
        "_load_profile_and_fixture",
        lambda pid: ({}, verification, fixture_dir, {"outputs": {}}),
    )
    monkeypatch.setattr(
        gate_d9, "_import_paths", lambda: _fake_paths(tmp_path, [1984])
    )
    scratch = tmp_path / "scratch"
    dirty = pa.table({"k": [1, 2], "v": [10, 99], "lon": [1.0, 2.0]})
    _write_parquet(
        scratch / "data_tmp" / "mtbs_CONUS_1984" / "part.0.parquet", dirty
    )
    rc = gate_d9.run_gate(
        _args(scratch_root=scratch, skip_run=True, skip_checksum=True)
    )
    assert rc == 1


# ---------------------------------------------------------------------------
# fixture-rot abort


def test_fixture_rot_aborts(tmp_path, monkeypatch):
    verification = _verification()
    fixture_dir = tmp_path / "fixture"
    manifest = {
        "outputs": {
            "outputs/mtbs_CONUS_1984/part.0.parquet": {
                "sha256": "0" * 64,
                "size": 1,
            }
        }
    }
    monkeypatch.setattr(
        gate_d9,
        "_load_profile_and_fixture",
        lambda pid: ({}, verification, fixture_dir, manifest),
    )
    # _import_paths should never be reached; make it explode if it is.
    monkeypatch.setattr(
        gate_d9,
        "_import_paths",
        lambda: (_ for _ in ()).throw(AssertionError("reached import")),
    )
    rc = gate_d9.run_gate(
        _args(
            scratch_root=tmp_path / "scratch",
            skip_run=True,
            skip_checksum=True,
        )
    )
    assert rc == 1
