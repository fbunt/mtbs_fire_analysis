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
    if kw.get("verification"):
        argv += ["--verification", str(kw["verification"])]
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
        lambda pid, vname: (
            {},
            _verification(),
            tmp_path / "fix",
            {"outputs": {}},
        ),
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
        lambda pid, vname: ({}, verification, fixture_dir, {"outputs": {}}),
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
        lambda pid, vname: ({}, verification, fixture_dir, {"outputs": {}}),
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
# step 5: expected_diff classification in the diff step


def _seed_diff_pair(tmp_path, actual_tbl, golden_tbl, year=1984):
    """Seed a scratch output and a fixture golden for one year; return
    (scratch_root, fixture_dir)."""
    scratch = tmp_path / "scratch"
    fixture = tmp_path / "fixture"
    _write_parquet(
        scratch / "data_tmp" / f"mtbs_CONUS_{year}" / "part.0.parquet",
        actual_tbl,
    )
    _write_parquet(
        fixture / "outputs" / f"mtbs_CONUS_{year}" / "part.0.parquet",
        golden_tbl,
    )
    return scratch, fixture


def test_diff_all_expected_column_within_bound_is_ok(tmp_path):
    v = _verification()
    v["expected_diff"] = {
        "nlcd_mode": {
            "reason": "known 1985 double-count",
            "max_fraction": 0.5,
        }
    }
    # k/lon match; nlcd_mode differs on 1 of 4 rows (0.25 <= 0.5).
    actual = pa.table(
        {
            "k": [1, 2, 3, 4],
            "lon": [1.0, 2.0, 3.0, 4.0],
            "nlcd_mode": [10, 20, 30, 99],
        }
    )
    golden = pa.table(
        {
            "k": [1, 2, 3, 4],
            "lon": [1.0, 2.0, 3.0, 4.0],
            "nlcd_mode": [10, 20, 30, 40],
        }
    )
    scratch, fixture = _seed_diff_pair(tmp_path, actual, golden)
    ok, per_year, _msg = gate_d9._diff_all(
        scratch, fixture, v, [1984], skip_run=True
    )
    assert ok is True
    y = per_year["1984"]
    assert y["ok"] is True
    assert y["strict_ok"] is False  # the comparator sees a raw mismatch
    assert "nlcd_mode" in y["expected"]
    assert y["expected"]["nlcd_mode"]["count"] == 1
    assert y["expected"]["nlcd_mode"]["fraction"] == 0.25
    assert y["expected"]["nlcd_mode"]["reason"] == "known 1985 double-count"
    assert y["unexpected"] == {}


def test_diff_all_expected_column_over_bound_fails(tmp_path):
    v = _verification()
    v["expected_diff"] = {
        "nlcd_mode": {"reason": "known", "max_fraction": 0.1}
    }
    # nlcd_mode differs on 2 of 4 rows (0.5 > 0.1) -> unexpected -> FAIL.
    actual = pa.table(
        {
            "k": [1, 2, 3, 4],
            "lon": [1.0, 2.0, 3.0, 4.0],
            "nlcd_mode": [99, 99, 30, 40],
        }
    )
    golden = pa.table(
        {
            "k": [1, 2, 3, 4],
            "lon": [1.0, 2.0, 3.0, 4.0],
            "nlcd_mode": [10, 20, 30, 40],
        }
    )
    scratch, fixture = _seed_diff_pair(tmp_path, actual, golden)
    ok, per_year, _msg = gate_d9._diff_all(
        scratch, fixture, v, [1984], skip_run=True
    )
    assert ok is False
    y = per_year["1984"]
    assert y["ok"] is False
    assert "nlcd_mode" in y["unexpected"]
    assert y["expected"] == {}


def test_diff_all_undeclared_mismatch_fails(tmp_path):
    v = _verification()  # no expected_diff at all
    actual = pa.table({"k": [1, 2], "lon": [1.0, 2.0], "nlcd_mode": [99, 20]})
    golden = pa.table({"k": [1, 2], "lon": [1.0, 2.0], "nlcd_mode": [10, 20]})
    scratch, fixture = _seed_diff_pair(tmp_path, actual, golden)
    ok, per_year, _msg = gate_d9._diff_all(
        scratch, fixture, v, [1984], skip_run=True
    )
    assert ok is False
    assert "nlcd_mode" in per_year["1984"]["unexpected"]


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
        lambda pid, vname: ({}, verification, fixture_dir, manifest),
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


# ---------------------------------------------------------------------------
# step 2b: profile [inputs] role sweep


def test_role_sweep_static_and_temporal(tmp_path, monkeypatch):
    from mtbs_fire_analysis.pipeline import catalog_paths as cp

    base = tmp_path / "store"
    base.mkdir()
    static_p = base / "eco.gpkg"
    static_p.write_bytes(b"x")

    def fake_resolve(role, *, year=None):
        # eco_regions is STATIC: yields one Item with no year.
        if role == "eco_regions":
            assert year is None
            return static_p
        # mtbs_bs is TEMPORAL: no-year raises (many), per-year resolves.
        if role == "mtbs_bs":
            if year is None:
                raise cp.CatalogResolveError("many items")
            p = base / f"mtbs_{year}.tif"
            p.write_bytes(b"x")
            return p
        raise KeyError(role)

    monkeypatch.setattr(cp, "resolve", fake_resolve)
    profile = {"inputs": {"eco_regions": "col-eco", "mtbs_bs": "col-mtbs"}}
    records = gate_d9._role_sweep(profile, [1984, 2005])

    by_role: dict = {}
    for r in records:
        by_role.setdefault(r["role"], []).append(r)
    assert len(by_role["eco_regions"]) == 1
    assert by_role["eco_regions"][0]["year"] is None
    assert by_role["eco_regions"][0]["collection"] == "col-eco"
    assert {r["year"] for r in by_role["mtbs_bs"]} == {1984, 2005}
    assert all(r["collection"] == "col-mtbs" for r in by_role["mtbs_bs"])
    assert all(r["exists"] for r in records)


def test_role_sweep_missing_path_aborts(tmp_path, monkeypatch):
    from mtbs_fire_analysis.pipeline import catalog_paths as cp

    monkeypatch.setattr(
        cp, "resolve", lambda role, *, year=None: tmp_path / "missing.tif"
    )
    profile = {"inputs": {"eco_regions": "col-eco"}}
    with pytest.raises(gate_d9.GateAbort) as e:
        gate_d9._role_sweep(profile, [1984])
    assert e.value.step == "resolution"
    assert e.value.extra["role"] == "eco_regions"


def test_role_sweep_empty_profile_is_noop():
    assert gate_d9._role_sweep({}, [1984]) == []


# ---------------------------------------------------------------------------
# step 3: checksum assertion golden forms


def test_checksum_golden_subtable(tmp_path, monkeypatch):
    from mtbs_fire_analysis.pipeline import catalog_paths as cp

    resolved_file = tmp_path / "resolved_1985.tif"
    resolved_file.write_bytes(b"r")
    golden_dir = tmp_path / "golden_art"
    golden_dir.mkdir()
    member = "Annual_NLCD_LndCov_1984_CU_C1V2.tif"
    (golden_dir / member).write_bytes(b"g")

    class FakeClient:
        def find(self, collection, *, year=None, include_deprecated=None):
            if collection == "nlcd-landcover-c1v2":
                return [{"id": "resolved"}]
            if collection == "legacy-nlcd":
                assert include_deprecated is True
                return [{"id": "golden"}]
            return []

        def localize(self, item):
            return resolved_file if item["id"] == "resolved" else golden_dir

    monkeypatch.setattr(cp, "client", lambda: FakeClient())
    monkeypatch.setattr(gate_d9.comparator, "band_checksum", lambda p: 7)

    verification = {
        "checksum_assertions": [
            {
                "collection": "nlcd-landcover-c1v2",
                "year": 1984,
                "golden": {"collection": "legacy-nlcd", "member": member},
            }
        ]
    }
    records = gate_d9._run_checksum_assertions(verification)
    assert len(records) == 1
    assert records[0]["golden_form"] == "golden"
    assert records[0]["match"] is True
    assert records[0]["golden_member"].endswith(member)


def test_checksum_golden_member_fallback(tmp_path, monkeypatch):
    from mtbs_fire_analysis.pipeline import catalog_paths as cp

    art = tmp_path / "art"
    art.mkdir()
    resolved_file = art / "resolved.tif"
    resolved_file.write_bytes(b"r")
    (art / "golden_member.tif").write_bytes(b"g")

    class FakeClient:
        def find(self, collection, *, year=None, include_deprecated=None):
            return [{"id": "resolved"}]

        def localize(self, item):
            return resolved_file

    monkeypatch.setattr(cp, "client", lambda: FakeClient())
    monkeypatch.setattr(gate_d9.comparator, "band_checksum", lambda p: 5)

    verification = {
        "checksum_assertions": [
            {
                "collection": "legacy-x",
                "year": 2005,
                "golden_member": "golden_member.tif",
            }
        ]
    }
    records = gate_d9._run_checksum_assertions(verification)
    assert records[0]["golden_form"] == "golden_member"
    assert records[0]["match"] is True


def test_checksum_mismatch_aborts(tmp_path, monkeypatch):
    from mtbs_fire_analysis.pipeline import catalog_paths as cp

    art = tmp_path / "art"
    art.mkdir()
    resolved_file = art / "resolved.tif"
    resolved_file.write_bytes(b"r")
    (art / "golden_member.tif").write_bytes(b"g")

    class FakeClient:
        def find(self, collection, *, year=None, include_deprecated=None):
            return [{"id": "resolved"}]

        def localize(self, item):
            return resolved_file

    monkeypatch.setattr(cp, "client", lambda: FakeClient())
    # resolved vs golden differ by path -> different checksum
    monkeypatch.setattr(
        gate_d9.comparator,
        "band_checksum",
        lambda p: 1 if str(p).endswith("resolved.tif") else 2,
    )
    verification = {
        "checksum_assertions": [
            {
                "collection": "legacy-x",
                "year": 2005,
                "golden_member": "golden_member.tif",
            }
        ]
    }
    with pytest.raises(gate_d9.GateAbort) as e:
        gate_d9._run_checksum_assertions(verification)
    assert e.value.step == "checksum"


# ---------------------------------------------------------------------------
# --verification flag selects the registered entry


def test_verification_flag_selects_entry(tmp_path, monkeypatch):
    captured = {}

    def fake_load(pid, vname):
        captured["vname"] = vname
        return ({}, _verification(), tmp_path / "fix", {"outputs": {}})

    monkeypatch.setattr(gate_d9, "_load_profile_and_fixture", fake_load)
    monkeypatch.setattr(
        gate_d9, "_import_paths", lambda: _fake_paths(tmp_path, [1984])
    )
    gate_d9.run_gate(
        _args(
            scratch_root=tmp_path / "scratch",
            skip_run=True,
            skip_checksum=True,
            verification="d9_full",
        )
    )
    assert captured["vname"] == "d9_full"


def test_verification_flag_defaults_to_d9(tmp_path):
    args = _args(scratch_root=tmp_path / "s", skip_run=True)
    assert args.verification == "d9"
