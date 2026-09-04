"""Integration tests for the catalog adapter against the real store.

Skipped unless JLAB_ROOT is set and its index/catalog.parquet exists.
Read-only use of the store; the API (JLAB_API_URL) is optional -- the
client falls back to the index."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from mtbs_fire_analysis.pipeline import catalog_paths as cp

# The skip guard runs at collection time, before paths.py has loaded the
# repo-root .env, so load it here too: a configured machine runs these
# tests from a plain `uv run pytest tests` without exporting JLAB_ROOT.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_JLAB_ROOT = os.environ.get("JLAB_ROOT")
_INDEX_OK = (
    _JLAB_ROOT is not None
    and (Path(_JLAB_ROOT) / "index" / "catalog.parquet").exists()
)

pytestmark = pytest.mark.skipif(
    not _INDEX_OK,
    reason="requires JLAB_ROOT with an index/catalog.parquet",
)

GATE_YEARS = (1984, 2005, 2010, 2022)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    # Keep perims_rasters symlinks out of the real user cache.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def test_static_inputs_resolve():
    for coll in (
        "legacy-eco-regions",
        "legacy-perims",
        "legacy-nlcd-mode",
        "legacy-edna-dem",
        "legacy-edna-slope",
        "legacy-edna-aspect",
        "hex_grid",
    ):
        p = cp.resolve(coll)
        assert p.exists(), coll
    assert cp.states_path().exists()


def test_perims_rasters_dir_has_gate_years():
    d = cp.perims_rasters_dir()
    for y in GATE_YEARS:
        assert (d / f"dse_{y}.tif").exists(), y


def test_mtbs_resolves_for_gate_years():
    for y in GATE_YEARS:
        p = cp.get_mtbs_raster_path(y, "CONUS")
        assert p.exists(), y


def test_nlcd_resolves_for_gate_years_and_neighbors():
    for y in GATE_YEARS:
        assert cp.get_nlcd_raster_path(y).exists(), y
    # m10 also asks for year+1; 2022+1 = 2023 must resolve.
    assert cp.get_nlcd_raster_path(2023).exists()


def test_wui_resolves_for_all_flavors_and_gate_years():
    for y in GATE_YEARS:
        for flavor in ("bool", "class", "flag", "prox"):
            assert cp.get_wui_flavor_path(y, flavor).exists(), (y, flavor)


def test_nlcd_1984_floors_to_1985():
    assert "1985" in cp.get_nlcd_raster_path(1984).name


def test_wui_bucket_policy_outcomes():
    assert "2000" in cp.get_wui_flavor_path(2005, "prox").name
    assert "1990" in cp.get_wui_flavor_path(1984, "bool").name
    assert "2010" in cp.get_wui_flavor_path(2010, "class").name
    assert "2020" in cp.get_wui_flavor_path(2022, "flag").name
