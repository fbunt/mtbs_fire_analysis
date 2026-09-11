"""Integration tests for the catalog adapter against the real store.

Skipped unless JLAB_ROOT is set and its index/catalog.parquet exists.
Read-only use of the store; the API (JLAB_API_URL) is optional -- the
client falls back to the index.

Parametrized over both profiles via FIRE_CATALOG_PROFILE: ``wave0-gate``
(the legacy-* imports, always present here) and ``fire`` (the wave-1
products, which self-skip until S9/S10 has produced them). Each param
reloads the adapter so PROFILE_ID and the client cache pick up the env."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from jlab import profiles

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


@pytest.fixture(params=["wave0-gate", "fire"])
def adapter(request, tmp_path, monkeypatch):
    """The catalog_paths module bound to the parametrized profile.

    Sets FIRE_CATALOG_PROFILE, reloads the module (so PROFILE_ID re-reads the
    env), and clears the client cache. The ``fire`` parametrization self-skips
    until its products exist: if the profile binds ``mtbs_bs`` to a collection
    with no 2010 Item, the wave-1 recipes (S9/S10) have not run yet."""
    profile_id = request.param
    monkeypatch.setenv("FIRE_CATALOG_PROFILE", profile_id)
    # Keep perims_rasters symlinks out of the real user cache.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    importlib.reload(cp)
    cp.client.cache_clear()
    if profile_id == "fire":
        c = cp.client()
        col = profiles.input_collection(c.profile, "mtbs_bs")
        if not c.find(col, year=2010):
            pytest.skip(
                f"fire products not built yet (S9/S10): {col} has no 2010 Item"
            )
    yield cp
    cp.client.cache_clear()


def test_static_inputs_resolve(adapter):
    for role in (
        "eco_regions",
        "perims",
        "nlcd_mode",
        "dem",
        "slope",
        "aspect",
        "hex_grid",
    ):
        p = adapter.resolve(role)
        assert p.exists(), role
    assert adapter.states_path().exists()


def test_perims_rasters_dir_has_gate_years(adapter):
    d = adapter.perims_rasters_dir()
    for y in GATE_YEARS:
        assert (d / f"dse_{y}.tif").exists(), y


def test_mtbs_resolves_for_gate_years(adapter):
    for y in GATE_YEARS:
        p = adapter.get_mtbs_raster_path(y, "CONUS")
        assert p.exists(), y


def test_nlcd_resolves_for_gate_years_and_neighbors(adapter):
    for y in GATE_YEARS:
        assert adapter.get_nlcd_raster_path(y).exists(), y
    # m10 also asks for year+1; 2022+1 = 2023 must resolve.
    assert adapter.get_nlcd_raster_path(2023).exists()


def test_wui_resolves_for_all_flavors_and_gate_years(adapter):
    for y in GATE_YEARS:
        for flavor in ("bool", "class", "flag", "prox"):
            assert adapter.get_wui_flavor_path(y, flavor).exists(), (y, flavor)


def test_nlcd_1984_floors_to_1985(adapter):
    assert "1985" in adapter.get_nlcd_raster_path(1984).name


def test_wui_bucket_policy_outcomes(adapter):
    assert "2000" in adapter.get_wui_flavor_path(2005, "prox").name
    assert "1990" in adapter.get_wui_flavor_path(1984, "bool").name
    assert "2010" in adapter.get_wui_flavor_path(2010, "class").name
    assert "2020" in adapter.get_wui_flavor_path(2022, "flag").name
