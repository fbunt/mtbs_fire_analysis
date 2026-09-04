"""Unit tests for the catalog path adapter with a fake jlab client."""

from __future__ import annotations

import pytest

from mtbs_fire_analysis.pipeline import catalog_paths as cp


class FakeClient:
    """A stand-in for jlab.client.Client with scripted find/localize.

    `responses` maps a collection (or a (collection, year) tuple, which
    wins) to the item list `find` returns; `localized` maps an item id to
    the path `localize` returns."""

    def __init__(self):
        self.responses = {}
        self.localized = {}

    def find(self, collection, *, year=None):
        if (collection, year) in self.responses:
            return self.responses[(collection, year)]
        return self.responses.get(collection, [])

    def localize(self, item, *, verify="size"):
        return self.localized[item["id"]]


@pytest.fixture
def fake(monkeypatch):
    c = FakeClient()
    monkeypatch.setattr(cp, "client", lambda: c)
    return c


def _item(iid, datetime=None):
    props = {"datetime": datetime} if datetime is not None else {}
    return {"id": iid, "properties": props}


# --- exactly-one enforcement ------------------------------------------------


def test_resolve_one(fake, tmp_path):
    target = tmp_path / "a.tif"
    target.touch()
    fake.responses["legacy-nlcd-mode"] = [_item("legacy-nlcd-mode-x")]
    fake.localized["legacy-nlcd-mode-x"] = target
    assert cp.resolve("legacy-nlcd-mode") == target


def test_resolve_zero_raises_with_candidates(fake):
    fake.responses["legacy-perims"] = []
    with pytest.raises(cp.CatalogResolveError) as e:
        cp.resolve("legacy-perims")
    msg = str(e.value)
    assert "legacy-perims" in msg
    assert "got 0" in msg


def test_resolve_many_raises_with_candidate_ids(fake):
    fake.responses["legacy-nlcd"] = [_item("a-1"), _item("a-2")]
    with pytest.raises(cp.CatalogResolveError) as e:
        cp.resolve("legacy-nlcd", year=2005)
    msg = str(e.value)
    assert "year=2005" in msg
    assert "a-1" in msg and "a-2" in msg


# --- argument guards --------------------------------------------------------


def test_aoi_guard(fake, tmp_path):
    t = tmp_path / "m.tif"
    t.touch()
    fake.responses[("legacy-mtbs", 2005)] = [_item("legacy-mtbs-2005")]
    fake.localized["legacy-mtbs-2005"] = t
    assert cp.get_mtbs_raster_path(2005, "CONUS") == t
    with pytest.raises(cp.CatalogResolveError):
        cp.get_mtbs_raster_path(2005, "AK")


def test_flavor_guard(fake, tmp_path):
    t = tmp_path / "w.tif"
    t.touch()
    fake.responses[("legacy-wui-bool", 2005)] = [_item("legacy-wui-bool-2000")]
    fake.localized["legacy-wui-bool-2000"] = t
    assert cp.get_wui_flavor_path(2005, "bool") == t
    with pytest.raises(cp.CatalogResolveError):
        cp.get_wui_flavor_path(2005, "bogus")


# --- states bundle member ---------------------------------------------------


def test_states_path_composes_shp(fake, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "states.shp").touch()
    fake.responses["state_borders"] = [_item("state_borders-static")]
    fake.localized["state_borders-static"] = bundle
    assert cp.states_path() == bundle / "states.shp"


def test_states_path_missing_member_raises(fake, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    fake.responses["state_borders"] = [_item("state_borders-static")]
    fake.localized["state_borders-static"] = bundle
    with pytest.raises(cp.CatalogResolveError):
        cp.states_path()


# --- perims_rasters_dir -----------------------------------------------------


def _seed_dse(fake, tmp_path, years):
    items = []
    for y in years:
        src = tmp_path / f"src_dse_{y}.tif"
        src.touch()
        iid = f"legacy-dse-{y}"
        items.append(_item(iid, datetime=f"{y}-01-01T00:00:00Z"))
        fake.localized[iid] = src
    fake.responses["legacy-dse"] = items


def test_perims_rasters_dir_creates_symlinks(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _seed_dse(fake, tmp_path, [1984, 2005, 2010, 2022])
    d = cp.perims_rasters_dir()
    for y in (1984, 2005, 2010, 2022):
        link = d / f"dse_{y}.tif"
        assert link.is_symlink()
        assert link.resolve() == (tmp_path / f"src_dse_{y}.tif").resolve()


def test_perims_rasters_dir_idempotent(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _seed_dse(fake, tmp_path, [1984, 2005])
    d1 = cp.perims_rasters_dir()
    before = {p.name: p.readlink() for p in d1.iterdir()}
    d2 = cp.perims_rasters_dir()
    after = {p.name: p.readlink() for p in d2.iterdir()}
    assert d1 == d2
    assert before == after


def test_perims_rasters_dir_relinks_on_target_change(
    fake, tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _seed_dse(fake, tmp_path, [1984])
    d = cp.perims_rasters_dir()
    new_target = tmp_path / "new_dse_1984.tif"
    new_target.touch()
    fake.localized["legacy-dse-1984"] = new_target
    d = cp.perims_rasters_dir()
    link = d / "dse_1984.tif"
    assert link.resolve() == new_target.resolve()


def test_perims_rasters_dir_preserves_foreign_files(
    fake, tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _seed_dse(fake, tmp_path, [1984])
    d = cp.perims_rasters_dir()
    foreign = d / "keep_me.txt"
    foreign.write_text("x")
    cp.perims_rasters_dir()
    assert foreign.exists()


# --- catalog_enabled truth table --------------------------------------------


@pytest.mark.parametrize(
    "fire_catalog,jlab_root,jlab_api,expected",
    [
        ("1", None, None, True),
        ("true", None, None, True),
        ("on", None, None, True),
        ("0", "/x", "http://y", False),
        ("false", "/x", None, False),
        ("off", None, "http://y", False),
        (None, "/x", None, True),
        (None, None, "http://y", True),
        (None, None, None, False),
    ],
)
def test_catalog_enabled(
    monkeypatch, fire_catalog, jlab_root, jlab_api, expected
):
    for name, val in (
        ("FIRE_CATALOG", fire_catalog),
        ("JLAB_ROOT", jlab_root),
        ("JLAB_API_URL", jlab_api),
    ):
        if val is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, val)
    assert cp.catalog_enabled() is expected


# --- get_points_path stays local scratch ------------------------------------


def test_get_points_path_is_local_scratch(monkeypatch):
    import importlib

    monkeypatch.setenv("FIRE_CATALOG", "0")
    from mtbs_fire_analysis.pipeline import paths

    importlib.reload(paths)
    p = paths.get_points_path(2005, "CONUS")
    assert str(p).startswith(str(paths.ROOT_TMP_DIR))
