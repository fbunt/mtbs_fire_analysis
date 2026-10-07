"""Unit tests for the adapter's injected mode, on a synthetic manifest.

When the platform runs m10 as its pixel-event recipe it resolves every
input itself, writes an inputs manifest (schema
``jlab.entrypoint_inputs/v1``) and sets ``JLAB_INPUTS`` to it and
``JLAB_INPUTS_USED`` to a file the code appends the paths it served to.
These tests build such a manifest for one year with exactly the recipe's
input roles, read every input the way m10 does, and check that the served
set equals the declared set under the platform's rule, that a missing role
fails loudly, and that this mode imports no jlab and loads no repo .env."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import dotenv
import pytest

from mtbs_fire_analysis.pipeline import catalog_paths as cp

# The input roles of the platform's mtbs-pixel-events recipe
# (datasets/mtbs-pixel-events/recipe.toml in the platform repo): the
# manifest keys this adapter must serve. NLCD is requested three times
# (the year, the year before, the year after).
RECIPE_ROLES = frozenset(
    {
        "dse",
        "mtbs_bs",
        "nlcd",
        "nlcd_pre",
        "nlcd_post",
        "wui_flag",
        "wui_class",
        "wui_bool",
        "wui_prox",
        "perims",
        "states",
        "eco_regions",
        "hex_grid",
        "nlcd_mode",
        "dem",
        "slope",
        "aspect",
    }
)

_STATIC_FILES = {
    "perims": "perims.gpkg",
    "eco_regions": "eco_regions.gpkg",
    "hex_grid": "hex_grid.gpkg",
    "nlcd_mode": "nlcd_mode.tif",
    "dem": "dem.tif",
    "slope": "slope.tif",
    "aspect": "aspect.tif",
}


def _file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _nlcd_file(store, requested):
    # The platform floors NLCD to 1985: a request below it lands on the
    # 1985 artifact, so in 1984 all three NLCD roles name one file.
    y = max(requested, 1985)
    return _file(store / f"nlcd-{y}" / f"nlcd_{y}.tif")


def _manifest_inputs(store, year):
    """The manifest ``inputs`` table for `year`: each role in its own
    artifact directory under `store`, as the executor resolves them."""
    entries = {}

    def put(role, path, key):
        entries[role] = {
            "path": str(path),
            "collection": f"col-{role}",
            "item": f"item-{role}",
            "temporal_key": key,
            "sha256": "0" * 64,
        }

    put("dse", _file(store / f"dse-{year}" / f"dse_{year}.tif"), year)
    put("mtbs_bs", _file(store / f"bs-{year}" / f"bs_{year}.tif"), year)
    put("nlcd", _nlcd_file(store, year), year)
    put("nlcd_pre", _nlcd_file(store, year - 1), year - 1)
    put("nlcd_post", _nlcd_file(store, year + 1), year + 1)
    for flavor in ("flag", "class", "bool", "prox"):
        path = _file(store / f"wui-{flavor}" / f"wui_{flavor}.tif")
        put(f"wui_{flavor}", path, year)
    bundle = store / "states"
    _file(bundle / "states.shp")
    _file(bundle / "states.dbf")
    put("states", bundle, None)
    for role, name in _STATIC_FILES.items():
        put(role, _file(store / role / name), None)
    return entries


@pytest.fixture
def injected(tmp_path, monkeypatch):
    """Injected-mode env over a writable manifest; returns a writer that
    stores a manifest ``inputs`` table (or a whole document) and the
    used-paths file."""
    manifest = tmp_path / "scratch" / "inputs.json"
    used = tmp_path / "scratch" / "inputs_used.txt"
    manifest.parent.mkdir()
    monkeypatch.setenv("JLAB_INPUTS", str(manifest))
    monkeypatch.setenv("JLAB_INPUTS_USED", str(used))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "scratch" / "cache"))
    monkeypatch.setenv("FIRE_DATA_ROOT", str(tmp_path / "scratch"))
    # JLAB_INPUTS wins over an explicit FIRE_CATALOG=0 and a stray platform
    # env; the client is never reached.
    monkeypatch.setenv("FIRE_CATALOG", "0")
    monkeypatch.setenv("JLAB_ROOT", str(tmp_path / "no-store"))

    def no_client():
        raise AssertionError("injected mode must not use the jlab client")

    monkeypatch.setattr(cp, "client", no_client)

    def write(inputs, *, document=None):
        doc = document or {"schema": cp.MANIFEST_SCHEMA, "inputs": inputs}
        manifest.write_text(json.dumps(doc))
        return used

    return write


def _served(used):
    if not used.exists():
        return []
    return [ln for ln in used.read_text().splitlines() if ln.strip()]


def _load_paths():
    """A fresh execution of paths.py, kept out of sys.modules so the
    injected rebinding does not leak into other tests."""
    spec = importlib.util.find_spec("mtbs_fire_analysis.pipeline.paths")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _m10_reads(paths, year):
    """Every input path m10 reads for `year`, obtained as m10 does."""
    reads = [
        paths.ASPECT_PATH,
        paths.ECO_REGIONS_PATH,
        paths.ELEVATION_PATH,
        paths.HEX_GRID_PATH,
        paths.NLCD_MODE_RASTER_PATH,
        paths.PERIMS_PATH,
        paths.SLOPE_PATH,
        paths.STATES_PATH,
        paths.PERIMS_RASTERS_PATH / f"dse_{year}.tif",
        paths.get_mtbs_raster_path(year, "CONUS"),
        paths.get_nlcd_raster_path(year),
        paths.get_nlcd_raster_path(max(year - 1, 1984)),
        paths.get_nlcd_raster_path(year + 1),
    ]
    reads += [
        paths.get_wui_flavor_path(year, f)
        for f in ("flag", "class", "bool", "prox")
    ]
    return reads


def _inside(path, root):
    return path == root or path.startswith(root.rstrip("/") + "/")


def _verdict(served, declared):
    """The platform's comparison on realpaths: (served paths outside every
    declared input, declared inputs no served path covers)."""
    s = [os.path.realpath(p) for p in served]
    d = [os.path.realpath(p) for p in declared]
    undeclared = {p for p in s if not any(_inside(p, r) for r in d)}
    unserved = {r for r in d if not any(_inside(p, r) for p in s)}
    return undeclared, unserved


# --- the role contract -------------------------------------------------------


@pytest.mark.parametrize("year", [1984, 1985, 2005, 2022])
def test_m10_reads_serve_exactly_the_recipe_roles(
    injected, tmp_path, year, monkeypatch
):
    calls = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: calls.append(a))
    inputs = _manifest_inputs(tmp_path / "store", year)
    assert set(inputs) == RECIPE_ROLES
    used = injected(inputs)
    paths = _load_paths()
    reads = _m10_reads(paths, year)
    assert calls == [], "injected mode must not load a repo .env"
    for p in reads:
        assert Path(p).exists(), p
    served = _served(used)
    declared = [e["path"] for e in inputs.values()]
    # every served line is a manifest path, verbatim
    assert set(served) <= set(declared)
    assert _verdict(served, declared) == (set(), set())


def test_platform_comparison_accepts_the_served_paths(injected, tmp_path):
    rec = pytest.importorskip("jlab.recipes.primitives.run_entrypoint")
    inputs = _manifest_inputs(tmp_path / "store", 2005)
    used = injected(inputs)
    _m10_reads(_load_paths(), 2005)
    declared = [e["path"] for e in inputs.values()]
    assert rec.inputs_used_verdict(_served(used), declared) == ([], [])


def test_role_list_matches_the_platform_recipe():
    spec = importlib.util.find_spec("jlab")
    if spec is None or spec.origin is None:
        pytest.skip("the platform package is not installed")
    recipe = (
        Path(spec.origin).resolve().parents[1]
        / "datasets"
        / "mtbs-pixel-events"
        / "recipe.toml"
    )
    if not recipe.is_file():
        pytest.skip(f"no platform recipe at {recipe}")
    decl = tomllib.loads(recipe.read_text())
    assert set(decl["inputs"]) == RECIPE_ROLES
    params = decl["params"]
    assert params["inputs_used"] is True
    assert params["module"] == "mtbs_fire_analysis.pipeline.m10_data_extract"


@pytest.mark.parametrize("role", sorted(RECIPE_ROLES))
def test_a_missing_role_raises(injected, tmp_path, role):
    inputs = _manifest_inputs(tmp_path / "store", 2005)
    del inputs[role]
    injected(inputs)
    with pytest.raises(cp.CatalogResolveError, match="manifest"):
        _m10_reads(_load_paths(), 2005)


# --- per-name lookups --------------------------------------------------------


def test_injected_wins_over_fire_catalog(injected):
    assert cp.injected()
    assert cp.catalog_enabled() is True


def test_nlcd_lookup_by_requested_key(injected, tmp_path):
    inputs = _manifest_inputs(tmp_path / "store", 2005)
    used = injected(inputs)
    assert cp.get_nlcd_raster_path(2005) == Path(inputs["nlcd"]["path"])
    assert cp.get_nlcd_raster_path(2004) == Path(inputs["nlcd_pre"]["path"])
    assert cp.get_nlcd_raster_path(2006) == Path(inputs["nlcd_post"]["path"])
    assert _served(used) == [
        inputs["nlcd"]["path"],
        inputs["nlcd_pre"]["path"],
        inputs["nlcd_post"]["path"],
    ]
    with pytest.raises(cp.CatalogResolveError, match="2010"):
        cp.get_nlcd_raster_path(2010)


def test_first_year_nlcd_roles_share_the_floor_artifact(injected, tmp_path):
    inputs = _manifest_inputs(tmp_path / "store", 1984)
    injected(inputs)
    floor = Path(inputs["nlcd"]["path"])
    assert floor.name == "nlcd_1985.tif"
    # m10 asks for 1984 twice (the year and max(year - 1, 1984)) and 1985.
    assert cp.get_nlcd_raster_path(1984) == floor
    assert cp.get_nlcd_raster_path(1985) == floor


def test_year_mismatch_raises(injected, tmp_path):
    injected(_manifest_inputs(tmp_path / "store", 2005))
    with pytest.raises(cp.CatalogResolveError, match="2006"):
        cp.get_mtbs_raster_path(2006, "CONUS")
    with pytest.raises(cp.CatalogResolveError, match="2006"):
        cp.get_wui_flavor_path(2006, "flag")


def test_wui_flavor_and_aoi_guards(injected, tmp_path):
    inputs = _manifest_inputs(tmp_path / "store", 2005)
    injected(inputs)
    assert cp.get_wui_flavor_path(2005, "prox") == Path(
        inputs["wui_prox"]["path"]
    )
    with pytest.raises(cp.CatalogResolveError):
        cp.get_wui_flavor_path(2005, "bogus")
    with pytest.raises(cp.CatalogResolveError):
        cp.get_mtbs_raster_path(2005, "AK")


def test_states_path_from_the_bundle_dir(injected, tmp_path):
    inputs = _manifest_inputs(tmp_path / "store", 2005)
    used = injected(inputs)
    bundle = Path(inputs["states"]["path"])
    assert cp.states_path() == bundle / "states.shp"
    assert _served(used) == [str(bundle)]


def test_states_path_from_the_member(injected, tmp_path):
    inputs = _manifest_inputs(tmp_path / "store", 2005)
    member = str(Path(inputs["states"]["path"]) / "states.shp")
    inputs["states"]["path"] = member
    used = injected(inputs)
    assert cp.states_path() == Path(member)
    assert _served(used) == [member]


def test_states_path_missing_member_raises(injected, tmp_path):
    inputs = _manifest_inputs(tmp_path / "store", 2005)
    (Path(inputs["states"]["path"]) / "states.shp").unlink()
    injected(inputs)
    with pytest.raises(cp.CatalogResolveError, match="states.shp"):
        cp.states_path()


def test_perims_rasters_dir_holds_only_the_manifest_year(injected, tmp_path):
    inputs = _manifest_inputs(tmp_path / "store", 2005)
    used = injected(inputs)
    cache = tmp_path / "scratch" / "cache"
    directory = cache / "mtbs_fire_analysis" / "perims_rasters"
    directory.mkdir(parents=True)
    (directory / "dse_1999.tif").symlink_to(tmp_path / "elsewhere.tif")
    (directory / "keep_me.txt").write_text("x")
    d = cp.perims_rasters_dir()
    assert d == directory
    link = d / "dse_2005.tif"
    assert link.is_symlink()
    assert link.readlink() == Path(inputs["dse"]["path"])
    assert not (d / "dse_1999.tif").is_symlink()
    assert (d / "keep_me.txt").exists()
    # the manifest path is what is served, not the symlink
    assert _served(used) == [inputs["dse"]["path"]]


# --- manifest errors ---------------------------------------------------------


def test_unreadable_manifest_raises(injected, monkeypatch, tmp_path):
    monkeypatch.setenv("JLAB_INPUTS", str(tmp_path / "absent.json"))
    with pytest.raises(cp.CatalogResolveError, match="absent.json"):
        cp.resolve("dem")


def test_wrong_schema_raises(injected, tmp_path):
    injected({}, document={"schema": "something/v9", "inputs": {}})
    with pytest.raises(cp.CatalogResolveError, match=cp.MANIFEST_SCHEMA):
        cp.resolve("dem")


def test_no_used_file_when_unset(injected, tmp_path, monkeypatch):
    monkeypatch.delenv("JLAB_INPUTS_USED")
    inputs = _manifest_inputs(tmp_path / "store", 2005)
    used = injected(inputs)
    assert cp.resolve("dem") == Path(inputs["dem"]["path"])
    assert not used.exists()


# --- a clean process: no jlab, no .env ---------------------------------------


def test_import_in_a_clean_process_skips_jlab_and_dotenv(tmp_path):
    inputs = _manifest_inputs(tmp_path / "store", 2005)
    manifest = tmp_path / "inputs.json"
    manifest.write_text(
        json.dumps({"schema": cp.MANIFEST_SCHEMA, "inputs": inputs})
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("JLAB_", "FIRE_"))
    }
    env.update(
        JLAB_INPUTS=str(manifest),
        JLAB_INPUTS_USED=str(tmp_path / "used.txt"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        FIRE_DATA_ROOT=str(tmp_path / "scratch"),
    )
    probe = (
        "import json, sys\n"
        "import dotenv\n"
        "calls = []\n"
        "dotenv.load_dotenv = lambda *a, **k: calls.append(1)\n"
        "from mtbs_fire_analysis.pipeline import paths\n"
        "print(json.dumps({'jlab': any(m == 'jlab' or m.startswith('jlab.')"
        " for m in sys.modules), 'dotenv_calls': len(calls),"
        " 'states': str(paths.STATES_PATH)}))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["jlab"] is False
    assert result["dotenv_calls"] == 0
    assert result["states"] == str(
        Path(inputs["states"]["path"]) / "states.shp"
    )
