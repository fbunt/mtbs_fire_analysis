"""Catalog-backed path adapter for the pipeline (jlab client).

Resolves the catalog-backed inputs `m10` needs through the jlab thin
client (`jlab.client.Client.find()` / `localize()`) against the platform
store, returning local `pathlib.Path`s.  `paths.py` rebinds exactly the
12 catalog-backed names to this module when `catalog_enabled()`; every
other path in `paths.py` (scratch, results, cache, raw layout) stays on
the legacy `FIRE_DATA_ROOT` layout.

This adapter carries NO collection literal.  Each input is named by a
stable INPUT ROLE (see `ROLES`); the active profile's `[inputs]` table
maps the role to the collection it resolves against
(`jlab.profiles.input_collection`).  So the same adapter reads the
wave-0 imported collections under the `wave0-gate` profile and the
wave-1 products under the `fire` profile with no code change -- only the
profile's role bindings differ.

Enable rule (`catalog_enabled()`):
  * `FIRE_CATALOG` in {1, true, on}  -> enabled
  * `FIRE_CATALOG` in {0, false, off} -> disabled
  * `FIRE_CATALOG` unset -> enabled iff `JLAB_ROOT` or `JLAB_API_URL` is
    set (both read from the process env, which `paths.py` populates from
    the repo-root `.env` via python-dotenv before importing this module).

The client finds the index via `JLAB_ROOT` and the API via
`JLAB_API_URL`.  Temporal substitutions (NLCD 1984->1985, WUI decade
buckets) are done INSIDE the client by the loaded profile's resolve
policies; this adapter contains no year arithmetic.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path

# Profile that supplies the temporal resolve policies (NLCD floor, WUI
# buckets) and the role -> collection input bindings this adapter reads.
# Default `fire` = the wave-1 products; `wave0-gate` (the wave-0 imports) stays
# selectable for regression runs.
PROFILE_ID = os.environ.get("FIRE_CATALOG_PROFILE", "fire")

_WUI_FLAVORS = ("bool", "class", "flag", "prox")

# The 15 input roles the adapter resolves. These are the contract shared
# with the profile's [inputs] table; the profile decides which collection
# each maps to. Kept as data so tests and callers can assert against it.
ROLES = (
    "nlcd",
    "nlcd_mode",
    "mtbs_bs",
    "dse",
    "perims",
    "states",
    "eco_regions",
    "hex_grid",
    "dem",
    "slope",
    "aspect",
    "wui_flag",
    "wui_class",
    "wui_bool",
    "wui_prox",
)

# The states product ships a `states.shp` member: both the wave-0 states
# bundle and the wave-1 states product write this same member name, so it
# is stable across profiles.
_STATES_MEMBER = "states.shp"


class CatalogResolveError(RuntimeError):
    """A catalog input did not resolve to exactly one localizable path
    (an unknown role, zero or many candidates, an unsupported argument, or
    a missing bundle member)."""


@functools.cache
def client():
    """A process-cached `jlab.client.Client` bound to the active profile.

    The client reads the index from `JLAB_ROOT` and, when set, the API
    from `JLAB_API_URL`.  Cached so a single import-time resolution
    sweep reuses one client (and one API probe)."""
    from jlab.client import Client

    return Client(profile=PROFILE_ID)


def catalog_enabled():
    """Whether `paths.py` should serve catalog-backed inputs.  See the
    module docstring for the truth table."""
    v = os.environ.get("FIRE_CATALOG")
    if v is not None:
        s = v.strip().lower()
        if s in ("1", "true", "on"):
            return True
        if s in ("0", "false", "off"):
            return False
    return bool(os.environ.get("JLAB_ROOT") or os.environ.get("JLAB_API_URL"))


def _collection_for(role):
    """The collection the active profile binds input `role` to. Raises
    `CatalogResolveError` (naming the role and the profile id) when the
    profile has no such role."""
    from jlab.profiles import ProfileError, input_collection

    try:
        return input_collection(client().profile, role)
    except ProfileError as e:
        raise CatalogResolveError(
            f"input role {role!r} is not bound by profile {PROFILE_ID!r}: {e}"
        ) from e


def resolve(role, *, year=None):
    """Resolve input `role` (optionally at `year`) to a single local path.

    The profile's `[inputs]` table maps `role` to a collection; `find()`
    must then return EXACTLY one item -- zero or many is a
    `CatalogResolveError` naming the role, the collection, the year, and
    the candidate ids.  Never falls back to a `FIRE_DATA_ROOT` path."""
    collection = _collection_for(role)
    items = client().find(collection, year=year)
    if len(items) != 1:
        ids = [it.get("id") for it in items]
        raise CatalogResolveError(
            f"resolve({role!r}, year={year!r}) via collection {collection!r} "
            f"expected exactly one item, got {len(ids)}: {ids}"
        )
    return client().localize(items[0])


def states_path():
    """The `states.shp` member of the states product bundle (`states`
    role).

    The states product localizes as the artifact DIRECTORY (a multi-member
    bundle); assert the `.shp` member exists."""
    directory = resolve("states")
    shp = Path(directory) / _STATES_MEMBER
    if not shp.exists():
        raise CatalogResolveError(
            f"states bundle at {directory} has no {_STATES_MEMBER} member"
        )
    return shp


def get_mtbs_raster_path(year, aoi_code):
    """MTBS burn-severity raster for `year` (`mtbs_bs` role).  The bound
    collection is CONUS only, so any other `aoi_code` is an error."""
    if aoi_code != "CONUS":
        raise CatalogResolveError(
            f"get_mtbs_raster_path: aoi_code {aoi_code!r} is not "
            "supported; the bound collection is CONUS only"
        )
    return resolve("mtbs_bs", year=year)


def get_nlcd_raster_path(year):
    """NLCD land-cover raster for `year` (`nlcd` role).  The profile's
    nearest-earlier-with-floor(1985) policy does the 1984->1985
    substitution inside the client; no year arithmetic here."""
    return resolve("nlcd", year=year)


def get_wui_flavor_path(year, flavor):
    """WUI raster of `flavor` for `year` (the `wui_<flavor>` role).  The
    profile's bucket-select policy does the decade bucketing inside the
    client; no year arithmetic here."""
    if flavor not in _WUI_FLAVORS:
        raise CatalogResolveError(
            f"get_wui_flavor_path: flavor {flavor!r} is not one of "
            f"{_WUI_FLAVORS}"
        )
    return resolve(f"wui_{flavor}", year=year)


def _dse_year(item):
    """Parse the year of a `dse`-role item from its
    `properties.datetime`, falling back to a trailing 4-digit id
    suffix."""
    props = item.get("properties") or {}
    dt = props.get("datetime")
    if isinstance(dt, str) and dt[:4].isdigit():
        return int(dt[:4])
    iid = item.get("id") or ""
    m = re.search(r"(\d{4})$", iid)
    if m:
        return int(m.group(1))
    raise CatalogResolveError(f"cannot parse a year from dse item {iid!r}")


def _ensure_symlink(link, target):
    """Point `link` at `target`, relinking only when the current target
    differs.  Idempotent; never removes anything but a stale `link`."""
    target = Path(target)
    if link.is_symlink():
        try:
            current = link.readlink()
        except OSError:
            current = None
        if current == target:
            return
        link.unlink()
    elif link.exists():
        link.unlink()
    link.symlink_to(target)


def perims_rasters_dir():
    """A per-machine directory holding `dse_{year}.tif` symlinks into the
    localized `dse`-role artifacts, one per item of the collection.

    Lives under `$XDG_CACHE_HOME/mtbs_fire_analysis/perims_rasters/`
    (fallback `~/.cache`).  Idempotent (re-links only when the target
    changed) and never deletes anything else in the directory.  `m10`
    composes `<dir>/dse_{year}.tif` itself."""
    cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache) if cache else Path.home() / ".cache"
    directory = base / "mtbs_fire_analysis" / "perims_rasters"
    directory.mkdir(parents=True, exist_ok=True)
    for item in client().find(_collection_for("dse")):
        year = _dse_year(item)
        target = client().localize(item)
        _ensure_symlink(directory / f"dse_{year}.tif", target)
    return directory
