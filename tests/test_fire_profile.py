"""Tests for the project profile (profiles/fire.toml).

The profile is the project (values only), submitted to the platform by path and
served read-only. These tests load it BY PATH and validate it through the
platform's own loader (jlab.profiles), so a schema drift on either side is
caught here. jlab is installed editable, so its grid registry already carries
the conus-albers-30m grid and conus mask the profile names.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jlab import profiles

PROFILE = Path(__file__).resolve().parent.parent / "profiles" / "fire.toml"

EXPECTED_ROLES = {
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
}


@pytest.fixture(scope="module")
def profile() -> dict:
    p = profiles.load_profile(str(PROFILE))
    profiles.validate_profile(p)
    return p


def test_loads_and_validates(profile: dict) -> None:
    assert profile["id"] == "fire"
    # 21 products + 8 raw inputs.
    assert len(profile["selection"]["collections"]) == 29


def test_inputs_are_selected_collections(profile: dict) -> None:
    selected = set(profile["selection"]["collections"])
    for role, col in profile["inputs"].items():
        assert col in selected, f"input {role} -> {col} not selected"


def test_input_roles_are_exactly_the_fifteen(profile: dict) -> None:
    assert set(profile["inputs"]) == EXPECTED_ROLES


def test_level1_keys_are_selected_collections(profile: dict) -> None:
    selected = set(profile["selection"]["collections"])
    level1 = profiles.verification_for(profile, "level1")
    assert level1["kind"] == "artifact-equivalence"
    for col in level1["collections"]:
        assert col in selected, f"level1 collection {col} not selected"


def test_exclude_deprecated_is_true(profile: dict) -> None:
    assert profiles.excludes_deprecated(profile) is True


def test_policy_bindings(profile: dict) -> None:
    floor = profiles.policy_for(profile, "nlcd-landcover-c1v2")
    assert floor is not None
    assert floor["kind"] == "nearest-earlier-with-floor"
    assert floor["floor"] == 1985
    bucket = profiles.policy_for(profile, "silvis-wui-flag")
    assert bucket is not None
    assert bucket["kind"] == "bucket-select"
    assert bucket["buckets"] == [1990, 2000, 2010, 2020]


def test_input_collection_accessor(profile: dict) -> None:
    assert profiles.input_collection(profile, "nlcd") == (
        "nlcd-landcover-c1v2"
    )
    with pytest.raises(profiles.ProfileError):
        profiles.input_collection(profile, "no-such-role")
