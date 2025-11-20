from mtbs_fire_analysis.analysis import registry
from mtbs_fire_analysis.analysis.fit_constraints import (
    get_bounds_log_for,
    get_soft_box_for,
)


def test_all_registry_dists_have_constraints():
    for ctor in registry.REGISTRY.values():
        model = ctor()  # rely on default constructor working
        cls_name = type(model).__name__

        # bounds must exist and match theta dimensionality
        bounds = get_bounds_log_for(cls_name)
        theta = model._theta_get()
        assert len(bounds) == len(theta), (
            f"Bounds len mismatch for {cls_name}: "
            f"{len(bounds)} vs {len(theta)}"
        )

        # soft box must exist and be non-empty; params must supply keys
        sb = get_soft_box_for(cls_name)
        assert isinstance(sb, dict) and len(sb) > 0, (
            f"Empty soft_box for {cls_name}"
        )
        params = model.params
        for key in sb:
            assert (key in params or hasattr(model, key)), (
                f"Model {cls_name} missing parameter key '{key}' "
                f"required by soft_box"
            )
