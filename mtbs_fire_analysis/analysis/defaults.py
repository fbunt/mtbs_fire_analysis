from datetime import datetime as dt

MTBS_START = dt(1984, 1, 1)
MTBS_END = dt(2023, 1, 1)

DEFAULT_FIXED_PIVOTS = ["eco_lvl_3"]
DEFAULT_VARIED_PIVOTS = ["nlcd"]

EXTRA_FIXED_FIELDS = ["hexel_id", "nlcd_mode"]

FIXED_LABELS = ["eco_lvl_1", "eco_lvl_2", "eco_lvl_3"]
VARIED_LABELS = ["nlcd"]

MINIMUM_DT = 0.25 # Minimum time between fires in years

# Add covariates here (altitude, solar radiation proxy, burn severity)
