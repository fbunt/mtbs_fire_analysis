# %%

import polars as pl
from datetime import datetime as dt
from pathlib import Path
from mtbs_fire_analysis.analysis.mining import build_event_histories


fit_data_path = Path("mtbs_fire_analysis") / "outputs" / "HLH_Fits_Eco3"
fit_data_file = fit_data_path / "fits_2022-01-01.parquet"
summary_data_file = Path("mtbs_fire_analysis") / "data" / "eco_nlcd_summary.pqt"

fit_data = pl.read_parquet(fit_data_file)
summary_data = pl.read_parquet(summary_data_file)

fit_data 

# %%

