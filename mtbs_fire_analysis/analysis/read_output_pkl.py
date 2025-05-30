# %%
from pathlib import Path
import pickle
import polars as pl
import numpy as np

from mtbs_fire_analysis.analysis.mining import (
    build_dts_df,
    build_survival_times,
)


out_dir = Path("mtbs_fire_analysis/outputs/HLH_Fits")

pickle_file = out_dir / "HLH_fits.pkl"

outputs = pickle.load(open(pickle_file, "rb"))

print(outputs)


refresh = False
# %%
cache_path = Path("/fastdata") / "FredData" / "cache"

if refresh:
    data_path = Path("/fastdata") / "FredData" / "mtbs_CONUS_1984_2022"
    lf = pl.scan_parquet(data_path)
    cache_path.mkdir(parents=True, exist_ok=True)

if refresh:
    sts = build_survival_times(lf, extra_cols=["nlcd", "Event_ID"]).collect()
    sts.write_parquet(cache_path / "sts.parquet")
else:
    sts = pl.scan_parquet(cache_path / "sts.parquet").collect()

unique_sts = np.concatenate([
    sts.select(pl.col("st")).unique().sort(pl.col("st")).sort("st").to_numpy().flatten(),
    np.array([38])
])


# Build a polars dataframe that is a map from eco, nlcd, and dt to the fitter.hazard(dt)

records = []

for name in outputs:
    output = outputs[name]
    for eco in output['eco']:
        for nlcd in output['nlcd']:
            hazards = output['fitter'].hazard(unique_sts)
            records.extend(
                [
                    {
                        "eco": eco,
                        "nlcd": nlcd,
                        "st": unique_sts[i],
                        "hazard": hazards[i],
                    }
                    for i in range(len(hazards))
                ]
            )
hazard_tbl = pl.from_dicts(records).with_columns(
    pl.col("eco"),
    pl.col("nlcd"),
    pl.col("st").cast(pl.Float32),
    pl.col("hazard").cast(pl.Float32),
)

print(hazard_tbl)

# join survival times with hazard table (THIS TAKE AGES, just an example)
# bps = sts.join(
#     hazard_tbl,
#     left_on=["eco", "nlcd", "st"],
#     right_on=["eco", "nlcd", "st"],
#     how="left",
# ).with_columns(
#     pl.col("hazard").cast(pl.Float32),
# )
# %%
