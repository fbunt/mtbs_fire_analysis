# %%
import polars as pl
score_dfs = {}
ref_dfs = {}
for year in range(2018,2019):
    score_dfs[year] = pl.read_parquet(f"/fire_analysis_data/data/results/scores_{year}.parquet")
    ref_dfs[year] = pl.read_parquet(f"/fire_analysis_data/data/results/scores_ref_{year}.parquet")

# %%
