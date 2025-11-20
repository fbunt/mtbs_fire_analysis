Architecture overview

    Project purpose
    - Analyze MTBS fire histories and geospatial layers to model time-to-fire and inter-fire intervals; fit HLH distributions; produce lookup tables, burn probabilities, statistics, and plots.

    Repository structure (selected)
    - mtbs_fire_analysis/analysis: Analysis scripts a00..a50 and bash pipelines; HLH fits, scoring, statistics, plotting.
    - mtbs_fire_analysis/pipeline: Geospatial preprocessing and rasterization scripts m00..m30.
    - CA_Simulation: Cellular automata style fire spread examples (separate from primary pipeline).
    - data/: Sources (e.g., ecoregions shapefile, parquet summaries).
    - outputs/: Model outputs (fits, lookups, plots, parquet).

    Entrypoints detected (__main__ guards)
    - Analysis scripts with CLI:
    - mtbs_fire_analysis/analysis/a00_get_histories.py
- mtbs_fire_analysis/analysis/a20_hlh_fits.py
- mtbs_fire_analysis/analysis/a21_hlh_bootstraps.py
- mtbs_fire_analysis/analysis/a30_create_lookup.py
- mtbs_fire_analysis/analysis/a40_create_bp.py
- mtbs_fire_analysis/analysis/a50_score_bp.py
- mtbs_fire_analysis/analysis/calc_burn_probability.py
- mtbs_fire_analysis/analysis/plot_dt.py
- mtbs_fire_analysis/analysis/score_burn_probability.py
- mtbs_fire_analysis/analysis/statistical_tests/all_sensors_repeat.py
- mtbs_fire_analysis/analysis/statistical_tests/cell_auto_test.py
- mtbs_fire_analysis/analysis/t25.py
- mtbs_fire_analysis/analysis/table_2.py

    - Pipeline scripts with CLI:
    - mtbs_fire_analysis/pipeline/m00_eco_regions_preprocess.py
- mtbs_fire_analysis/pipeline/m00_mtbs_rasters_preprocess.py
- mtbs_fire_analysis/pipeline/m00_perims_preprocess.py
- mtbs_fire_analysis/pipeline/m00_states_preprocess.py
- mtbs_fire_analysis/pipeline/m00_wui_preprocess.py
- mtbs_fire_analysis/pipeline/m01_nlcd_preprocess.py
- mtbs_fire_analysis/pipeline/m01_wui_preprocess.py
- mtbs_fire_analysis/pipeline/m10_data_extract.py
- mtbs_fire_analysis/pipeline/m11_join_raster_data.py
- mtbs_fire_analysis/pipeline/m20_combine.py
- mtbs_fire_analysis/pipeline/m30_build_st.py
- mtbs_fire_analysis/pipeline/pixel_counts.py

    Suggested execution flow (high level)
    1) Data pipeline (m00..m30): preprocess vector layers, rasterize where needed, extract/join data, build spatio-temporal dataset
    2) Analysis (a00..a50): extract histories → configure → fit HLH → build lookups → compute stats → produce burn probability → score

    Conventions
    - Python 3.12; uv for dependencies; ruff + black (79 cols) for style
    - Prefer 'uv run python path/to/script.py'
    - Heavy geospatial steps may require time and disk; log outputs in outputs/logs
