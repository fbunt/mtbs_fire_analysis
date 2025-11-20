for year in {2018..2022}; do
    echo "Scoring burn probability for year: $year"
    uv run python mtbs_fire_analysis/analysis/a50_score_bp.py --year $year --config_path mtbs_fire_analysis/analysis/configs.yaml --bp_path /fire_analysis_data/data/bp_eco_lvl_3_${year}.tif --out_path /fire_analysis_data/data/results/scores_${year}.parquet
done