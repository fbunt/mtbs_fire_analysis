for year in {2018..2023}; do
    uv run python mtbs_fire_analysis/analysis/a00_get_histories.py --max_date "${year}-01-01"
done