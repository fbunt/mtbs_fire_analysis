for year in {2018..2023}; do
    echo "Running fits for year: $year"
    uv run python mtbs_fire_analysis/analysis/a20_hlh_fits.py --max_date "${year}-01-01"
done