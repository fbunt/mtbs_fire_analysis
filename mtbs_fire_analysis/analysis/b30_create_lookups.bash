for year in {2018..2023}; do
    echo "Creating lookup for year: $year"
    uv run python mtbs_fire_analysis/analysis/a30_create_lookup.py --max_date "${year}-01-01"
done