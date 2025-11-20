for year in {2018..2023}; do
    echo "Creating burn probability for year: $year"
    uv run python mtbs_fire_analysis/analysis/a40_create_bp.py --eco_level 3 --year $year
done