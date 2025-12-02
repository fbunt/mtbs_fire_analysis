
for year in {2018..2023}; do
    echo "Compressing burn probability for year: $year"
    gdal_translate -co COMPRESS=ZSTD -co ZSTD_LEVEL=1 -co TILED=YES \
        "/run/media/fire_analysis/data/burn_probabilities/bp_eco_lvl_3_${year}.tif" \
        "/run/media/fire_analysis/data/burn_probabilities/compressed/bp_eco_lvl_3_${year}.tif"
done

