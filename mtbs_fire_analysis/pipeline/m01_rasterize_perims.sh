#!/usr/bin/env bash
exe() {
    echo "\$ $@"
    "$@"
}

rasterize() {
    year=$1
    src=$(python -c "from mtbs_fire_analysis.pipeline.paths import PERIMS_BY_YEAR_PATH; print(PERIMS_BY_YEAR_PATH)")
    dst_dir=$(python -c "from mtbs_fire_analysis.pipeline.paths import PERIMS_RASTERS_PATH; print(PERIMS_RASTERS_PATH)")
    dst="${dst_dir}/${year}.tif"

    if [ ! -f "${dst}" ]; then
        exe gdal_rasterize -l "${year}" -a days_since_epoch \
            -ot Int16 -a_nodata -1 -init -1 -of GTiff \
            -tr 30.0 30.0 -te -2406135.0 218085.0 2308185.0 3222585.0 \
            -co TILED=YES -co BIGTIFF=YES -co COMPRESS=LZW \
            "${src}" "${dst}"
    else
        echo "Skipping ${year}"
    fi
}


parallel -j 4 --tag --verbose --linebuffer --keep-order \
    rasterize {} ::: $(seq 1984 2022)
