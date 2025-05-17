#!/usr/bin/env bash
exe() {
    echo "\$ $@"
    "$@"
}

src=$(python -c "from mtbs_fire_analysis.pipeline.paths import INTERMEDIATE_WUI; print(INTERMEDIATE_WUI)")
dst_dir=$(python -c "from mtbs_fire_analysis.pipeline.paths import WUI_PATH; print(WUI_PATH)")
declare -a years=("1990" "2000" "2010" "2020")
declare -a flavors=("flag" "class")
for year in "${years[@]}"; do
    for flavor in "${flavors[@]}"; do
        key="wui_${flavor}_${year}"
        # Use Byte because Int8 is not working for some reason
        exe gdal_rasterize -l wui -a "${key}" \
            -ot Byte -a_nodata 255 -init 255 -of GTiff \
            -tr 30.0 30.0 -te -2406135.0 218085.0 2308185.0 3222585.0 \
            "${src}" "${dst_dir}/${key}.tif"
    done
    key="wui_bool_${year}"
    exe gdal_rasterize -l "${key}" -a "${key}" \
        -ot Byte -a_nodata 255 -init 255 -of GTiff \
        -tr 30.0 30.0 -te -2406135.0 218085.0 2308185.0 3222585.0 \
        "${src}" "${dst_dir}/${key}.tif"
done


src_dir=$(python -c "from mtbs_fire_analysis.pipeline.aths import WUI_PATH; print(WUI_PATH)")
dst_dir=$(python -c "from mtbs_fire_analysis.pipeline.aths import WUI_PATH; print(WUI_PATH)")
max_dist=$(python -c "from dmtbs_fire_analysis.pipeline.efaults import DEFAULT_PROX_MAX_DIST; print(DEFAULT_PROX_MAX_DIST)")
for year in "${years[@]}"; do
    src="${src_dir}/wui_bool_${year}.tif"
    dst="${dst_dir}/wui_prox_${year}.tif"
    exe gdal_proximity "${src}" "${dst}" \
        -values 1 -distunits GEO -maxdist "${max_dist}"
done
