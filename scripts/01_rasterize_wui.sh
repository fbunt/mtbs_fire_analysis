#!/usr/bin/env bash
src=$(python -c "from paths import INTERMEDIATE_WUI; print(INTERMEDIATE_WUI)")
dst_dir=$(python -c "from paths import WUI_PATH; print(WUI_PATH)")
declare -a years=("1990" "2000" "2010" "2020")
declare -a flavors=("flag" "class")
for year in "${years[@]}"; do
    for flavor in "${flavors[@]}"; do
        key="wui_${flag}_${year}"
        set -o xtrace
        # Use -co PIXELTYPE=SIGNEDBYTE because -ot int8 is not working
        gdal_rasterize -l wui -a "${key}" \
            -ot Byte -a_nodata 255 -init 255 -of GTiff \
            -tr 30.0 30.0 -te -2406135.0 218085.0 2308185.0 3222585.0 \
            "${src}" "${dst_dir}/${key}.tif"
        set +o xtrace
    done
done
