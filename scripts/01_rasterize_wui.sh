#!/usr/bin/env bash
src="/var/mnt/fastdata02/mtbs/wui/intermediate/wui.gpkg"
dst_dir="/var/mnt/fastdata02/mtbs/wui/cleaned"
declare -a years=("1990" "2000" "2010" "2020")
declare -a flavors=("flag" "class")
for year in "${years[@]}"; do
    for flavor in "${flavors[@]}"; do
        key="wui${flavor}${year}"
        field=$(tr "[:lower:]" "[:upper:]" <<< "${key}")
        set -o xtrace
        # Use -co PIXELTYPE=SIGNEDBYTE because -ot int8 is not working
        gdal_rasterize -l "${key}" -a "${field}" \
            -ot Byte -a_nodata 255 -init 255 -of GTiff \
            -tr 30.0 30.0 -te -2406135.0 218085.0 2308185.0 3222585.0 \
            "${src}" "${dst_dir}/${key}.tif"
        set +o xtrace
    done
done
