#!/usr/bin/env bash
exe() {
    echo "\$ $@"
    "$@"
}

rasterize_eco() {
    eco_level=$1
    name="eco_lvl_${eco_level}"
    src=$(python -c "from mtbs_fire_analysis.pipeline.paths import ECO_REGIONS_PATH; print(ECO_REGIONS_PATH)")
    dst_dir=$(python -c "from mtbs_fire_analysis.pipeline.paths import ECO_REGIONS_RASTER_PATH; print(ECO_REGIONS_RASTER_PATH)")
    dst="${dst_dir}/${name}.tif"

    out_type="Byte"
    nodata="-1"
    case $eco_level in
        1)
            out_type="Int8"
            nodata="-1"
            ;;
        2)
            out_type="Byte"
            nodata="255"
            ;;
        3)
            out_type="Int16"
            nodata="-1"
            ;;
    esac

    if [ ! -f "${dst}" ]; then
        exe gdal_rasterize -l eco_regions -a "${name}" \
            -ot $out_type -a_nodata $nodata -init $nodata -of GTiff \
            -tr 30.0 30.0 -te -2406135.0 218085.0 2308185.0 3222585.0 \
            -co TILED=YES -co COMPRESS=ZSTD -co ZSTD_LEVEL=1 \
            "${src}" "${dst}"
    else
        echo "Skipping eco level ${eco_level}"
    fi
}


parallel --tag --verbose --linebuffer --keep-order \
    rasterize_eco {} ::: 1 2 3
