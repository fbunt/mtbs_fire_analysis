#!/usr/bin/env bash
exe() {
    echo "\$ $@"
    "$@"
}

rasterize() {
    year=$1
    src=$(python -c "from mtbs_fire_analysis.pipeline.paths import PERIMS_BY_YEAR_PATH; print(PERIMS_BY_YEAR_PATH)")
    dst_dir=$(python -c "from mtbs_fire_analysis.pipeline.paths import PERIMS_RASTERS_PATH; print(PERIMS_RASTERS_PATH)")
    dst="${dst_dir}/dse_${year}.tif"

    # Rasterization grid (resolution + extent) is derived from the analysis
    # grid selector, so a coarse run (FIRE_PIXEL_M=120) rasterizes perimeters
    # NATIVELY at that resolution: pixel-CENTER / point-sampled membership (no
    # -at flag => a cell burns iff its centre is inside the perimeter). With
    # FIRE_PIXEL_M unset this is 30 m, byte-identical to the former hardcoded
    # `-tr 30 30 -te -2406135 218085 2308185 3222585`. The extent comes from
    # grid_for_pixel_m's FLOOR grid (NOT the legacy 30 m -te) so the dse lands
    # on exactly geobox_for_pixel_m(N) and the analysis read's alignment gate
    # passes -- reusing the 30 m -te would ceil to +1 row at 120 m.
    read RES TE_L TE_B TE_R TE_T < <(python -c "from mtbs_fire_analysis.defaults import grid_for_pixel_m, pixel_m_from_env; aff, (h, w) = grid_for_pixel_m(pixel_m_from_env()); print(aff.a, aff.c, aff.f + aff.e * h, aff.c + aff.a * w, aff.f)")

    if [ ! -f "${dst}" ]; then
        exe gdal_rasterize -l "${year}" -a days_since_epoch \
            -ot Int16 -a_nodata -1 -init -1 -of GTiff \
            -tr "${RES}" "${RES}" -te "${TE_L}" "${TE_B}" "${TE_R}" "${TE_T}" \
            -co TILED=YES -co BIGTIFF=YES -co COMPRESS=LZW \
            "${src}" "${dst}"
    else
        echo "Skipping ${year}"
    fi
}


# GNU parallel runs each job in a fresh shell, so the functions must be
# exported or every job fails with "rasterize: command not found".
export -f exe rasterize

# Year window is the single source of truth in paths.py (same constants m00
# uses) — bump MTBS_PERIM_YEAR_END there to extend; default 1984-2022 is
# byte-identical to the former hardcoded `seq 1984 2022`.
YEARS=$(python -c "from mtbs_fire_analysis.pipeline.paths import MTBS_PERIM_YEAR_START, MTBS_PERIM_YEAR_END; print(*range(MTBS_PERIM_YEAR_START, MTBS_PERIM_YEAR_END + 1))")

parallel -j 4 --tag --verbose --linebuffer --keep-order \
    rasterize {} ::: ${YEARS}
