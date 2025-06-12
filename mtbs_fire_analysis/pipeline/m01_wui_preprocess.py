import glob

import dask
import dask.array as da
import dask.dataframe as dd
import dask_geopandas as dgpd
import numpy as np
from dask.distributed import Client, LocalCluster
from paths import INTERMEDIATE_WUI, ROOT_TMP_DIR

dask.config.set(
    {
        "distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0,
        "distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_": 0,
    }
)

PART_OUT_FMT = "{name}_part{ipart:02}.pqt"
PART_GLOB_FMT = "{name}_part*.pqt"


def get_part(path, src_col, dst_col, ipart, nparts=20):
    gdf = (
        dgpd.read_file(path, npartitions=nparts)[[src_col, "geometry"]]
        .partitions[ipart]
        .compute()
    )
    gdf[src_col] = (gdf[src_col] > 0).astype("uint8")
    return gdf.rename(columns={src_col: dst_col})


def build_save_part(
    ipart, in_path=None, src_name=None, dst_name=None, nparts=20
):
    ipart = ipart.item()
    out_path = ROOT_TMP_DIR / PART_OUT_FMT.format(name=dst_name, ipart=ipart)
    if out_path.exists():
        print(
            f"{ipart}: Output path {out_path} already exists. Skipping.\nDone"
        )
        return np.array([ipart])
    print(
        f"{ipart}: Loading {ipart + 1}/{nparts} from {in_path}|name={src_name}"
    )
    part = get_part(in_path, src_name, dst_name, ipart, nparts=nparts)
    print(f"{ipart}: Done. {len(part) = }")
    # print(f"{ipart}: Disolving")
    # part = part.dissolve(by=dst_name)
    # print(f"{ipart}: Done. {len(part) = }")
    # print(f"{ipart}: Exploding")
    # part = part.explode()
    print(f"{ipart}: Done. {len(part) = }")
    print(f"{ipart}: Saving to {out_path}")
    part.to_parquet(out_path)
    print(f"{ipart}: Done")
    return np.array([ipart])


def build_wui_bool_parts_on_disk(
    in_path, src_name, dst_name, nparts, batch_size, client
):
    iparts = da.arange(nparts, chunks=1)
    iparts = iparts.map_blocks(
        build_save_part,
        in_path=in_path,
        src_name=src_name,
        dst_name=dst_name,
        nparts=nparts,
        meta=np.array((), dtype=iparts.dtype),
    )
    for i in range(0, nparts, batch_size):
        batch = iparts[i : i + batch_size]
        print(batch.compute())
        # Free memory because dask tries to keep as much as it can and never
        # reuse it. Some might call this a leak.
        client.restart()


def _get_frame_from_parts(name):
    part_files = sorted(
        glob.glob(str(ROOT_TMP_DIR / PART_GLOB_FMT.format(name=name)))
    )
    parts = [
        dgpd.read_parquet(pf, npartitions=1).reset_index() for pf in part_files
    ]
    return dd.concat(parts)


def assemble_wui_bool_from_parts_and_save(name, path):
    print(f"Assembling {name}")
    print("Loading parts")
    gdf = _get_frame_from_parts(name).compute()
    # print("Spatial shuffling")
    # meta = gdf._meta.dissolve(by=name, as_index=False)
    # gdf = gdf.spatial_shuffle()
    # gdf = gdf.map_partitions(
    #     gpd.GeoDataFrame.dissolve, by=name, as_index=False, meta=meta
    # )
    print("Saving")
    gdf.to_file(path, driver="GPKG", layer=name)


def main():
    with LocalCluster(n_workers=2) as cluster, Client(cluster) as client:
        nparts = 20
        batch_size = 10
        for year in range(1990, 2030, 10):
            src_name = f"wui_flag_{year}"
            dst_name = f"wui_bool_{year}"
            build_wui_bool_parts_on_disk(
                INTERMEDIATE_WUI,
                src_name,
                dst_name,
                nparts,
                batch_size,
                client,
            )
        for year in range(1990, 2030, 10):
            name = f"wui_bool_{year}"
            assemble_wui_bool_from_parts_and_save(name, INTERMEDIATE_WUI)
            client.restart()


if __name__ == "__main__":
    main()
