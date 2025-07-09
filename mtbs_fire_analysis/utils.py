import contextlib
import itertools
import pprint
import subprocess

from dask.diagnostics import ProgressBar


def flatmap(func, iterable):
    return itertools.chain.from_iterable(map(func, iterable))


def protected_raster_save_with_cleanup(
    raster, path, skip_if_exists=True, progress=True, **save_opts
):
    if skip_if_exists and path.exists():
        print("Already exists. Skipping.")
        return
    try:
        with ProgressBar() if progress else contextlib.nullcontext():
            raster.save(path, tiled=True, **save_opts)
    except (Exception, KeyboardInterrupt) as err:
        print("Removing unfinished file")
        path.unlink()
        raise err


def stack_rasters_as_vrt(src_paths, dst_path):
    for p in src_paths:
        assert p.exists(), f"{p} does not exist"
    paths = map(str, src_paths)
    if dst_path.exists():
        dst_path.unlink()
    # We could do this by importing gdal but rasterio STRONGLY recommends not
    # doing so because it will probably break rasterio. Thus we do it through
    # a shell call.
    # See: https://rasterio.readthedocs.io/en/stable/topics/switch.html#mutual-incompatibilities
    command = ["gdalbuildvrt", "-separate", str(dst_path)]
    command.extend(paths)
    print(f"Build VRT command:\n{pprint.pformat(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as err:
        print(f"STDOUT\n------\n{err.stdout}")
        print(f"STDERR\n------\n{err.stderr}")
        raise err
