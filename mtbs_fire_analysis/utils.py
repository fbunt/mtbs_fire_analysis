import contextlib
import itertools

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
