import shutil
import os
from pathlib import Path
out_dir = Path("mtbs_fire_analysis") / "outputs" / "HLH_Fits_Eco3"
print(out_dir.exists())
copy_dir = Path("/run/media/fire_analysis") / "data" / "cache"
print(copy_dir.exists())
shutil.copy(out_dir /"lookup_table_2022-01-01.parquet",copy_dir / "lookup_table_2022-01-01.parquet")