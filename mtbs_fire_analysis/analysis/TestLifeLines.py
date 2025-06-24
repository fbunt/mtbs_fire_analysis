from pathlib import Path

import yaml

configs = Path("mtbs_fire_analysis") / "analysis" / "configs.yaml"

with open(configs) as f:
    config_data = yaml.safe_load(f)

configs = config_data["main"]


data_path = Path("mtbs_fire_analysis") / "data"

pixel_counts_path = data_path / "pixel_counts.csv"
