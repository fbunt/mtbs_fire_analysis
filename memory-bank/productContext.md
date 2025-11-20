# productContext.md

Describes why the project exists and how it should work.
Initialized: 2025-08-14T00:00:00Z

## Scope and purpose
- Analyze MTBS fire histories and related geospatial layers to model time-to-fire and inter-fire intervals across ecoregions and land-cover classes.
- Fit distributions/hazard models (“HLH fits”), generate lookup tables, compute burn probabilities, and produce statistics and plots.
- Support reproducible pipelines for preprocessing, model fitting, lookup generation, and scoring.

## Key directories / responsibilities
- `mtbs_fire_analysis/analysis`: Analysis scripts (a00..a50) and bash pipelines (b00..b50) — HLH fitting, scoring, statistics, plotting.
- `mtbs_fire_analysis/pipeline`: Geospatial preprocessing and rasterization scripts (m00..m30) and path utilities.
- `CA_Simulation/`: Cellular automata fire spread simulator and demonstration outputs (separate from the main analysis pipeline).
- `data/`: Source data (ecoregions shapefile, parquet summaries, pixel counts).
- `outputs/`: Results (fits, lookup tables, plots, parquet outputs).

## Conventions
- Python 3.12; use `uv` for dependency management (run `uv sync` to install; use `uv run` to execute Python where practical).
- Style: ruff + black (79 cols). isort via ruff; known_first_party = ["mtbs_fire_analysis"].
- Prefer `python -m package.module` or `uv run python ...` to avoid path issues.
- Numbered scripts follow pipeline order (a00..a50, m00..m30) to indicate typical execution sequence.

## Open concerns (seeded)
- NLCD assignment choices for survival times and intervals are heuristic and should be documented/validated.
- Many very small event histories (<10 pixels) exist; propose a QGIS/auditing workflow to inspect and triage these.
- HLH distribution implementation should be consolidated into `distributions` with a registry and extensible interfaces.

## Usage guidance (short)
- Long runs and heavy raster operations require explicit approval (confirm disk, memory, expected runtime).
- Log long runs to `outputs/logs/`, e.g. `uv run python ... 2>&1 | tee -a outputs/logs/run_YYYYMMDD_HHMM.log`.
- Avoid committing large output files into the repo; use `outputs/` or external data mounts.

## Source
- Consolidated from `docs/CLINE_SETUP.md` (archived at `memory-bank/archive/CLINE_SETUP.md`).
