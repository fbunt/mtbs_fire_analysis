Cline setup for mtbs_fire_analysis

This repo is ready to use with Cline in a VS Code devcontainer. This document provides:
1) Project-specific Cline rules (how to operate safely here)
2) A memory seed to keep key context top-of-mind
3) Ready-to-run workflows/commands you can trigger via Cline
4) A deep-planning roadmap Cline can follow to analyze and improve the repo


1) Cline rules for this repo

Environment and tools
- Python: 3.12 (see pyproject.toml)
- Package manager: uv (use uv sync to install; .venv lives in the workspace)
- Linters/formatters: ruff + black (line length 79, isort via ruff)
- Shell: bash
- Working directory: /workspaces/mtbs_fire_analysis
- Data mount: /fire_analysis_data (from devcontainer.json binds)
- GPU: NVIDIA GPU is exposed in the container (CUDA runtime base image)

Operating rules
- Auto-approve: OFF. Always ask before:
  - Running long or GPU-heavy jobs (bash pipelines, HLH fits, rasterization)
  - Writing to outputs/ or copying rasters
  - Installing system packages or changing container config
  - Deleting/moving files or overwriting existing content
- Commands:
  - Prefer uv run to execute Python (ensures correct venv and deps)
  - Prefer python -m package.module where practical to avoid path issues
  - Log long runs to outputs/logs/, e.g. uv run python ... 2>&1 | tee -a outputs/logs/run_YYYYMMDD_HHMM.log
- Editing code:
  - Use ruff + black conventions (79 cols). Keep imports organized (ruff/isort).
  - Avoid creating large tracked assets in the repo; use outputs/ or /fire_analysis_data for heavy files
- Geospatial specifics:
  - GDAL/PROJ are installed in Docker. If adding new geospatial deps, confirm compatibility with Ubuntu 22.04.
- Web/docs tools:
  - The devcontainer lacks some headless browser libs; Cline web_fetch may fail. Avoid web_fetch unless explicitly approved and container packages are updated to include libatk and friends.

Safety checks before runs
- Confirm available disk/memory and expected runtime for large jobs.
- Validate input paths (e.g., raster and parquet sources) exist before computing.
- If a script writes many files, create or target a dated subfolder in outputs/.


2) Memory seed (project essentials)

Scope and purpose
- Analyze MTBS fire histories and related geospatial layers to model time-to-fire and inter-fire intervals across ecoregions/land cover classes.
- Fit distributions/hazard models (“HLH fits”), generate lookup tables, compute burn probabilities, and produce statistics/plots.

Key directories
- mtbs_fire_analysis/analysis: Analysis scripts (a00..a50) and bash pipelines (b00..b50), HLH fitting, scoring, statistics, plotting.
- mtbs_fire_analysis/pipeline: Data preprocessing/rasterization/extraction scripts (m00..m30) and paths utilities.
- CA_Simulation: A separate cellular automata style fire spread simulator producing example videos (not the main analysis pipeline).
- data/: Source files (e.g., ecoregions shapefile, parquet summaries).
- outputs/: Results of HLH fits, lookup tables, plots, parquet outputs.

Conventions
- Python 3.12; uv for dependency management (uv sync on create/start).
- Style: ruff + black (79 cols), isort via ruff; known_first_party = ["mtbs_fire_analysis"].
- Many scripts are numbered to indicate pipeline order (a00..a50, b00.., c00.. etc.).

Open concerns (from 10ThingsIHateAboutThisProject.md)
- NLCD assignment choices for survival times and intervals are heuristic.
- Many small event histories (<10 pixels) exist; need a better auditing/QGIS inspection workflow.
- HLH distribution implementation should be merged into distributions with a registry and interfaces for extensibility.

Constraints and cautions
- Large raster operations are expensive; ask before large/multi-hour runs.
- Don’t commit large outputs. Keep reproducible pipelines documented.


3) Workflows and commands to trigger via Cline

Environment
- Install/refresh deps:
  - uv sync

Core analysis chain (Python)
- Get histories:
  - uv run python mtbs_fire_analysis/analysis/a00_get_histories.py
- Create config for fits:
  - uv run python mtbs_fire_analysis/analysis/a10_create_config.py
- Run HLH fits (per-ecoregion/land cover):
  - uv run python mtbs_fire_analysis/analysis/a20_hlh_fits.py
- Build lookup tables:
  - uv run python mtbs_fire_analysis/analysis/a30_create_lookup.py
- Full statistics:
  - uv run python mtbs_fire_analysis/analysis/a31_full_statistics.py
- Create burn probability surfaces:
  - uv run python mtbs_fire_analysis/analysis/a40_create_bp.py
- Score burn probability:
  - uv run python mtbs_fire_analysis/analysis/a50_score_bp.py

Bash pipelines (wrapper scripts; confirm paths and runtime)
- Histories:
  - bash mtbs_fire_analysis/analysis/b00_get_histories.bash
- Fits:
  - bash mtbs_fire_analysis/analysis/b20_run_fits.bash
- Lookups:
  - bash mtbs_fire_analysis/analysis/b30_create_lookups.bash
- Burn probabilities:
  - bash mtbs_fire_analysis/analysis/b40_create_bps.bash
- Score BPs:
  - bash mtbs_fire_analysis/analysis/b50_score_bps.bash
- Score reference BPs:
  - bash mtbs_fire_analysis/analysis/b51_score_ref_bps.bash
- Compress BPs:
  - bash mtbs_fire_analysis/analysis/c00_compress_bps.bash
- Copy rasters (ask before running; can be heavy):
  - bash mtbs_fire_analysis/analysis/c00_copy_rasters.bash
  - bash mtbs_fire_analysis/analysis/c01_copy_rasters.bash
  - bash mtbs_fire_analysis/analysis/c01_copy_rasters_host.bash

Data pipeline (geospatial preprocessing; ask before heavy rasterization)
- Eco regions preprocessing:
  - uv run python mtbs_fire_analysis/pipeline/m00_eco_regions_preprocess.py
- MTBS rasters preprocessing:
  - uv run python mtbs_fire_analysis/pipeline/m00_mtbs_rasters_preprocess.py
- Perimeters preprocessing:
  - uv run python mtbs_fire_analysis/pipeline/m00_perims_preprocess.py
- States preprocessing:
  - uv run python mtbs_fire_analysis/pipeline/m00_states_preprocess.py
- WUI preprocessing:
  - uv run python mtbs_fire_analysis/pipeline/m00_wui_preprocess.py
- NLCD preprocessing:
  - uv run python mtbs_fire_analysis/pipeline/m01_nlcd_preprocess.py
- Rasterize eco regions/perims/WUI:
  - bash mtbs_fire_analysis/pipeline/m01_rasterize_eco_regions.sh
  - bash mtbs_fire_analysis/pipeline/m01_rasterize_perims.sh
  - bash mtbs_fire_analysis/pipeline/m02_rasterize_wui.sh
- Data extract/join/combine/build spatio-temporal set:
  - uv run python mtbs_fire_analysis/pipeline/m10_data_extract.py
  - uv run python mtbs_fire_analysis/pipeline/m11_join_raster_data.py
  - uv run python mtbs_fire_analysis/pipeline/m20_combine.py
  - uv run python mtbs_fire_analysis/pipeline/m30_build_st.py

Recommended logging pattern for long runs
- mkdir -p outputs/logs
- uv run python mtbs_fire_analysis/analysis/a20_hlh_fits.py 2>&1 | tee -a outputs/logs/hlh_fits_$(date +%F_%H%M).log


4) Deep-planning roadmap (what Cline will do on approval)

Goal: Analyze repository structure, identify improvements, and build clear documentation.

Plan
A) Codebase mapping
- Enumerate modules, scripts, and their roles (analysis/pipeline/CA_Simulation).
- Trace data flow: sources → preprocessing → extraction → modeling → outputs.
- Identify heavy operations and their inputs/outputs for resource planning.

B) API and script interface documentation
- For each numbered script (a00.., m00..), extract:
  - Purpose, inputs/outputs, CLI args (if any), dependencies, runtime considerations.
- Produce ANALYSIS_PIPELINES.md and DATA_PIPELINE.md with execution graphs.

C) HLH modeling refactor plan
- Propose structure to merge HLH implementation into distributions with:
  - Registry of distributions
  - Common base class and SciPy-backed subclass pattern
  - Extensibility for covariates (PH/AFT/AH) via transformation arguments

D) Validation and QA
- Define quick-check workflows:
  - Sample a handful of small event histories for QGIS inspection
  - Unit-style smoke tests for key functions
  - Deterministic seeds/logging for reproducibility

E) Documentation set
- README refresh (high-level workflow and quickstart)
- ARCHITECTURE.md (module layout, data flow diagrams)
- ANALYSIS_PIPELINES.md (a00..a50)
- DATA_PIPELINE.md (m00..m30)
- CONTRIBUTING.md (style, lint, tests, PR review)
- DECISIONS.md (documenting choices like NLCD assignment heuristics)
- TODO.md (from 10ThingsIHate... and new items)

F) Execution safety and reliability
- Propose idempotent outputs with dated folders
- Centralize config (e.g., YAML) and avoid hard-coded paths
- Add lightweight CLI entry points where helpful

Constraints to honor during deep planning
- Do not run heavy jobs without explicit approval
- Avoid web_fetch until container has required libs
- Maintain 79-char, ruff/black style
- Use uv run for Python execution

On approval from the user, Cline will proceed with step A and produce an initial ARCHITECTURE.md and execution graph, then iterate through B–F.


Appendix: Quick commands

- Environment
  - uv sync

- Lint/format (ruff will handle import sort; black line length 79)
  - uv run ruff check .
  - uv run ruff format .

- Run examples
  - uv run python mtbs_fire_analysis/analysis/a10_create_config.py
  - uv run python mtbs_fire_analysis/analysis/a20_hlh_fits.py
  - uv run python mtbs_fire_analysis/analysis/a31_full_statistics.py

- Pipelines
  - bash mtbs_fire_analysis/analysis/b20_run_fits.bash
  - uv run python mtbs_fire_analysis/pipeline/m01_nlcd_preprocess.py
