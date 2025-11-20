# techContext.md

Technologies, tooling, and constraints.
Initialized: 2025-08-14T00:00:00Z

## Environment & Tools
- Python: 3.12 (see pyproject.toml)
- Package manager: uv (use `uv sync` to install; `.venv` lives in the workspace)
- Execution: prefer `uv run` to ensure correct venv and deps
- Linters/formatters: ruff + black (79 cols). isort via ruff
- Shell: bash
- Working directory: /workspaces/mtbs_fire_analysis (devcontainer)
- Data mount (devcontainer): /fire_analysis_data
- GPU: NVIDIA GPU may be available in the container (CUDA runtime base image) — verify before use

## Development Setup
- Repository includes a devcontainer and Dockerfile for consistent environments.
- Use the devcontainer or a venv created by `uv` for development.
- Recommended logging pattern for long runs:
  - mkdir -p outputs/logs
  - uv run python mtbs_fire_analysis/analysis/a20_hlh_fits.py 2>&1 | tee -a outputs/logs/hlh_fits_$(date +%F_%H%M).log

## Operating Rules (safety / approvals)
- Auto-approve: OFF. Always ask before:
  - Running long or GPU-heavy jobs (bash pipelines, HLH fits, rasterization)
  - Writing to `outputs/` or copying rasters
  - Installing system packages or changing container config
  - Deleting/moving files or overwriting existing content
- Confirm available disk, memory, and expected runtime before heavy jobs.
- Validate input paths (rasters, parquet) exist before computing.
- If a script writes many files, target a dated subfolder in `outputs/`.

## Geospatial considerations
- GDAL/PROJ are installed in Docker. If adding new geospatial deps, confirm compatibility with Ubuntu 22.04.
- Large raster operations are expensive; plan and approve before running.

## Web/docs tooling caveats
- Devcontainer lacks some headless browser libs; `web_fetch` may fail. Avoid `web_fetch` unless approved and container packages are updated (libatk, etc).

## Dependencies & Constraints
- See `pyproject.toml` for primary dependencies; analysis-specific deps may be listed in docs.
- Data files can be large — prefer parquet/streaming where possible.
- Long-running analyses may require batch execution; use provided bash wrappers where appropriate.

## Source
- Consolidated from `docs/CLINE_SETUP.md` (archived at `memory-bank/archive/CLINE_SETUP.md`).
