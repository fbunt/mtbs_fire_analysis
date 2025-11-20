# activeContext.md

Tracks current work focus and immediate next steps.
Initialized: 2025-08-14T00:00:00Z

## Current Work Focus
- Consolidate repository-level Cline setup into the memory bank.
- Ensure safety rules and workflows are recorded in memory-bank for Cline to read at task start.
- Archive original docs and tooling under `memory-bank/archive/`.

## Recent Changes
- Created `memory-bank/` scaffold and core files.
- Archived `docs/CLINE_SETUP.md` and `tools/generate_repo_docs.py` to `memory-bank/archive/`.
- Consolidated core environment and usage guidance into `techContext.md` and `productContext.md`.

## Operating Rules (copied / consolidated)
- Auto-approve: OFF. Always ask before:
  - Running long or GPU-heavy jobs (bash pipelines, HLH fits, rasterization)
  - Writing to `outputs/` or copying rasters
  - Installing system packages or changing container config
  - Deleting/moving files or overwriting existing content
- Prefer `uv run` to execute Python to ensure the correct venv and deps.
- Prefer `python -m package.module` when practical to avoid path issues.
- Long runs should be logged to `outputs/logs/` (see techContext.md for examples).
- Keep code style consistent: ruff + black (79 cols) and isort via ruff.

## Workflows & Ready Commands (quick reference)
- Install/refresh deps:
  - `uv sync`
- Core analysis chain:
  - `uv run python mtbs_fire_analysis/analysis/a00_get_histories.py`
  - `uv run python mtbs_fire_analysis/analysis/a10_create_config.py`
  - `uv run python mtbs_fire_analysis/analysis/a20_hlh_fits.py`
  - `uv run python mtbs_fire_analysis/analysis/a30_create_lookup.py`
  - `uv run python mtbs_fire_analysis/analysis/a31_full_statistics.py`
  - `uv run python mtbs_fire_analysis/analysis/a40_create_bp.py`
  - `uv run python mtbs_fire_analysis/analysis/a50_score_bp.py`
- Bash pipeline wrappers (confirm paths & runtime before running):
  - `bash mtbs_fire_analysis/analysis/b00_get_histories.bash`
  - `bash mtbs_fire_analysis/analysis/b20_run_fits.bash`
  - `bash mtbs_fire_analysis/analysis/b30_create_lookups.bash`

## Next Steps
- Populate `projectbrief.md` with a concise mission statement and success criteria.
- Run `tools/generate_repo_docs.py` to generate docs/ARCHITECTURE.md, ANALYSIS_PIPELINES.md, DATA_PIPELINE.md, DECISIONS.md (requires approval).
- Iterate on memory-bank content after reviewing generated docs; record actions in `progress.md`.

## Important Patterns & Preferences
- MUST read all `memory-bank/` files at the start of every task.
- Archive originals before removal; prefer non-destructive edits.
- Use dated output folders for reproducibility and to avoid accidental overwrites.

## Source / provenance
- Consolidated from `docs/CLINE_SETUP.md` (archived at `memory-bank/archive/CLINE_SETUP.md`).
