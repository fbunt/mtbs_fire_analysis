# Analysis TODOs (mtbs_fire_analysis/analysis)

This list tracks work specific to the analysis package (renewal likelihoods, HLH, SciPy-backed models, pipelines).

Last updated: 2025-11-10

## ✅ Recently completed
- [x] Add `BaseLifetime.copy()` (constructor from `params` with deepcopy fallback)
- [x] Make `BaseLifetime.fit(...)` tolerate extra HLH-style kwargs via `**kwargs`
- [x] Add `SciPyParametric.rvs(size, rng)` delegating to frozen SciPy rv
- [x] Closed-form `Weibull.tail_survival(W)` using upper incomplete gamma
- [x] Smoke test `all_sensors_weibull` uses generic `.params` across models

## 🔜 Next actions
- [ ] Style and lint cleanup in `analysis/scipy_dist.py`
  - [ ] Rename locals to PEP 8 (e.g., `S` → `s`, `W`/`W_arr` → `w`/`w_arr`)
  - [ ] Replace `_QUAD_OPTS = dict(...)` with literal `{"...": ...}`
  - [ ] Remove unused imports (`dataclass`, `Iterable`) and long-line wraps
  - [ ] Decide on alias conventions (e.g., `HLH`, `WBD`) vs linter config
- [ ] HLH style polish
  - [ ] Rename mixed-case math tokens to snake_case (e.g., `dlogS_dh` → `dlog_s_dh`)
  - [ ] Wrap long comments/docstrings under 79 chars
  - [ ] Resolve any remaining ruff warnings in `hlh_dist.py`
- [ ] Add more SciPy distributions with closed forms where possible
  - [ ] Exponential: analytic mean and tail integral
  - [ ] Gamma: analytic tail via regularized gamma
  - [ ] LogNormal: tail via `sf` integral or approximation
- [ ] Add Inverse Gaussian distribution (separate code path)
  - [ ] Implement `InverseGauss` subclass (params: `mu`, `lam`) with `pdf/sf/mean`
  - [ ] Provide `rvs(size, rng)` via Michael–Schucany–Haas algorithm
  - [ ] Use numeric tail integral default; add soft bounds/penalty for fit
  - [ ] Add polished smoke test in `statistical_tests/inverse_gauss/`
- [ ] Restructure statistical tests by distribution
  - [ ] Create per-distribution subfolders: `hlh/`, `weibull/`, `inverse_gauss/`
  - [ ] Move existing HLH/Weibull tests after IG smoke works as template
- [ ] Optional: closed-form `expected_hazard_ge(W)` for Weibull using gamma ratio
- [ ] Robust fit diagnostics
  - [ ] Return/attach `OptimizeResult` and best params on failure
  - [ ] Small unit tests for failure/edge cases (bounds, NaNs)
- [ ] Test coverage for likelihood pieces
  - [ ] Events, right-censor, initial gaps (forward-recurrence), empty windows
  - [ ] Vectorized counts handling and zero-length inputs
- [ ] Performance checks
  - [ ] Bench tail integrals (numeric vs closed form) for typical W ranges
  - [ ] Add micro-bench harness; document guidance in README
- [ ] Reporting
  - [ ] Provide a tiny adapter or schema for cross-model reporting (e.g., HLH: `hazard_inf`, `half_life`; Weibull: `shape`, `scale`)
  - [ ] Ensure `params` order and names are stable for downstream tables
- [ ] Docs
  - [ ] README for `analysis/` detailing model API, likelihood terms, and examples
  - [ ] Document random generation API (`rvs(size, rng)`) and reproducibility

## ✅ Sanity checks and validation (distributions)
- [x] IG sampler mean sanity: `IG.rvs` sample mean ≈ `mu` within 0.1% at high `lam` (tests/test_inverse_gauss_rvs_mean.py)
- [x] dtctutet vs naive mean consistency: `mean_10pc`, `mean_mean`, `mean_90pc` within 0.1% across HLH/Weibull/IG (tests/test_repeat_mean_vs_naive.py)

## 🧪 Planned comprehensive sanity checks
- [ ] Parameter mapping checks
  - [ ] Inverse Gaussian: verify `mu_s = mu/lam`, `scale = lam` by comparing sample mean/variance against SciPy formulas across a grid of `(mu, lam)`
  - [ ] Weibull: sample-mean ≈ `scale * Gamma(1 + 1/shape)` across several `(shape, scale)`
  - [ ] HLH: sample-mean from `rvs` ≈ analytic `mean()` over a grid of `(hazard_inf, half_life)`
- [ ] Renewal simulator invariants
  - [ ] Check `n_ut == n_ct` and document why swap adds only a constant when equal
  - [ ] Verify expected events per pixel ≈ `W / mu` at equilibrium; naive mean ≈ `mu`
- [ ] Likelihood pieces
  - [ ] Empty-window stability: numeric `tail_survival(W)` monotone in `W`, finite; compare against closed forms where available (Weibull)
  - [ ] Forward-gap vs right-censor separation: small synthetic cases where only one term present
- [ ] Health checks
  - [ ] `quick_health()` no false positives on typical fits; flags bound hits and non-finite means
- [ ] Reproducibility
  - [ ] Ensure `rvs(size, rng)` yields repeatable sequences with fixed seed across all models

## HLH alignment and refactor tasks
- [x] HLH: make gradient fit optional and disabled by default
- [x] HLH: add soft-penalty in finite-difference objective for stability
- [x] HLH: add `copy()` to mirror base API
- [ ] HLH: accept kwargs-compatible `fit(...)` signature across call sites
  - [ ] Optionally switch to `BaseLifetime.fit` after breaking circular import (or extract base class)
- [ ] HLH: add `_soft_penalty` and `default_bounds` methods to mirror SciPy-backed subclasses
- [ ] HLH: minimal wrapper `expected_hazard_ge(w: float)` that forwards to detailed version
- [ ] HLH: migrate `neg_log_likelihood` to shared BaseLifetime implementation (after base extraction)
- [ ] Tests: run repeat and single-run smoke tests for HLH with new defaults and compare parameter recovery

## ▶️ How to run the current smoke test

Using uv (reads dependencies from `pyproject.toml`):

```bash
uv run python -m mtbs_fire_analysis.analysis.statistical_tests.all_sensors_weibull
```

This prints fitted parameters and mean estimates for several datasets derived from simulated truth.

## ↗️ Related lists (repo-wide)
- Project TODO index: see `../../TODO.md`
- Rupert’s todo: `../../RupertTodo.md`
- Progress log: `../../memory-bank/progress.md`
- Architecture and pipelines: `../../docs/ARCHITECTURE.md`, `../../docs/ANALYSIS_PIPELINES.md`
