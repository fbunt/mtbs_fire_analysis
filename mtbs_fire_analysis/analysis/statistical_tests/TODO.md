# Statistical tests – TODO

A running list of sanity checks and extensions to keep distributions healthy and comparable.

- IG sampling and parameterization
  - [x] Verify `rvs` mean matches μ statistically (z-score based)
  - [ ] Cross-check variance vs μ^3/λ with confidence bounds
  - [ ] Tail sanity: compare empirical survival vs model `sf` across quantiles
- Repeat-fit consistency
  - [x] Centralize default truth and simulation settings
  - [x] Unify pipeline via shared helper; correct ut/ct handling
  - [ ] Add CI-smoke runs for a tiny `iterations` count per dist
  - [ ] Track `quick_health().issues` frequency and histogram per mode
- Naive vs dtctutet checks
  - [x] Unit test: dtctutet mean stats (10pc/mean/90pc) ~ naive within 0.1%
  - [ ] Investigate sensitivity to `pre_window` (esp. long-tail IG)
  - [ ] Optionally compute and plot denom components (dt vs ct counts)
- Summary robustness
  - [x] Numeric coercion and NaN-robust aggregation
  - [ ] Add bootstrap CIs on key stats in overview scripts
- Distribution overview
  - [x] Table of mean/median/variance/skew/kurtosis and percentiles
  - [ ] Compare sample stats to analytic values when available (e.g., Weibull)
  - [ ] Add configurable sample size and random seed via CLI args
