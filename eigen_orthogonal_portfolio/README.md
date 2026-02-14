# Eigen Orthogonal Portfolio

This project demonstrates eigen-portfolios and builds a **long-only** equity portfolio where **component return series** are as orthogonal (uncorrelated) as possible under practical constraints.

It implements a strict rolling no-lookahead backtest using US large-cap equities from `yfinance` adjusted close prices.

You can also run on:
- the full current S&P 500 constituent list, or
- the OMXS30 (Stockholm 30),
- the DAX40 (Germany 40),
and compare against a benchmark:
- `XLG` ETF by default for S&P 500 runs,
- `EXS1.DE` ETF by default for DAX40 runs,
- `XACT-OMXS30.ST` ETF by default for OMXS30 runs,
- market-cap-weighted synthetic benchmark for other universes.

## Core idea

Given a rolling covariance estimate \(C_t\):

1. Compute eigendecomposition \(C_t = V_t \Lambda_t V_t^\top\).
2. Build unconstrained eigen-portfolios (`raw`) for diagnostics; their in-sample component returns are near-diagonal in covariance/correlation.
3. Build long-only approximate component portfolios by solving, for each component \(k\):

\[
\min_{w \ge 0,\; \mathbf{1}^\top w = 1}
\|w - v_k\|_2^2 + \gamma w^\top C_t w
\]

4. Blend long-only components into final long-only asset weights:

\[
w_t = \widetilde{W}_t a_t, \quad a_t \ge 0,\; \mathbf{1}^\top a_t = 1
\]

where `a_t` is either:
- `ECB` (default): component risk-parity style blend (equalize approximate component variance contributions), or
- `ECB-EqualA`: equal component blend.

## What “orthogonal returns” means here

Orthogonality is measured on **component return series** \(g_t = \widetilde{W}^\top r_t\), not asset pairwise correlations.

For ECB strategies we track:

1. In-sample diagonality: mean absolute off-diagonal of `corr(G_window)`.
2. Out-of-sample diagonality: mean absolute off-diagonal of `corr(g_hold)` each hold period.
3. Stability: eigenvector sign alignment and turnover.

## Rebalance / no-lookahead policy

- Covariance window: `252` trading days by default.
- Rebalance: **first trading day of each month**.
- Strict no-lookahead at rebalance date `t`:
  - estimation uses returns `t-window ... t-1` only,
  - new weights apply from day `t` onward.

## Implemented strategies

- `ew`: equal weight across assets.
- `mv_lo`: long-only minimum variance.
- `ecb`: Eigen-Component Balanced with component risk-parity blend.
- `ecb_equala`: ECB with equal blend weights.
- `market_cap`: market-cap-weighted benchmark (uses `marketCap` from Yahoo metadata).

For `--universe sp500`, the CLI compares against an ETF benchmark (`XLG` by default) instead of the synthetic `market_cap` strategy.

## Covariance estimators

- `sample`
- `ledoitwolf` (uses `sklearn` if installed; otherwise falls back)
- `shrink` with
  \(C_{shrunk} = (1-\delta)C + \delta\mu I\), \(\mu = \mathrm{tr}(C)/N\)

All covariance outputs are symmetrized and forced PSD by eigenvalue clipping.

## Project layout

```
eigen_orthogonal_portfolio/
  pyproject.toml
  README.md
  src/eop/
    __init__.py
    data.py
    preprocess.py
    cov.py
    eigen.py
    components.py
    optimizer.py
    strategies.py
    backtest.py
    metrics.py
    plots.py
    cli.py
  tests/
    test_cov.py
    test_eigen.py
    test_components.py
    test_backtest.py
  outputs/
```

## Installation

From `eigen_orthogonal_portfolio/`:

```bash
pip install -e .
pip install pytest
```

Optional for Ledoit-Wolf estimator:

```bash
pip install scikit-learn
```

## Run

```bash
python -m eop.cli \
  --universe sp500 \
  --start 2015-01-01 \
  --end 2025-12-31 \
  --window 252 \
  --rebalance monthly \
  --k 5 \
  --strategy ecb \
  --cov sample \
  --tcost_bps 5 \
  --winsorize none
```

OMXS30 example:

```bash
python -m eop.cli \
  --universe omxs30 \
  --omxs30_benchmark_ticker XACT-OMXS30.ST \
  --start 2015-01-01 \
  --end 2025-12-31 \
  --window 252 \
  --rebalance monthly \
  --k 5 \
  --strategy ecb \
  --cov sample \
  --tcost_bps 5 \
  --winsorize none
```

DAX40 example:

```bash
python -m eop.cli \
  --universe dax40 \
  --dax40_benchmark_ticker EXS1.DE \
  --start 2015-01-01 \
  --end 2025-12-31 \
  --window 252 \
  --rebalance monthly \
  --k 5 \
  --strategy ecb \
  --cov sample \
  --tcost_bps 5 \
  --winsorize none
```

Notes:
- `--strategy` selects the focus strategy, but the CLI backtests all five strategies for side-by-side comparison outputs.
- `--winsorize p01_p99` clips each asset return series at 1st/99th percentiles.
- `--universe sp500` loads S&P 500 constituents (yfinance/ETF holdings first, then Wikipedia fallback, then cache).
- For `--universe sp500`, default is `--sp500_top_n 50` (top 50 by Yahoo market cap). Set `--sp500_top_n 0` to use all fetched S&P 500 symbols.
- For `--universe sp500`, benchmark ETF ticker is set by `--sp500_benchmark_ticker` (default `XLG`).
- `--universe omxs30` loads OMXS30 constituents (Wikipedia with cache and built-in fallback list).
- For `--universe omxs30`, benchmark ETF ticker is set by `--omxs30_benchmark_ticker` (default `XACT-OMXS30.ST`).
- `--universe dax40` loads DAX40 constituents (Wikipedia with cache and built-in fallback list).
- For `--universe dax40`, benchmark ETF ticker is set by `--dax40_benchmark_ticker` (default `EXS1.DE`).
- `--sp500_tickers_file`, `--omxs30_tickers_file`, and `--dax40_tickers_file` force using local symbol lists.
- If your Python has SSL certificate issues, use local ticker files or run `/Applications/Python 3.13/Install Certificates.command` on macOS.
- `--tickers` overrides `--universe`.

## Outputs

Written to `outputs/`:

- `summary.csv`
- `market_cap_meta.csv` (cached metadata used for market-cap weighting)
- `returns_<strategy>.csv`
- `weights_<strategy>.csv`
- `turnover_<strategy>.csv`
- `component_returns_<strategy>.csv` (ECB variants)
- `oos_diagonality_<strategy>.csv` (ECB variants)
- `component_weight_drift_<strategy>.csv` (ECB variants)
- `equity_curves.png`
- `rolling_vol_sharpe.png`
- `diagonality_<focus_strategy>.png` (ECB focus)
- `component_corr_heatmap_<focus_strategy>.png` (ECB focus)

## Tests

```bash
pytest
```

The test suite checks:
- covariance symmetry + PSD enforcement,
- eigen sorting/reconstruction,
- no-lookahead windowing,
- long-only and fully invested constraints,
- component off-diagonal correlation metric behavior.

## Practical interpretation

- Exact eigen-portfolio orthogonality is straightforward in unconstrained long-short space.
- Under long-only constraints, exact orthogonality is generally impossible.
- This implementation measures how close component return correlations stay to diagonal in-sample and out-of-sample, while keeping final asset weights long-only and fully invested.
- The `market_cap` benchmark uses Yahoo `marketCap` and is useful for baseline comparison, but may not exactly replicate official S&P index methodology.
