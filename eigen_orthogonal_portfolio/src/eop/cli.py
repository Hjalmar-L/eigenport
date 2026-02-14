from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .data import (
    fetch_adj_close,
    fetch_dax40_tickers,
    fetch_omxs30_tickers,
    fetch_sp500_tickers,
    load_tickers_from_csv,
    load_or_fetch_market_cap_metadata,
    parse_tickers,
)
from .metrics import summarize_backtest
from .plots import (
    save_component_corr_heatmaps,
    save_diagonality_series,
    save_equity_curves,
    save_rolling_vol_sharpe,
)
from .preprocess import compute_returns, preprocess_prices


ALL_STRATEGIES = ["ew", "mv_lo", "ecb", "ecb_equala", "market_cap"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eigen orthogonal portfolio backtest")
    parser.add_argument(
        "--universe",
        type=str,
        choices=["default", "sp500", "omxs30", "dax40"],
        default="default",
    )
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    parser.add_argument(
        "--sp500_tickers_file",
        type=str,
        default=None,
        help="Local CSV for S&P500 symbols (uses Symbol column or first column)",
    )
    parser.add_argument(
        "--sp500_top_n",
        type=int,
        default=50,
        help="When universe=sp500, keep top N by market cap (set <=0 to keep all).",
    )
    parser.add_argument(
        "--omxs30_tickers_file",
        type=str,
        default=None,
        help="Local CSV for OMXS30 symbols (uses Symbol column or first column)",
    )
    parser.add_argument(
        "--dax40_tickers_file",
        type=str,
        default=None,
        help="Local CSV for DAX40 symbols (uses Symbol column or first column)",
    )
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--rebalance", type=str, choices=["monthly"], default="monthly")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["ew", "mv_lo", "ecb", "ecb_equala", "market_cap", "float_cap"],
        default="ecb",
    )
    parser.add_argument("--cov", type=str, choices=["sample", "ledoitwolf", "shrink"], default="sample")
    parser.add_argument("--tcost_bps", type=float, default=5.0)
    parser.add_argument("--winsorize", type=str, choices=["none", "p01_p99"], default="none")
    parser.add_argument("--max_missing", type=float, default=0.05)
    parser.add_argument("--fill_limit", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1e-3)
    parser.add_argument("--shrink_delta", type=float, default=0.1)
    parser.add_argument("--refresh_sp500_tickers", action="store_true")
    parser.add_argument("--refresh_omxs30_tickers", action="store_true")
    parser.add_argument("--refresh_dax40_tickers", action="store_true")
    parser.add_argument("--refresh_market_cap_meta", action="store_true")
    parser.add_argument("--refresh_float_meta", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--sp500_benchmark_ticker",
        type=str,
        default="XLG",
        help="ETF ticker used as benchmark when universe=sp500 (default: XLG).",
    )
    parser.add_argument(
        "--dax40_benchmark_ticker",
        type=str,
        default="EXS1.DE",
        help="ETF ticker used as benchmark when universe=dax40 (default: EXS1.DE).",
    )
    parser.add_argument(
        "--omxs30_benchmark_ticker",
        type=str,
        default="XACT-OMXS30.ST",
        help="ETF ticker used as benchmark when universe=omxs30 (default: XACT-OMXS30.ST).",
    )
    return parser


def _summary_table(
    results: dict[str, object],
    benchmark_returns: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, bt in results.items():
        row = summarize_backtest(bt.portfolio_returns, bt.turnover)
        row["strategy"] = name
        if bt.component_returns is not None:
            row["mean_insample_offdiag"] = float(bt.insample_offdiag_long_only.mean())
            row["mean_oos_offdiag"] = float(bt.oos_offdiag.mean())
            row["mean_component_weight_drift"] = float(bt.component_weight_drift.mean())
        else:
            row["mean_insample_offdiag"] = float("nan")
            row["mean_oos_offdiag"] = float("nan")
            row["mean_component_weight_drift"] = float("nan")
        rows.append(row)

    if benchmark_returns:
        for name, rets in benchmark_returns.items():
            row = summarize_backtest(rets, turnover=None)
            row["strategy"] = name
            row["mean_insample_offdiag"] = float("nan")
            row["mean_oos_offdiag"] = float("nan")
            row["mean_component_weight_drift"] = float("nan")
            rows.append(row)

    df = pd.DataFrame(rows)
    cols = [
        "strategy",
        "total_return",
        "cagr",
        "vol",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "annualized_turnover",
        "mean_insample_offdiag",
        "mean_oos_offdiag",
        "mean_component_weight_drift",
    ]
    return df[cols].sort_values("strategy").reset_index(drop=True)


def run_cli(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tickers:
        tickers = parse_tickers(args.tickers)
    elif args.universe == "sp500":
        min_sp500_count = args.sp500_top_n if args.sp500_top_n > 0 else 300
        if args.sp500_tickers_file:
            tickers = load_tickers_from_csv(args.sp500_tickers_file)
        else:
            tickers = fetch_sp500_tickers(
                cache_path=output_dir / "sp500_tickers.csv",
                refresh=args.refresh_sp500_tickers,
                min_count=min_sp500_count,
            )
    elif args.universe == "omxs30":
        if args.omxs30_tickers_file:
            tickers = load_tickers_from_csv(args.omxs30_tickers_file)
        else:
            tickers = fetch_omxs30_tickers(
                cache_path=output_dir / "omxs30_tickers.csv",
                refresh=args.refresh_omxs30_tickers,
            )
    elif args.universe == "dax40":
        if args.dax40_tickers_file:
            tickers = load_tickers_from_csv(args.dax40_tickers_file)
        else:
            tickers = fetch_dax40_tickers(
                cache_path=output_dir / "dax40_tickers.csv",
                refresh=args.refresh_dax40_tickers,
            )
    else:
        tickers = parse_tickers(None)

    if args.universe == "sp500" and args.sp500_top_n > 0 and len(tickers) > args.sp500_top_n:
        sp500_meta = load_or_fetch_market_cap_metadata(
            tickers,
            cache_path=output_dir / "sp500_universe_market_cap_meta.csv",
            refresh=args.refresh_market_cap_meta or args.refresh_float_meta,
        )
        ranked = sp500_meta["market_cap"].fillna(0.0).sort_values(ascending=False)
        tickers = ranked.head(args.sp500_top_n).index.tolist()
        print(f"Selected top {len(tickers)} S&P 500 names by market cap.")

    print(f"Using {len(tickers)} input tickers from universe='{args.universe}'.")
    prices = fetch_adj_close(tickers=tickers, start=args.start, end=args.end)
    clean_prices = preprocess_prices(
        prices,
        max_missing_frac=args.max_missing,
        fill_limit=args.fill_limit,
    )
    print(f"Using {clean_prices.shape[1]} tickers after preprocessing and missing-data filters.")
    returns = compute_returns(clean_prices, method="log", winsorize=args.winsorize)

    strategies_to_run = ALL_STRATEGIES.copy()
    benchmark_returns: dict[str, pd.Series] = {}

    if args.universe in {"sp500", "dax40", "omxs30"}:
        # For S&P 500 / DAX40 / OMXS30 tests, benchmark against live ETF(s)
        # instead of synthetic market-cap weights.
        strategies_to_run = [s for s in strategies_to_run if s != "market_cap"]
        if args.universe == "sp500":
            bench_ticker = args.sp500_benchmark_ticker.strip().upper()
        elif args.universe == "dax40":
            bench_ticker = args.dax40_benchmark_ticker.strip().upper()
        else:
            bench_ticker = args.omxs30_benchmark_ticker.strip().upper()
        bench_prices = fetch_adj_close(
            tickers=[bench_ticker],
            start=args.start,
            end=args.end,
        )
        bench_returns = compute_returns(
            bench_prices,
            method="log",
            winsorize=args.winsorize,
        ).iloc[:, 0]
        benchmark_returns[bench_ticker] = bench_returns
    else:
        market_cap_meta = load_or_fetch_market_cap_metadata(
            list(clean_prices.columns),
            cache_path=output_dir / "market_cap_meta.csv",
            refresh=args.refresh_market_cap_meta or args.refresh_float_meta,
        )

    results = {}
    for strategy in strategies_to_run:
        kwargs = {
            "returns": returns,
            "prices": clean_prices,
            "strategy": strategy,
            "window": args.window,
            "rebalance": args.rebalance,
            "k": args.k,
            "cov_estimator": args.cov,
            "tcost_bps": args.tcost_bps,
            "gamma": args.gamma,
            "shrink_delta": args.shrink_delta,
        }
        if strategy == "market_cap":
            kwargs["market_cap_meta"] = market_cap_meta
        results[strategy] = run_backtest(**kwargs)

    for name, bt in results.items():
        bt.weights.to_csv(output_dir / f"weights_{name}.csv")
        bt.portfolio_returns.to_frame("return").to_csv(output_dir / f"returns_{name}.csv")
        bt.turnover.to_frame("turnover").to_csv(output_dir / f"turnover_{name}.csv")
        if bt.component_returns is not None:
            bt.component_returns.to_csv(output_dir / f"component_returns_{name}.csv")
            bt.oos_offdiag.to_frame("mean_abs_offdiag_corr").to_csv(
                output_dir / f"oos_diagonality_{name}.csv"
            )
            bt.component_weight_drift.to_frame("component_weight_drift").to_csv(
                output_dir / f"component_weight_drift_{name}.csv"
            )

    if benchmark_returns and results:
        ref_index = next(iter(results.values())).portfolio_returns.index
        for key, rets in list(benchmark_returns.items()):
            benchmark_returns[key] = rets.reindex(ref_index).dropna()
            benchmark_returns[key].to_frame("return").to_csv(output_dir / f"returns_{key}.csv")

    summary = _summary_table(results, benchmark_returns=benchmark_returns)
    summary.to_csv(output_dir / "summary.csv", index=False)

    returns_map = {name: bt.portfolio_returns for name, bt in results.items()}
    returns_map.update(benchmark_returns)
    save_equity_curves(returns_map, output_dir / "equity_curves.png")
    save_rolling_vol_sharpe(returns_map, output_dir / "rolling_vol_sharpe.png")

    focus_key = "market_cap" if args.strategy == "float_cap" else args.strategy
    if focus_key in results:
        focus = results[focus_key]
        if focus.component_returns is not None:
            save_diagonality_series(focus.oos_offdiag, output_dir / f"diagonality_{focus_key}.png")
            save_component_corr_heatmaps(
                focus.insample_component_corr,
                focus.oos_component_corr,
                output_dir / f"component_corr_heatmap_{focus_key}.png",
            )

    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary.round(4).to_string(index=False))
    print(f"\nOutputs written to: {output_dir.resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
