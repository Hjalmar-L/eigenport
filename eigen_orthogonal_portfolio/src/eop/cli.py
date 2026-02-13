from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .data import parse_tickers, fetch_adj_close
from .metrics import summarize_backtest
from .plots import (
    save_component_corr_heatmaps,
    save_diagonality_series,
    save_equity_curves,
    save_rolling_vol_sharpe,
)
from .preprocess import compute_returns, preprocess_prices


ALL_STRATEGIES = ["ew", "mv_lo", "ecb", "ecb_equala"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eigen orthogonal portfolio backtest")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--rebalance", type=str, choices=["monthly"], default="monthly")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["ew", "mv_lo", "ecb", "ecb_equala"],
        default="ecb",
    )
    parser.add_argument("--cov", type=str, choices=["sample", "ledoitwolf", "shrink"], default="sample")
    parser.add_argument("--tcost_bps", type=float, default=5.0)
    parser.add_argument("--winsorize", type=str, choices=["none", "p01_p99"], default="none")
    parser.add_argument("--max_missing", type=float, default=0.05)
    parser.add_argument("--fill_limit", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1e-3)
    parser.add_argument("--shrink_delta", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser


def _summary_table(results: dict[str, object]) -> pd.DataFrame:
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

    tickers = parse_tickers(args.tickers)
    prices = fetch_adj_close(tickers=tickers, start=args.start, end=args.end)
    clean_prices = preprocess_prices(
        prices,
        max_missing_frac=args.max_missing,
        fill_limit=args.fill_limit,
    )
    returns = compute_returns(clean_prices, method="log", winsorize=args.winsorize)

    results = {}
    for strategy in ALL_STRATEGIES:
        results[strategy] = run_backtest(
            returns=returns,
            strategy=strategy,
            window=args.window,
            rebalance=args.rebalance,
            k=args.k,
            cov_estimator=args.cov,
            tcost_bps=args.tcost_bps,
            gamma=args.gamma,
            shrink_delta=args.shrink_delta,
        )

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

    summary = _summary_table(results)
    summary.to_csv(output_dir / "summary.csv", index=False)

    returns_map = {name: bt.portfolio_returns for name, bt in results.items()}
    save_equity_curves(returns_map, output_dir / "equity_curves.png")
    save_rolling_vol_sharpe(returns_map, output_dir / "rolling_vol_sharpe.png")

    focus = results[args.strategy]
    if focus.component_returns is not None:
        save_diagonality_series(focus.oos_offdiag, output_dir / f"diagonality_{args.strategy}.png")
        save_component_corr_heatmaps(
            focus.insample_component_corr,
            focus.oos_component_corr,
            output_dir / f"component_corr_heatmap_{args.strategy}.png",
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
