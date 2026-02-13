from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_equity_curves(returns_map: dict[str, pd.Series], out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for name, rets in returns_map.items():
        wealth = (1.0 + rets).cumprod()
        plt.plot(wealth.index, wealth.values, label=name)
    plt.title("Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def save_rolling_vol_sharpe(returns_map: dict[str, pd.Series], out_path: Path, window: int = 63) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name, rets in returns_map.items():
        roll_mean = rets.rolling(window).mean() * 252
        roll_vol = rets.rolling(window).std(ddof=1) * np.sqrt(252)
        roll_sharpe = roll_mean / roll_vol.replace(0, np.nan)

        axes[0].plot(roll_vol.index, roll_vol.values, label=name)
        axes[1].plot(roll_sharpe.index, roll_sharpe.values, label=name)

    axes[0].set_title(f"Rolling {window}D Volatility")
    axes[0].set_ylabel("Annualized Vol")
    axes[0].grid(alpha=0.3)

    axes[1].set_title(f"Rolling {window}D Sharpe")
    axes[1].set_ylabel("Sharpe")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)

    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def save_diagonality_series(diagonality: pd.Series, out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    if not diagonality.empty:
        plt.plot(diagonality.index, diagonality.values)
    plt.title("ECB Out-of-Sample Mean |Off-Diagonal Corr|")
    plt.xlabel("Rebalance Date")
    plt.ylabel("Mean |Off-Diagonal Corr|")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_heatmap(ax, matrix: pd.DataFrame, title: str) -> None:
    if matrix is None or matrix.empty:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return

    im = ax.imshow(matrix.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_component_corr_heatmaps(
    insample_corr: pd.DataFrame | None,
    oos_corr: pd.DataFrame | None,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _plot_heatmap(axes[0], insample_corr, "In-Sample Component Corr")
    _plot_heatmap(axes[1], oos_corr, "Out-of-Sample Component Corr")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
