# bootstrap_core.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def safe_name(value: str) -> str:
    return str(value).strip().replace(" ", "_").replace("/", "_").replace(".", "_").replace(":", "_")


def replace_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def moving_block_bootstrap_sample(
    values: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = int(values.shape[0])
    if n <= 0:
        return np.asarray([], dtype=np.float64)

    block_size = max(1, min(int(block_size), n))
    if n == 1:
        return values.copy()

    starts = np.arange(0, n - block_size + 1, dtype=np.int64)
    if starts.size == 0:
        return values.copy()

    out: list[float] = []
    while len(out) < n:
        start = int(rng.choice(starts))
        block = values[start : start + block_size]
        out.extend(block.tolist())

    return np.asarray(out[:n], dtype=np.float64)


def bootstrap_mean_samples(
    values: np.ndarray,
    n_bootstrap: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([], dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(n_bootstrap), dtype=np.float64)

    for i in range(int(n_bootstrap)):
        sampled = moving_block_bootstrap_sample(values, block_size=block_size, rng=rng)
        samples[i] = float(np.mean(sampled)) if sampled.size else np.nan

    return samples[np.isfinite(samples)]


def equity_curve_from_returns(
    returns: np.ndarray,
    start_balance: float = 100.0,
) -> np.ndarray:
    rets = np.asarray(returns, dtype=np.float64)
    rets = rets[np.isfinite(rets)]
    if rets.size == 0:
        return np.asarray([], dtype=np.float64)

    rets = np.clip(rets, -0.999999, None)
    equity = float(start_balance) * np.cumprod(1.0 + rets)
    return equity.astype(np.float64, copy=False)


def max_drawdown_pct_from_equity(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=np.float64)
    equity = equity[np.isfinite(equity)]
    if equity.size == 0:
        return np.nan

    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(np.min(dd) * 100.0)


# research/bootstrap_core.py additions

def get_max_streaks(returns: np.ndarray) -> tuple[int, int]:
    """Returns (max_consecutive_wins, max_consecutive_losses)"""
    if returns.size == 0:
        return 0, 0
    
    wins = returns > 0
    losses = returns <= 0
    
    def max_run(arr):
        if not arr.any(): return 0
        # Find runs of True
        bounded = np.hstack(([False], arr, [False]))
        diffs = np.diff(bounded.view(np.int8))
        run_starts = np.where(diffs == 1)[0]
        run_ends = np.where(diffs == -1)[0]
        return int(np.max(run_ends - run_starts))

    return max_run(wins), max_run(losses)

# UPDATE the bootstrap_equity_drawdown_samples function in research/bootstrap_core.py:
def bootstrap_equity_drawdown_samples(
    returns: np.ndarray,
    n_bootstrap: int,
    block_size: int,
    seed: int,
    start_balance: float = 100.0,
) -> Dict[str, np.ndarray]:
    rets = np.asarray(returns, dtype=np.float64)
    rets = rets[np.isfinite(rets)]
    if rets.size == 0:
        empty = np.asarray([], dtype=np.float64)
        return {
            "terminal_balance": empty,
            "terminal_return_pct": empty,
            "max_drawdown_pct": empty,
            "max_con_wins": empty,
            "max_con_losses": empty,
        }

    rng = np.random.default_rng(int(seed))
    terminal_balance = np.empty(int(n_bootstrap), dtype=np.float64)
    terminal_return_pct = np.empty(int(n_bootstrap), dtype=np.float64)
    max_drawdown_pct = np.empty(int(n_bootstrap), dtype=np.float64)
    max_con_wins = np.empty(int(n_bootstrap), dtype=np.float64)
    max_con_losses = np.empty(int(n_bootstrap), dtype=np.float64)

    for i in range(int(n_bootstrap)):
        sampled = moving_block_bootstrap_sample(rets, block_size=block_size, rng=rng)
        
        # Calculate streaks on the sampled sequence
        win_streak, loss_streak = get_max_streaks(sampled)
        max_con_wins[i] = win_streak
        max_con_losses[i] = loss_streak

        sampled_clipped = np.clip(sampled, -0.999999, None)
        equity = float(start_balance) * np.cumprod(1.0 + sampled_clipped)
        
        if equity.size == 0:
            terminal_balance[i] = np.nan
            terminal_return_pct[i] = np.nan
            max_drawdown_pct[i] = np.nan
            continue

        terminal_balance[i] = float(equity[-1])
        terminal_return_pct[i] = float((equity[-1] / float(start_balance) - 1.0) * 100.0)
        max_drawdown_pct[i] = max_drawdown_pct_from_equity(equity)

    return {
        "terminal_balance": terminal_balance[np.isfinite(terminal_balance)],
        "terminal_return_pct": terminal_return_pct[np.isfinite(terminal_return_pct)],
        "max_drawdown_pct": max_drawdown_pct[np.isfinite(max_drawdown_pct)],
        "max_con_wins": max_con_wins[np.isfinite(max_con_wins)],
        "max_con_losses": max_con_losses[np.isfinite(max_con_losses)],
    }


def describe_numeric_series(values: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p05": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def _save_histogram(
    values: np.ndarray,
    out_png: Path,
    title: str,
    xlabel: str,
    observed: Optional[float] = None,
    extra_vlines: Optional[list[float]] = None,
    bins: int = 40,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return out_png

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_min == x_max:
        pad = max(abs(x_min) * 0.05, 1e-9)
    else:
        pad = max((x_max - x_min) * 0.05, 1e-9)
    x_min -= pad
    x_max += pad

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    ax.hist(x, bins=np.linspace(x_min, x_max, int(bins) + 1))
    if observed is not None and np.isfinite(observed):
        ax.axvline(float(observed), linestyle="-")
    if extra_vlines:
        for v in extra_vlines:
            if v is not None and np.isfinite(v):
                ax.axvline(float(v), linestyle=":")

    if x_min <= 0.0 <= x_max:
        ax.axvline(0.0, linestyle="--")

    ax.set_xlim(x_min, x_max)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)

    replace_file(out_png)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


def save_histogram(
    values: np.ndarray,
    out_png: Path,
    title: str,
    xlabel: str,
    observed: Optional[float] = None,
    extra_vlines: Optional[list[float]] = None,
    bins: int = 40,
) -> Path:
    return _save_histogram(
        values=values,
        out_png=out_png,
        title=title,
        xlabel=xlabel,
        observed=observed,
        extra_vlines=extra_vlines,
        bins=bins,
    )