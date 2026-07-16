# expectancy_mae_mfe.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from research.expectancy_config import OUTPUT_DIR, _market_tag, _safe_name

logger = logging.getLogger(__name__)

PLOT_BINS = 8
ENTRY_RESET_TOL_PCT = 0.0
PLOT_MAX_R = 2.0
PLOT_BIN_EDGES = np.linspace(0.0, PLOT_MAX_R, PLOT_BINS + 1)


def _import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        sns = None

    return plt, sns


def _safe_div(n: float, d: float) -> Optional[float]:
    if d == 0:
        return None
    return float(n / d)


def _replace_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def _save_figure(fig, out: Path) -> Path:
    _replace_file(out)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    return out


def make_mae_mfe_stats() -> Dict[str, Any]:
    return {
        "entry_revisit_mae_sum": 0.0,
        "entry_revisit_mae_count": 0,
        "entry_revisit_mfe_sum": 0.0,
        "entry_revisit_mfe_count": 0,
        "entry_revisit_mae_win_sum": 0.0,
        "entry_revisit_mae_win_count": 0,
        "entry_revisit_mae_loss_sum": 0.0,
        "entry_revisit_mae_loss_count": 0,
        "entry_revisit_mfe_win_sum": 0.0,
        "entry_revisit_mfe_win_count": 0,
        "entry_revisit_mfe_loss_sum": 0.0,
        "entry_revisit_mfe_loss_count": 0,
        "entry_revisit_mae_hist": np.zeros(PLOT_BINS, dtype=np.int64),
        "entry_revisit_mfe_hist": np.zeros(PLOT_BINS, dtype=np.int64),
        "entry_revisit_mae_win_hist": np.zeros(PLOT_BINS, dtype=np.int64),
        "entry_revisit_mae_loss_hist": np.zeros(PLOT_BINS, dtype=np.int64),
        "entry_revisit_mfe_win_hist": np.zeros(PLOT_BINS, dtype=np.int64),
        "entry_revisit_mfe_loss_hist": np.zeros(PLOT_BINS, dtype=np.int64),
    }


def _path_pre_resolution_excursion_stats(
    side: int,
    entry_price: float,
    highs: np.ndarray,
    lows: np.ndarray,
    tp_ratio: float,
    sl_ratio: float,
    clip_to_one_r: bool = True,
) -> Dict[str, float]:
    """Compute MAE/MFE from entry until the already-sliced path ends.

    The caller is expected to pass only the bars up to the first resolution event
    (TP, SL, or censoring horizon). This fixes the old entry-revisit gating,
    which could miss Case B and could also count post-resolution movement.
    """
    nan_out = {
        "entry_revisit_mae_r": np.nan,
        "entry_revisit_mfe_r": np.nan,
        "had_reset": False,
    }

    if not np.isfinite(entry_price) or entry_price <= 0.0:
        return nan_out
    if tp_ratio <= 0.0 or sl_ratio <= 0.0:
        return nan_out
    if highs.size == 0 or lows.size == 0:
        return nan_out

    n = int(min(len(highs), len(lows)))
    if n <= 0:
        return nan_out

    highs = np.asarray(highs[:n], dtype=np.float64)
    lows = np.asarray(lows[:n], dtype=np.float64)

    if side == 1:
        mae_raw = np.maximum(0.0, entry_price - lows)
        mfe_raw = np.maximum(0.0, highs - entry_price)
    else:
        mae_raw = np.maximum(0.0, highs - entry_price)
        mfe_raw = np.maximum(0.0, entry_price - lows)

    if mae_raw.size == 0 or mfe_raw.size == 0:
        return nan_out

    mae_max_raw = float(np.nanmax(mae_raw))
    mfe_max_raw = float(np.nanmax(mfe_raw))

    mae_r = (mae_max_raw / entry_price) / sl_ratio if sl_ratio > 0.0 else np.nan
    mfe_r = (mfe_max_raw / entry_price) / tp_ratio if tp_ratio > 0.0 else np.nan

    if np.isfinite(mae_r) and clip_to_one_r:
        mae_r = min(mae_r, 1.0)
    if np.isfinite(mfe_r) and clip_to_one_r:
        mfe_r = min(mfe_r, 1.0)

    return {
        "entry_revisit_mae_r": float(mae_r) if np.isfinite(mae_r) else np.nan,
        "entry_revisit_mfe_r": float(mfe_r) if np.isfinite(mfe_r) else np.nan,
        "had_reset": True,
    }


# Backward-compatible alias so expectancy_analysis.py can keep importing the old name
# while you migrate to the pre-resolution version.
def _path_entry_revisit_excursion_stats(
    side: int,
    entry_price: float,
    highs: np.ndarray,
    lows: np.ndarray,
    tp_ratio: float,
    sl_ratio: float,
    reset_tol_pct: float = ENTRY_RESET_TOL_PCT,
    exclude_last_bar: bool = True,
) -> Dict[str, float]:
    del reset_tol_pct, exclude_last_bar
    return _path_pre_resolution_excursion_stats(
        side=side,
        entry_price=entry_price,
        highs=highs,
        lows=lows,
        tp_ratio=tp_ratio,
        sl_ratio=sl_ratio,
    )


def _accumulate_histograms(summary_df: pd.DataFrame, hist_col: str) -> np.ndarray:
    total_hist = np.zeros(PLOT_BINS, dtype=np.float64)
    if hist_col not in summary_df.columns:
        return total_hist

    for v in summary_df[hist_col].dropna():
        if isinstance(v, str):
            try:
                arr = np.fromstring(v.strip("[]"), sep=",")
                if arr.size == PLOT_BINS:
                    total_hist += arr
                continue
            except Exception:
                continue

        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == PLOT_BINS:
            total_hist += np.asarray(v, dtype=np.float64)

    return total_hist


def save_mae_mfe_distribution_plots(
    summary_df: pd.DataFrame,
    instrument: str,
    pair: str,
    symbol: str,
) -> List[Path]:
    plt, _ = _import_plotting()
    if summary_df.empty:
        return []

    market = _market_tag(instrument, pair, symbol)
    saved: List[Path] = []

    specs = [
        ("entry_revisit_mae_hist", "Entry-revisit MAE (R)", f"mae_distribution_{market}.png"),
        ("entry_revisit_mfe_hist", "Entry-revisit MFE (R)", f"mfe_distribution_{market}.png"),
        ("entry_revisit_mae_win_hist", "Entry-revisit MAE win-only (R)", f"mae_win_distribution_{market}.png"),
        ("entry_revisit_mae_loss_hist", "Entry-revisit MAE loss-only (R)", f"mae_loss_distribution_{market}.png"),
        ("entry_revisit_mfe_win_hist", "Entry-revisit MFE win-only (R)", f"mfe_win_distribution_{market}.png"),
        ("entry_revisit_mfe_loss_hist", "Entry-revisit MFE loss-only (R)", f"mfe_loss_distribution_{market}.png"),
    ]

    centers = (PLOT_BIN_EDGES[:-1] + PLOT_BIN_EDGES[1:]) / 2.0
    width = (PLOT_BIN_EDGES[1] - PLOT_BIN_EDGES[0]) * 0.9

    for hist_col, y_label, fname in specs:
        total_hist = _accumulate_histograms(summary_df, hist_col)
        if total_hist.sum() <= 0:
            continue

        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        ax.bar(centers, total_hist, width=width)
        ax.set_title(f"{market} | {y_label}")
        ax.set_xlabel(y_label)
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)

        out = OUTPUT_DIR / fname
        _save_figure(fig, out)
        plt.close(fig)
        saved.append(out)

    return saved
