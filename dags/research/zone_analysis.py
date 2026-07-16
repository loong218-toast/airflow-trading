from __future__ import annotations

"""
zone_analysis.py

First-pass zone survival research built on top of trade_core.py.

Design goals:
- reuse the existing trade simulator from trade_core.py
- keep buy and sell analyzed separately
- derive only zone survival / length statistics
- write a single CSV: zone_survival_by_age.csv
- avoid plots, dashboards, and extra probe tables for now

Zone definition used here:
- a zone is a consecutive winning streak for one side
- win = trade_result == 1
- loss = trade_result == -1
- neutral = trade_result == 0 (timer / horizon / censored)

A neutral result breaks the zone in this first pass.
That keeps the logic simple and causal.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from .exploration_config import CONFIG, RESULTS_DIR, ExplorationConfig
except ImportError:
    from exploration_config import CONFIG, RESULTS_DIR, ExplorationConfig

try:
    from .trade_core import build_trade_universe, load_cache_df
except ImportError:
    from trade_core import build_trade_universe, load_cache_df

MINUTE_NS = 60 * 1_000_000_000
DEFAULT_OUTPUT_NAME = "zone_survival_by_age.csv"


def safe_name(value: Any) -> str:
    text = str(value).strip()
    out: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "value"


def _config_first_or_value(value: Any, default: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        if len(value) > 0:
            return value[0]
        return default
    return default if value is None else value


def _replace_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def _time_ns_to_myt(time_ns: int, timezone: str = "Asia/Kuala_Lumpur") -> pd.Timestamp:
    return pd.Timestamp(int(time_ns), unit="ns", tz="UTC").tz_convert(timezone)


def _infer_bar_minutes(time_ns: np.ndarray) -> float:
    if time_ns.size < 2:
        return 5.0

    diffs = np.diff(np.asarray(time_ns, dtype=np.int64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 5.0

    median_ns = float(np.median(diffs))
    if not np.isfinite(median_ns) or median_ns <= 0:
        return 5.0

    return median_ns / MINUTE_NS


def _trade_result_from_exit_type(row: pd.Series) -> int:
    exit_type = str(row.get("exit_type", "")).strip().lower()

    if exit_type in {"tp", "target", "win"}:
        return 1
    if exit_type in {"sl", "loss"}:
        return -1
    if exit_type in {"timer", "censored", "horizon", "neutral", ""}:
        return 0

    return 0


def _prepare_trade_df_for_side(
    df_cache: pd.DataFrame,
    config: ExplorationConfig,
    side_mode: str,
) -> pd.DataFrame:
    side_cfg = replace(config, side_mode=str(side_mode))
    trade_df = build_trade_universe(df_cache, side_cfg)

    if trade_df.empty:
        return trade_df

    trade_df = trade_df.copy()
    if "exit_type" not in trade_df.columns:
        trade_df["exit_type"] = ""

    trade_df["trade_result"] = trade_df.apply(_trade_result_from_exit_type, axis=1).astype(np.int8)

    if "time_ns" in trade_df.columns:
        trade_df["time_ns"] = pd.to_numeric(trade_df["time_ns"], errors="coerce").astype("Int64")
    if "exit_time_ns" in trade_df.columns:
        trade_df["exit_time_ns"] = pd.to_numeric(trade_df["exit_time_ns"], errors="coerce").astype("Int64")

    sort_cols = [c for c in ["time_ns", "anchor_idx", "exit_time_ns"] if c in trade_df.columns]
    if sort_cols:
        trade_df = trade_df.sort_values(sort_cols).reset_index(drop=True)

    return trade_df


def _extract_zone_lengths(trade_df: pd.DataFrame) -> list[int]:
    """
    Zone lengths are consecutive win streaks in the side-specific probe history.
    A non-win result (loss or neutral) ends the current zone.
    """
    if trade_df.empty or "trade_result" not in trade_df.columns:
        return []

    results = trade_df["trade_result"].fillna(0).astype(int).to_list()
    lengths: list[int] = []
    current = 0

    for result in results:
        if result == 1:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0

    if current > 0:
        lengths.append(current)

    return lengths


def _compute_survival_rows(
    *,
    side: str,
    zone_lengths: list[int],
    config: ExplorationConfig,
    bar_minutes: float,
) -> pd.DataFrame:
    if not zone_lengths:
        return pd.DataFrame()

    lengths = np.asarray(zone_lengths, dtype=np.int64)
    max_len = int(lengths.max())
    total_zones = int(lengths.size)

    rows: list[dict[str, Any]] = []

    for age_bars in range(1, max_len + 1):
        at_risk = int(np.sum(lengths >= age_bars))
        extended_to_next = int(np.sum(lengths >= (age_bars + 1)))
        ended_at_age = int(np.sum(lengths == age_bars))

        extension_prob = float(extended_to_next / at_risk) if at_risk > 0 else np.nan
        end_prob = float(ended_at_age / at_risk) if at_risk > 0 else np.nan
        survival_prob = float(at_risk / total_zones) if total_zones > 0 else np.nan

        rows.append(
            {
                "instrument": config.instrument,
                "pair": config.pair,
                "mt5_symbol": config.mt5_symbol,
                "side": side,
                "side_label": "buy" if side == "buy" else "sell",
                "simulation_mode": config.simulation_mode,
                "side_mode": config.side_mode,
                "target_pct": float(config.target_pct),
                "sl_pct": float(config.sl_pct),
                "horizon_hours": int(config.horizon_hours),
                "use_trade_timer": bool(_config_first_or_value(config.use_trade_timer, False)),
                "trade_timer_minutes": int(_config_first_or_value(config.trade_timer_minutes, 0)),
                "spread_pct": float(config.spread_pct),
                "risk_pct": float(config.risk_pct),
                "bar_minutes": float(bar_minutes),
                "zone_age_bars": int(age_bars),
                "zone_age_minutes": float(age_bars * bar_minutes),
                "at_risk_count": int(at_risk),
                "extended_to_next_count": int(extended_to_next),
                "ended_at_age_count": int(ended_at_age),
                "extension_prob_next_bar": extension_prob,
                "end_prob_this_bar": end_prob,
                "survival_prob_at_least_age": survival_prob,
                "observed_zone_count": int(total_zones),
                "mean_zone_length_bars": float(lengths.mean()),
                "median_zone_length_bars": float(np.median(lengths)),
                "p75_zone_length_bars": float(np.quantile(lengths, 0.75)),
                "p90_zone_length_bars": float(np.quantile(lengths, 0.90)),
                "max_zone_length_bars": int(max_len),
                "min_zone_length_bars": int(lengths.min()),
            }
        )

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["side", "zone_age_bars"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def build_zone_survival_by_age(
    config: ExplorationConfig = CONFIG,
) -> pd.DataFrame:
    """
    Build a survival table for buy and sell separately.

    First-pass rule:
    - a zone is a consecutive streak of trade_result == 1
    - any non-win result breaks the zone
    """
    df_cache = load_cache_df(config.cache_file)
    if df_cache.empty:
        return pd.DataFrame()

    bar_minutes = _infer_bar_minutes(df_cache["time_ns"].to_numpy(dtype=np.int64, copy=False))

    all_rows: list[pd.DataFrame] = []

    for side_mode in ("buy", "sell"):
        trade_df = _prepare_trade_df_for_side(df_cache=df_cache, config=config, side_mode=side_mode)
        if trade_df.empty:
            continue

        zone_lengths = _extract_zone_lengths(trade_df)
        side_rows = _compute_survival_rows(
            side=side_mode,
            zone_lengths=zone_lengths,
            config=replace(config, side_mode=side_mode),
            bar_minutes=bar_minutes,
        )
        if not side_rows.empty:
            all_rows.append(side_rows)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)

    round_map = {c: 6 for c in out.select_dtypes(include="number").columns}
    round_map["zone_age_bars"] = 0
    round_map["at_risk_count"] = 0
    round_map["extended_to_next_count"] = 0
    round_map["ended_at_age_count"] = 0
    round_map["observed_zone_count"] = 0
    round_map["max_zone_length_bars"] = 0
    round_map["min_zone_length_bars"] = 0

    out = out.round(round_map)
    return out


def _default_output_dir(config: ExplorationConfig) -> Path:
    return RESULTS_DIR / safe_name(config.instrument) / "zone_analysis" / safe_name(config.simulation_mode)


def save_zone_survival_by_age_csv(
    summary_df: pd.DataFrame,
    output_file: Path | None = None,
    config: ExplorationConfig = CONFIG,
) -> Path:
    if output_file is None:
        output_dir = _default_output_dir(config)
        output_file = output_dir / DEFAULT_OUTPUT_NAME

    output_file = Path(output_file)
    _replace_file(output_file)
    summary_df.to_csv(output_file, index=False, float_format="%.8f")
    return output_file


def run_zone_analysis(config: ExplorationConfig = CONFIG) -> dict[str, str]:
    summary_df = build_zone_survival_by_age(config=config)
    if summary_df.empty:
        return {}

    output_file = save_zone_survival_by_age_csv(summary_df, config=config)

    return {
        "summary_file": str(output_file),
        "instrument": str(config.instrument),
        "pair": str(config.pair),
        "mt5_symbol": str(config.mt5_symbol),
        "simulation_mode": str(config.simulation_mode),
    }


if __name__ == "__main__":
    result = run_zone_analysis()
    if result:
        print(result["summary_file"])
