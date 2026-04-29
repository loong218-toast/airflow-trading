from __future__ import annotations

import json
from typing import Any, Optional

import polars as pl


BASELINE_SCOPES = ("baseline:all_buy", "baseline:all_sell")
MASTER_START_BALANCE = 100.0

# Tune this at the top.
ERA_MIN_RETURN = 0.05


def first(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return first(v[0]) if v else None
    return v


def num(x: Any) -> str:
    if x is None:
        return "na"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return str(int(x)) if x.is_integer() else f"{x:g}"
    return str(x)


def signal_obj(x: Any) -> dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def signal_blocks(signal_json: Any) -> dict[str, Any]:
    return signal_obj(signal_json).get("signals", {}) or {}


def stoch_key(signal_json: Any, tf: str = "15m") -> str:
    signals = signal_blocks(signal_json)
    st = signals.get("stochastic", {}).get(tf, {}) if isinstance(signals, dict) else {}
    k = first(st.get("k"))
    d = first(st.get("d"))
    s = first(st.get("s"))
    th = st.get("thresholds", [[30, 70]])
    low, high = 30, 70
    if isinstance(th, (list, tuple)) and th:
        first_item = th[0]
        if isinstance(first_item, (list, tuple)) and len(first_item) >= 2:
            low, high = first_item[0], first_item[1]
        elif len(th) >= 2 and not any(isinstance(i, (list, tuple, dict)) for i in th[:2]):
            low, high = th[0], th[1]
    tol = first(st.get("threshold_tolerance", 10))
    return f"k{num(k)}_d{num(d)}_s{num(s)}_l{num(low)}_u{num(high)}_tol{num(tol)}"


def lookback_key(signal_json: Any, tf: str = "15m") -> str:
    signals = signal_blocks(signal_json)
    lb = first(signals.get("lookback", {}).get(tf, {}).get("entry_lookback_units"))
    return f"u{num(lb)}"


def add_human_readable_signal_keys(df: pl.DataFrame, timeframes: tuple[str, ...] = ("5m", "15m")) -> pl.DataFrame:
    if df is None or df.is_empty() or "signal_json" not in df.columns:
        return df

    out = df
    for tf in timeframes:
        out = out.with_columns(
            pl.col("signal_json")
            .map_elements(lambda x, _tf=tf: stoch_key(x, _tf), return_dtype=pl.Utf8)
            .alias(f"stoch_key_{tf}"),
            pl.col("signal_json")
            .map_elements(lambda x, _tf=tf: lookback_key(x, _tf), return_dtype=pl.Utf8)
            .alias(f"lookback_key_{tf}"),
        )
    return out


def get_risk_pct(run_cfg: dict[str, Any]) -> float:
    risk_pct = float(run_cfg.get("risk_pct", 0.0) or 0.0)
    if risk_pct <= 0.0:
        raise ValueError("run_config.json must contain a positive risk_pct.")
    return risk_pct


def _round_float_cols(df: pl.DataFrame, decimals: int = 5) -> pl.DataFrame:
    if df is None or df.is_empty():
        return df

    out = df
    for name, dtype in out.schema.items():
        if dtype in (pl.Float32, pl.Float64):
            out = out.with_columns(pl.col(name).cast(pl.Float64, strict=False).round(decimals).alias(name))
    return out


def prepare_master(
    df_master: pl.DataFrame,
    risk_pct: float,
    trade_overlap_filter: Optional[bool] = None,
    trade_flip_on_entry_filter: Optional[bool] = None,
) -> pl.DataFrame:
    if df_master is None or df_master.is_empty():
        return pl.DataFrame()

    out = df_master

    if "signal_scope_id" not in out.columns:
        raise ValueError("master parquet must contain signal_scope_id")
    if "signal_layer" not in out.columns:
        raise ValueError("master parquet must contain signal_layer")

    if "trade_overlap" not in out.columns:
        out = out.with_columns(pl.lit(None).alias("trade_overlap"))
    if "trade_flip_on_entry" not in out.columns:
        out = out.with_columns(pl.lit(None).alias("trade_flip_on_entry"))

    out = out.with_columns(
        pl.col("signal_scope_id").cast(pl.Utf8, strict=False).fill_null("").alias("signal_scope_id"),
        pl.col("signal_layer").cast(pl.Int64, strict=False).fill_null(-1).alias("signal_layer"),
        pl.col("trade_overlap").cast(pl.Boolean, strict=False).alias("trade_overlap"),
        pl.col("trade_flip_on_entry").cast(pl.Boolean, strict=False).alias("trade_flip_on_entry"),
    )

    out = out.with_columns(
        (
            (pl.col("balance").cast(pl.Float64, strict=False).fill_null(MASTER_START_BALANCE) / MASTER_START_BALANCE) - 1.0
        ).alias("session_return_pct")
    )

    out = out.with_columns(
        pl.when(pl.col("total_pos") > 0)
        .then(pl.col("session_return_pct") / pl.col("total_pos"))
        .otherwise(None)
        .alias("mean_return_pct_per_trade")
    )

    out = out.with_columns(
        (pl.col("mean_return_pct_per_trade").cast(pl.Float64, strict=False) / float(risk_pct)).alias("mean_expectancy_r_per_trade")
    )

    out = out.with_columns(
        (pl.col("mean_return_pct_per_trade").fill_null(0.0) * pl.col("total_pos").cast(pl.Float64, strict=False).fill_null(0.0)).alias("net_expectancy_pct"),
        (pl.col("mean_expectancy_r_per_trade").fill_null(0.0) * pl.col("total_pos").cast(pl.Float64, strict=False).fill_null(0.0)).alias("net_expectancy_r"),
    )

    out = out.with_columns(
        (
            MASTER_START_BALANCE
            * (1.0 + pl.col("mean_return_pct_per_trade").fill_null(0.0))
            ** pl.col("total_pos").cast(pl.Float64, strict=False).fill_null(0.0)
        ).alias("expected_balance_from_mean")
    )

    out = out.with_columns(
        (pl.col("balance") - pl.col("expected_balance_from_mean")).alias("balance_vs_mean_expected_diff")
    )

    if trade_overlap_filter is not None:
        out = out.filter(pl.col("trade_overlap") == bool(trade_overlap_filter))
    if trade_flip_on_entry_filter is not None:
        out = out.filter(pl.col("trade_flip_on_entry") == bool(trade_flip_on_entry_filter))

    out = add_human_readable_signal_keys(out)
    return _round_float_cols(out, 5)


def trade_mgmt_key_cols(df: pl.DataFrame) -> list[str]:
    cols = [
        "exit_window_h",
        "SL",
        "TP",
        "use_trailing_sl",
        "trailing_sl_pct",
        "trailing_sl_interval",
        "trailing_sl_stop_at_pos",
        "use_limit_entry",
        "limit_order_expiry_bars",
        "trade_overlap",
        "trade_flip_on_entry",
        "trade_window_interval",
    ]
    return [c for c in cols if c in df.columns]


def _uniq(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


def aggregate_master_combos(
    df_master: pl.DataFrame,
    risk_pct: float,
    era_min_return: float = ERA_MIN_RETURN,
) -> pl.DataFrame:
    """
    Aggregate master rows to one row per combo across all eras.
    Era consistency is computed from the per-era rows first, then rolled up.
    """
    if df_master is None or df_master.is_empty():
        return pl.DataFrame()

    if "era_int" not in df_master.columns:
        raise ValueError("master parquet must contain era_int")

    key_cols = trade_mgmt_key_cols(df_master)
    group_cols = _uniq(key_cols + ["signal_scope_id", "signal_layer"])
    group_cols = [c for c in group_cols if c in df_master.columns]

    per_era = (
        df_master
        .group_by(group_cols + ["era_int"])
        .agg(
            pl.col("regime_id").first().alias("regime_id"),

            pl.col("balance").mean().alias("era_balance"),
            pl.col("max_drawdown").mean().alias("era_max_drawdown"),
            pl.col("max_consecutive_losses").mean().alias("era_max_consecutive_losses"),
            pl.col("total_pos").sum().alias("era_total_pos"),
            pl.col("win_pos").sum().alias("era_win_pos"),
            pl.col("session_return_pct").mean().alias("era_session_return_pct"),
            pl.col("mean_return_pct_per_trade").mean().alias("era_mean_return_pct_per_trade"),
            pl.col("mean_expectancy_r_per_trade").mean().alias("era_mean_expectancy_r_per_trade"),
        )
        .with_columns(
            (pl.col("era_session_return_pct") >= float(era_min_return)).alias("era_hit"),
            pl.col("era_total_pos").cast(pl.Float64, strict=False).fill_null(0.0).alias("_era_weight"),
        )
    )

    rolled = (
        per_era
        .group_by(group_cols)
        .agg(
            pl.len().alias("master_rows"),
            pl.col("regime_id").n_unique().alias("regime_count"),
            pl.col("regime_id").first().alias("regime_id_first"),
            pl.col("era_int").n_unique().alias("era_count"),
            pl.col("era_hit").mean().alias("era_consistency_score"),
            pl.col("era_hit").sum().alias("era_hit_count"),
            pl.col("era_balance").sum().alias("balance_sum"),
            pl.col("era_balance").mean().alias("mean_balance"),
            pl.col("era_balance").median().alias("median_balance"),
            pl.col("era_balance").min().alias("min_balance"),
            pl.col("era_balance").max().alias("max_balance"),
            pl.col("era_max_drawdown").mean().alias("mean_max_drawdown"),
            pl.col("era_max_drawdown").max().alias("worst_max_drawdown"),
            pl.col("era_max_consecutive_losses").mean().alias("mean_max_consecutive_losses"),
            pl.col("era_max_consecutive_losses").max().alias("worst_max_consecutive_losses"),
            pl.col("era_session_return_pct").mean().alias("mean_session_return_pct"),
            pl.col("era_session_return_pct").median().alias("median_session_return_pct"),
            pl.col("era_mean_return_pct_per_trade").mean().alias("mean_return_pct_per_trade"),
            pl.col("era_mean_return_pct_per_trade").median().alias("median_return_pct_per_trade"),
            pl.col("era_mean_expectancy_r_per_trade").mean().alias("mean_expectancy_r_per_trade"),
            pl.col("era_mean_expectancy_r_per_trade").median().alias("median_expectancy_r_per_trade"),
            pl.col("era_total_pos").sum().alias("total_pos_sum"),
            pl.col("era_win_pos").sum().alias("win_pos_sum"),
            pl.col("era_total_pos").sum().cast(pl.Float64, strict=False).alias("_total_pos_sum_f"),
        )
        .with_columns(
            (pl.col("win_pos_sum") / (pl.col("total_pos_sum") + 1e-9)).alias("win_rate"),
            (pl.col("mean_return_pct_per_trade").fill_null(0.0) * pl.col("total_pos_sum").cast(pl.Float64, strict=False).fill_null(0.0)).alias("net_expectancy_pct"),
            (pl.col("mean_expectancy_r_per_trade").fill_null(0.0) * pl.col("total_pos_sum").cast(pl.Float64, strict=False).fill_null(0.0)).alias("net_expectancy_r"),
            (
                MASTER_START_BALANCE
                * (1.0 + pl.col("mean_return_pct_per_trade").fill_null(0.0))
                ** pl.col("total_pos_sum").cast(pl.Float64, strict=False).fill_null(0.0)
            ).alias("expected_balance_from_mean"),
        )
        .with_columns(
            (pl.col("mean_balance") - pl.col("expected_balance_from_mean")).alias("mean_balance_vs_expected_diff")
        )
        .drop("_total_pos_sum_f")
    )

    return _round_float_cols(rolled, 5)


def build_baseline_vs_signal_comparisons(df_combo_agg: pl.DataFrame) -> pl.DataFrame:
    """
    Compare combo aggregates against baseline controls.
    This is combo-level only: era rows are already rolled up.
    """
    if df_combo_agg is None or df_combo_agg.is_empty():
        return pl.DataFrame()

    base = df_combo_agg.filter(pl.col("signal_scope_id").is_in(list(BASELINE_SCOPES)))
    sig = df_combo_agg.filter(~pl.col("signal_scope_id").is_in(list(BASELINE_SCOPES)))

    if base.is_empty() or sig.is_empty():
        return pl.DataFrame()

    join_keys = trade_mgmt_key_cols(df_combo_agg)
    if not join_keys:
        return pl.DataFrame()

    metric_cols = [
        "signal_scope_id",
        "signal_layer",
        "regime_count",
        "regime_id_first",
        "era_count",
        "era_consistency_score",
        "era_hit_count",
        "master_rows",
        "total_pos_sum",
        "win_pos_sum",
        "mean_balance",
        "median_balance",
        "min_balance",
        "max_balance",
        "mean_max_drawdown",
        "worst_max_drawdown",
        "mean_max_consecutive_losses",
        "worst_max_consecutive_losses",
        "mean_session_return_pct",
        "median_session_return_pct",
        "mean_return_pct_per_trade",
        "median_return_pct_per_trade",
        "mean_expectancy_r_per_trade",
        "median_expectancy_r_per_trade",
        "net_expectancy_pct",
        "net_expectancy_r",
        "expected_balance_from_mean",
        "mean_balance_vs_expected_diff",
        "win_rate",
    ]

    base_cols = _uniq(join_keys + [c for c in metric_cols if c in base.columns])
    sig_cols = _uniq(join_keys + [c for c in metric_cols if c in sig.columns])

    base = base.select(base_cols)
    sig = sig.select(sig_cols)

    parts = []
    for baseline_scope in BASELINE_SCOPES:
        base_sub = base.filter(pl.col("signal_scope_id") == baseline_scope)
        if base_sub.is_empty():
            continue

        joined = sig.join(base_sub, on=join_keys, how="inner", suffix="_baseline")
        if joined.is_empty():
            continue

        joined = joined.with_columns(
            pl.lit(baseline_scope).alias("baseline_scope_id"),
            pl.col("signal_scope_id").alias("signal_scope_id_signal"),
            pl.col("signal_layer").alias("signal_layer_signal"),
            pl.col("signal_scope_id_baseline").alias("signal_scope_id_baseline"),
            pl.col("signal_layer_baseline").alias("signal_layer_baseline"),
            (pl.col("mean_balance") - pl.col("mean_balance_baseline")).alias("mean_balance_diff"),
            (pl.col("median_balance") - pl.col("median_balance_baseline")).alias("median_balance_diff"),
            (pl.col("mean_balance_vs_expected_diff") - pl.col("mean_balance_vs_expected_diff_baseline")).alias("expected_gap_diff"),
            (pl.col("mean_expectancy_r_per_trade") - pl.col("mean_expectancy_r_per_trade_baseline")).alias("mean_expectancy_r_diff"),
            (pl.col("mean_return_pct_per_trade") - pl.col("mean_return_pct_per_trade_baseline")).alias("mean_return_diff_pct"),
            (pl.col("net_expectancy_r") - pl.col("net_expectancy_r_baseline")).alias("net_expectancy_r_diff"),
            (pl.col("net_expectancy_pct") - pl.col("net_expectancy_pct_baseline")).alias("net_expectancy_pct_diff"),
            (pl.col("era_consistency_score") - pl.col("era_consistency_score_baseline")).alias("era_consistency_diff"),
            (pl.col("era_hit_count") - pl.col("era_hit_count_baseline")).alias("era_hit_count_diff"),
            (pl.col("master_rows") - pl.col("master_rows_baseline")).alias("row_count_diff"),
            (pl.col("win_rate") - pl.col("win_rate_baseline")).alias("win_rate_diff"),
            (pl.col("mean_max_drawdown") - pl.col("mean_max_drawdown_baseline")).alias("mean_drawdown_diff"),
            (pl.col("worst_max_drawdown") - pl.col("worst_max_drawdown_baseline")).alias("worst_drawdown_diff"),
            (pl.col("mean_max_consecutive_losses") - pl.col("mean_max_consecutive_losses_baseline")).alias("mean_streak_diff"),
            (pl.col("worst_max_consecutive_losses") - pl.col("worst_max_consecutive_losses_baseline")).alias("worst_streak_diff"),
        )

        parts.append(joined)

    if not parts:
        return pl.DataFrame()

    out = pl.concat(parts, how="vertical_relaxed")
    sort_cols = [c for c in ["baseline_scope_id", "signal_scope_id_signal", "mean_balance_diff"] if c in out.columns]
    if sort_cols:
        out = out.sort(sort_cols, descending=[False, False, True][: len(sort_cols)])

    return _round_float_cols(out, 5)


def summarize_comparisons(df_cmp: pl.DataFrame) -> pl.DataFrame:
    if df_cmp is None or df_cmp.is_empty():
        return pl.DataFrame()

    key_cols = [c for c in ["baseline_scope_id", "signal_scope_id_signal"] if c in df_cmp.columns]
    if not key_cols:
        return pl.DataFrame()

    out = (
        df_cmp
        .group_by(key_cols)
        .agg(
            pl.len().alias("pairs"),
            pl.col("mean_balance_diff").mean().alias("avg_mean_balance_diff"),
            pl.col("median_balance_diff").mean().alias("avg_median_balance_diff"),
            pl.col("mean_expectancy_r_diff").mean().alias("avg_expectancy_r_diff"),
            pl.col("mean_return_diff_pct").mean().alias("avg_return_diff_pct"),
            pl.col("net_expectancy_r_diff").mean().alias("avg_net_expectancy_r_diff"),
            pl.col("net_expectancy_pct_diff").mean().alias("avg_net_expectancy_pct_diff"),
            pl.col("era_consistency_diff").mean().alias("avg_era_consistency_diff"),
            pl.col("era_hit_count_diff").mean().alias("avg_era_hit_count_diff"),
            pl.col("row_count_diff").mean().alias("avg_row_count_diff"),
            pl.col("win_rate_diff").mean().alias("avg_win_rate_diff"),
            pl.col("mean_drawdown_diff").mean().alias("avg_mean_drawdown_diff"),
            pl.col("worst_drawdown_diff").mean().alias("avg_worst_drawdown_diff"),
            pl.col("mean_streak_diff").mean().alias("avg_mean_streak_diff"),
            pl.col("worst_streak_diff").mean().alias("avg_worst_streak_diff"),
            (pl.col("mean_balance_diff") > 0).mean().alias("signal_beats_baseline_rate"),
        )
        .sort(["baseline_scope_id", "avg_mean_balance_diff"], descending=[False, True] if "baseline_scope_id" in key_cols else [True])
    )

    return _round_float_cols(out, 5)