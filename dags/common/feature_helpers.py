# research/feature_helpers.py
# Keep the above .py filename for reference, but this is not a standalone script. It's meant to be imported and used by other code in the research pipeline for backtesting trading signals.

from __future__ import annotations

from typing import Any, Optional, Tuple
import re

import numpy as np
import polars as pl

from common.schema import get_schema


def _unwrap_singleton(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_bool(value: Any, default: bool = False) -> bool:
    value = _unwrap_singleton(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _as_int(value: Any, default: int = 0) -> int:
    value = _unwrap_singleton(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        return int(float(s))
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    value = _unwrap_singleton(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        return float(s)
    return default


def _as_str(value: Any, default: str = "") -> str:
    value = _unwrap_singleton(value)
    if value is None:
        return default
    return str(value)


def _as_int_list(value: Any) -> list[int]:
    return [_as_int(x) for x in _as_list(value)]


def _as_float_list(value: Any) -> list[float]:
    return [_as_float(x) for x in _as_list(value)]


def _as_threshold_pairs(value: Any) -> list[list[float]]:
    value = _unwrap_singleton(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return [[_as_float(a), _as_float(b)] for a, b in value]
    if isinstance(value, (list, tuple)):
        vals = [_as_float(x) for x in value]
        return [vals] if vals else []
    return [[_as_float(value)]]


def _as_ma_type_list(value: Any, count: int) -> list[str]:
    if count <= 0:
        return []

    if value is None:
        return ["sma"] * count

    if isinstance(value, str):
        v = value.strip().lower() or "sma"
        return [v] * count

    items = _as_list(value)
    if not items:
        return ["sma"] * count

    if len(items) == 1:
        v = _as_str(items[0], "sma").strip().lower() or "sma"
        return [v] * count

    out = [_as_str(x, "sma").strip().lower() or "sma" for x in items]
    if len(out) < count:
        out.extend(["sma"] * (count - len(out)))
    return out[:count]


def _float_token(value: Any) -> str:
    s = f"{float(value):.10g}"
    if "e" in s or "E" in s:
        return s.replace("+", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _ma_sort_key(col: str) -> tuple:
    m = re.match(r"^ma_(\d+)_", col)
    if m:
        return (0, int(m.group(1)), col)

    m = re.search(r"(\d+)$", col)
    if m:
        return (1, int(m.group(1)), col)

    return (2, 10**9, col)


def _resolve_ma_cols(cfg: dict, df: pl.DataFrame) -> list[str]:
    ma_cols = cfg.get("ma_cols", None)
    if isinstance(ma_cols, list) and ma_cols:
        out = [c for c in ma_cols if c in df.columns]
        if out:
            return out

    manifest = cfg.get("feature_manifest", None)
    if isinstance(manifest, dict):
        mf_cols = manifest.get("ma_cols", None)
        if isinstance(mf_cols, list) and mf_cols:
            out = [c for c in mf_cols if c in df.columns]
            if out:
                return out

    return sorted(
        [c for c in df.columns if c.startswith(("ma_", "sma_", "ema_", "kama_"))],
        key=_ma_sort_key,
    )


def _selected_ma_pair(cfg: dict, df: pl.DataFrame) -> list[str]:
    ma_cols = _resolve_ma_cols(cfg, df)
    if len(ma_cols) < 2:
        return []

    ma_int = _as_int(cfg.get("ma_int", 0), 0)
    active = [c for i, c in enumerate(ma_cols) if (ma_int >> i) & 1]

    if len(active) >= 2:
        return active[:2]

    return ma_cols[:2]


# -----------------------------
# time / index helpers
# -----------------------------
def normalize_signals_times(df_signals: pl.DataFrame, df_context: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    if df_signals is None or df_signals.height == 0:
        return df_signals

    df = df_signals.with_columns(
        pl.col("time")
        .cast(pl.Datetime("ns"))
        .dt.replace_time_zone(None)
    )

    df = df.with_columns(
        pl.col("time").cast(pl.Int64).alias("time_ns")
    )

    if df_context is not None and df_context.height > 0:
        main_times = df_context["time_ns"]
        sig_times = df["time_ns"]
        idxs = np.searchsorted(main_times, sig_times, side="left")
        df = df.with_columns(
            pl.Series("idx", idxs).clip(0, df_context.height - 1).cast(pl.Int64)
        )

    return df


# -----------------------------
# MA helpers
# -----------------------------
def _resolve_ma_col_name(df: pl.DataFrame, idx: int) -> Optional[str]:
    candidates = [
        f"ma_{idx:02d}",
        f"ma_{chr(97 + idx)}",
        f"kama_{chr(97 + idx)}",
        f"ema_{chr(97 + idx)}",
        f"sma_{chr(97 + idx)}",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -----------------------------
# stochastic helpers
# -----------------------------
def _resolve_stoch_bounds(cfg: dict) -> Tuple[float, float]:
    lower = cfg.get("stoch_lower", None)
    upper = cfg.get("stoch_upper", None)

    if lower is not None or upper is not None:
        return _as_float(lower, 30.0), _as_float(upper, 70.0)

    ths = cfg.get("stoch_thresholds", None)
    pairs = _as_threshold_pairs(ths)
    if pairs and len(pairs[0]) >= 2:
        return float(pairs[0][0]), float(pairs[0][1])

    return 30.0, 70.0


def _resolve_stoch_col(cfg: dict) -> Optional[str]:
    stoch_col = _as_str(cfg.get("stoch_col", ""), "").strip()
    if stoch_col:
        return stoch_col

    stoch_cols = cfg.get("stoch_cols", None)
    if isinstance(stoch_cols, list):
        stoch_cols = [str(x).strip() for x in stoch_cols if str(x).strip()]
        if len(stoch_cols) == 1:
            return stoch_cols[0]

    manifest = cfg.get("feature_manifest", None)
    if isinstance(manifest, dict):
        mf_cols = manifest.get("stoch_cols", None)
        if isinstance(mf_cols, list):
            mf_cols = [str(x).strip() for x in mf_cols if str(x).strip()]
            if len(mf_cols) == 1:
                return mf_cols[0]

    k = _as_int(cfg.get("stoch_k", 12), 12)
    d = _as_int(cfg.get("stoch_d", 3), 3)
    s = _as_int(cfg.get("stoch_s", 3), 3)
    return f"stoch_k{k}_d{d}_s{s}"


def _resolve_bbw_col(cfg: dict) -> Optional[str]:
    bbw_col = _as_str(cfg.get("bbw_col", ""), "").strip()
    if bbw_col:
        return bbw_col

    bbw_cols = cfg.get("bbw_cols", None)
    if isinstance(bbw_cols, list):
        bbw_cols = [str(x).strip() for x in bbw_cols if str(x).strip()]
        if len(bbw_cols) == 1:
            return bbw_cols[0]

    manifest = cfg.get("feature_manifest", None)
    if isinstance(manifest, dict):
        mf_cols = manifest.get("bbw_cols", None)
        if isinstance(mf_cols, list):
            mf_cols = [str(x).strip() for x in mf_cols if str(x).strip()]
            if len(mf_cols) == 1:
                return mf_cols[0]

    p = cfg.get("bbw_periods", None)
    s = cfg.get("bbw_std", None)

    if isinstance(p, list):
        p_val = _as_int(p[0], 0) if p else 0
    else:
        p_val = _as_int(p, 0)

    if isinstance(s, list):
        s_val = _as_float(s[0], 0.0) if s else 0.0
    else:
        s_val = _as_float(s, 0.0)

    if p_val <= 0:
        return None

    return f"bbw_p{p_val}_s{_float_token(s_val)}"


def _stoch_state_masks(
    stoch_values: np.ndarray,
    lower: float,
    upper: float,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(stoch_values, dtype=np.float64)
    n = vals.shape[0]
    buy_mask = np.zeros(n, dtype=bool)
    sell_mask = np.zeros(n, dtype=bool)

    if n == 0:
        return buy_mask, sell_mask

    tolerance = float(max(0.0, tolerance))

    buy_reset_level = min(float(upper), float(lower) + tolerance)
    sell_reset_level = max(float(lower), float(upper) - tolerance)

    buy_active = False
    sell_active = False

    for i in range(n):
        v = vals[i]
        if np.isnan(v):
            continue

        if buy_active:
            if v > buy_reset_level:
                buy_active = False
            else:
                buy_mask[i] = True
        else:
            if v <= lower:
                buy_active = True
                buy_mask[i] = True

        if sell_active:
            if v < sell_reset_level:
                sell_active = False
            else:
                sell_mask[i] = True
        else:
            if v >= upper:
                sell_active = True
                sell_mask[i] = True

    return buy_mask, sell_mask


# -----------------------------
# signal generator
# -----------------------------
def generate_filtered_signals(
    df_slice: pl.DataFrame,
    cfg: dict,
    df_context: Optional[pl.DataFrame] = None,
    return_stats: bool = False,
):
    if df_slice is None or not isinstance(df_slice, pl.DataFrame) or df_slice.height == 0:
        empty = pl.DataFrame([], schema=get_schema("signals"))
        if return_stats:
            return empty, {
                "input_rows": 0,
                "final_buy_signals": 0,
                "final_sell_signals": 0,
                "final_total_signals": 0,
                "buy_filtered_out": 0,
                "sell_filtered_out": 0,
            }
        return empty

    df_slice = normalize_signals_times(df_slice, df_context=df_context)
    if "idx" in df_slice.columns:
        df_slice = df_slice.sort(["idx", "time_ns"])
    else:
        df_slice = df_slice.sort(["time_ns"])

    n = df_slice.height
    final_buy = np.ones(n, dtype=bool)
    final_sell = np.ones(n, dtype=bool)

    stats = {
        "input_rows": int(n),
        "ma_enabled": False,
        "stochastic_enabled": False,
        "lookback_enabled": False,
        "bbw_enabled": False,
        "after_ma_buy": int(n),
        "after_ma_sell": int(n),
        "after_stochastic_buy": int(n),
        "after_stochastic_sell": int(n),
        "after_lookback_buy": int(n),
        "after_lookback_sell": int(n),
        "after_bbw_buy": int(n),
        "after_bbw_sell": int(n),
        "final_buy_signals": 0,
        "final_sell_signals": 0,
        "final_total_signals": 0,
        "buy_filtered_out": 0,
        "sell_filtered_out": 0,
    }

    # --- 1. MA filter ---
    ma_int = _as_int(cfg.get("ma_int", 0), 0)
    ma_cols = _resolve_ma_cols(cfg, df_slice)

    if ma_int > 0 and ma_cols:
        stats["ma_enabled"] = True
        ma_reversion = _as_bool(cfg.get("ma_reversion", False), False)

        close_arr = (
            df_slice["close"]
            .fill_nan(0.0)
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float64, copy=False)
        )

        ma_buy = np.ones(n, dtype=bool)
        ma_sell = np.ones(n, dtype=bool)

        for i, ma_col in enumerate(ma_cols):
            if (ma_int >> i) & 1 and ma_col in df_slice.columns:
                ma_arr = (
                    df_slice[ma_col]
                    .fill_nan(0.0)
                    .fill_null(0.0)
                    .to_numpy()
                    .astype(np.float64, copy=False)
                )

                if not ma_reversion:
                    ma_buy &= close_arr > ma_arr
                    ma_sell &= close_arr < ma_arr
                else:
                    ma_buy &= close_arr < ma_arr
                    ma_sell &= close_arr > ma_arr

        final_buy &= ma_buy
        final_sell &= ma_sell
        stats["after_ma_buy"] = int(ma_buy.sum())
        stats["after_ma_sell"] = int(ma_sell.sum())

    # --- 2. Stochastic gate ---
    if _as_bool(cfg.get("use_stochastic", False), False):
        stats["stochastic_enabled"] = True
        stoch_col = _resolve_stoch_col(cfg)
        lower, upper = _resolve_stoch_bounds(cfg)
        tolerance = _as_float(cfg.get("stoch_threshold_tolerance", 10.0), 10.0)

        if stoch_col and stoch_col in df_slice.columns:
            stoch_arr = (
                df_slice[stoch_col]
                .fill_nan(np.nan)
                .fill_null(np.nan)
                .to_numpy()
                .astype(np.float64, copy=False)
            )
            stoch_buy, stoch_sell = _stoch_state_masks(stoch_arr, lower, upper, tolerance)
            final_buy &= stoch_buy
            final_sell &= stoch_sell
            stats["after_stochastic_buy"] = int(stoch_buy.sum())
            stats["after_stochastic_sell"] = int(stoch_sell.sum())

    # --- 3. Lookback filter ---
        # --- 3. Lookback filter ---
    lb_units_list = _as_int_list(cfg.get("entry_lookback_units", []))
    modifier = max(1, _as_int(cfg.get("signal_timeframe_modifier", 3), 3))

    if lb_units_list:
        stats["lookback_enabled"] = True
        any_breakout_buy = np.zeros(n, dtype=bool)
        any_breakout_sell = np.zeros(n, dtype=bool)
        found_any = False

        close_arr = (
            df_slice["close"]
            .fill_nan(np.nan)
            .fill_null(np.nan)
            .to_numpy()
            .astype(np.float64, copy=False)
        )

        for lb in lb_units_list:
            if lb <= 0:
                continue

            periods = max(1, int(lb) * modifier)

            prior_high = (
                df_slice["high"]
                .rolling_max(periods)
                .shift(1)
                .fill_nan(np.nan)
                .fill_null(np.nan)
                .to_numpy()
                .astype(np.float64, copy=False)
            )

            prior_low = (
                df_slice["low"]
                .rolling_min(periods)
                .shift(1)
                .fill_nan(np.nan)
                .fill_null(np.nan)
                .to_numpy()
                .astype(np.float64, copy=False)
            )

            valid = ~np.isnan(prior_high) & ~np.isnan(prior_low) & ~np.isnan(close_arr)
            found_any = True

            # strict trend-following breakout logic
            any_breakout_buy |= valid & (close_arr >= prior_high)
            any_breakout_sell |= valid & (close_arr <= prior_low)

        if found_any:
            final_buy &= any_breakout_buy
            final_sell &= any_breakout_sell
            stats["after_lookback_buy"] = int(any_breakout_buy.sum())
            stats["after_lookback_sell"] = int(any_breakout_sell.sum())

    # --- 4. BBW filter ---
    if _as_bool(cfg.get("use_bbw", False), False):
        stats["bbw_enabled"] = True
        bbw_col = _resolve_bbw_col(cfg)
        threshold = _as_float(cfg.get("bbw_thresholds", 50), 50.0)

        if bbw_col and bbw_col in df_slice.columns:
            bbw_arr = (
                df_slice[bbw_col]
                .fill_nan(0.0)
                .fill_null(0.0)
                .to_numpy()
                .astype(np.float64, copy=False)
            )
            vol_gate = bbw_arr <= threshold
            final_buy &= vol_gate
            final_sell &= vol_gate
            stats["after_bbw_buy"] = int(vol_gate.sum())
            stats["after_bbw_sell"] = int(vol_gate.sum())

    regime_id = _as_int(cfg.get("regime_id", 0), 0)

    buy_df = df_slice.filter(pl.Series(final_buy)).select([
        pl.col("idx"),
        pl.col("time_ns"),
        pl.lit(1, dtype=pl.Int8).alias("side"),
        pl.lit(regime_id, dtype=pl.Int32).alias("regime_id"),
    ])

    sell_df = df_slice.filter(pl.Series(final_sell)).select([
        pl.col("idx"),
        pl.col("time_ns"),
        pl.lit(-1, dtype=pl.Int8).alias("side"),
        pl.lit(regime_id, dtype=pl.Int32).alias("regime_id"),
    ])

    out = pl.concat([buy_df, sell_df], how="vertical").sort(["idx", "side"])

    stats["final_buy_signals"] = int(final_buy.sum())
    stats["final_sell_signals"] = int(final_sell.sum())
    stats["final_total_signals"] = int(final_buy.sum() + final_sell.sum())
    stats["buy_filtered_out"] = int(n - final_buy.sum())
    stats["sell_filtered_out"] = int(n - final_sell.sum())

    if return_stats:
        return out, stats
    return out


__all__ = [
    "normalize_signals_times",
    "generate_filtered_signals",
]