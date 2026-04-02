from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl

from etl.schema import get_schema


# -----------------------------
# generic coercion helpers
# -----------------------------
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

# -----------------------------
# time / index helpers
# -----------------------------
def normalize_signals_times(df_signals: pl.DataFrame, df_main: Optional[pl.DataFrame] = None) -> pl.DataFrame:
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

    if df_main is not None and df_main.height > 0:
        main_times = df_main["time_ns"]
        sig_times = df["time_ns"]
        idxs = np.searchsorted(main_times, sig_times, side="left")
        df = df.with_columns(
            pl.Series("idx", idxs).clip(0, df_main.height - 1).cast(pl.Int64)
        )

    return df


# -----------------------------
# MA helpers
# -----------------------------
def _resolve_ma_cols(cfg: dict, df: pl.DataFrame) -> list[str]:
    ma_cols = cfg.get("ma_cols", None)
    if isinstance(ma_cols, list) and ma_cols:
        return [c for c in ma_cols if c in df.columns]

    # fallback for older caches
    return [
        c for c in df.columns
        if c.startswith("sma_") or c.startswith("ema_") or c.startswith("kama_")
    ]


def _resolve_ma_col_name(df: pl.DataFrame, idx: int) -> Optional[str]:
    candidates = [
        f"ma_{chr(97 + idx)}",
        f"kama_{chr(97 + idx)}",
        f"ema_{chr(97 + idx)}",
        f"sma_{chr(97 + idx)}",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_ma_price_gaps_for_indices(
    df_main: pl.DataFrame,
    entry_idxs: np.ndarray,
    exit_idxs: np.ndarray,
    regime_cfg: Optional[dict] = None,
    run_cfg: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    regime_cfg = regime_cfg or {}
    run_cfg = run_cfg or {}

    n_entry = int(entry_idxs.shape[0]) if entry_idxs is not None else 0
    n_exit = int(exit_idxs.shape[0]) if exit_idxs is not None else 0

    gap_a_entry = np.zeros(n_entry, dtype=np.float32)
    gap_b_entry = np.zeros(n_entry, dtype=np.float32)
    gap_a_exit = np.zeros(n_exit, dtype=np.float32)
    gap_b_exit = np.zeros(n_exit, dtype=np.float32)

    if df_main is None or df_main.height == 0:
        return gap_a_entry, gap_b_entry, gap_a_exit, gap_b_exit

    cfg = {**run_cfg, **regime_cfg}
    ma_cols = _resolve_ma_cols(cfg, df_main)

    ma_int = _as_int(cfg.get("ma_int", 0), 0)

    active_ma_cols: list[str] = []
    if ma_cols:
        for i, col in enumerate(ma_cols):
            if (ma_int >> i) & 1 and col in df_main.columns:
                active_ma_cols.append(col)

    if len(active_ma_cols) < 2:
        candidates = [
            c for c in df_main.columns
            if c.startswith("ma_") or c.startswith("kama_") or c.startswith("ema_") or c.startswith("sma_")
        ]
        candidates = sorted(candidates)
        if len(candidates) >= 2:
            active_ma_cols = candidates[:2]

    if len(active_ma_cols) < 2:
        return gap_a_entry, gap_b_entry, gap_a_exit, gap_b_exit

    fast_col = active_ma_cols[0]
    slow_col = active_ma_cols[1]

    fast_arr = (
        df_main[fast_col]
        .fill_nan(0.0)
        .fill_null(0.0)
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    slow_arr = (
        df_main[slow_col]
        .fill_nan(0.0)
        .fill_null(0.0)
        .to_numpy()
        .astype(np.float64, copy=False)
    )

    def _fill_gaps(idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idxs = np.asarray(idxs, dtype=np.int64)
        if idxs.size == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        valid = (idxs >= 0) & (idxs < fast_arr.shape[0])
        if not valid.any():
            return np.zeros(idxs.size, dtype=np.float32), np.zeros(idxs.size, dtype=np.float32)

        out_a = np.zeros(idxs.size, dtype=np.float32)
        out_b = np.zeros(idxs.size, dtype=np.float32)

        pos = idxs[valid]
        fast_v = fast_arr[pos]
        slow_v = slow_arr[pos]

        fast_v = np.nan_to_num(fast_v, nan=0.0, posinf=0.0, neginf=0.0)
        slow_v = np.nan_to_num(slow_v, nan=0.0, posinf=0.0, neginf=0.0)

        spread = fast_v - slow_v
        spread_pct = np.where(slow_v != 0.0, spread / slow_v, 0.0)

        out_a[valid] = spread.astype(np.float32)
        out_b[valid] = spread_pct.astype(np.float32)
        return out_a, out_b

    gap_a_entry, gap_b_entry = _fill_gaps(entry_idxs)
    gap_a_exit, gap_b_exit = _fill_gaps(exit_idxs)
    return gap_a_entry, gap_b_entry, gap_a_exit, gap_b_exit


# -----------------------------
# regime amplitude extraction
# -----------------------------
def get_regime_amp_index_for_indices(
    df_main: pl.DataFrame,
    entry_idxs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract regime amplitude values at trade entry positions.
    Returns:
        regime_amp_24h_entry, regime_amp_72h_entry, regime_amp_1w_entry, regime_amp_1m_entry
    """
    if entry_idxs is None:
        entry_idxs = np.asarray([], dtype=np.int64)
    else:
        entry_idxs = np.asarray(entry_idxs, dtype=np.int64)

    n = entry_idxs.shape[0]
    zero = np.zeros(n, dtype=np.float32)

    if df_main is None or df_main.height == 0 or n == 0:
        return zero, zero.copy(), zero.copy(), zero.copy()

    cols = [
        ("regime_amp_index_24h", "rng_24h"),
        ("regime_amp_index_72h", "rng_72h"),
        ("regime_amp_index_1w", "rng_1w"),
        ("regime_amp_index_1m", "rng_1m"),
    ]

    out = []
    for new_col, legacy_col in cols:
        col_name = new_col if new_col in df_main.columns else legacy_col if legacy_col in df_main.columns else None
        if col_name is None:
            out.append(zero.copy())
            continue

        arr = (
            df_main[col_name]
            .fill_nan(0.0)
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float32, copy=False)
        )

        idxs = np.clip(entry_idxs, 0, max(0, arr.shape[0] - 1))
        vals = arr[idxs]
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        out.append(vals)

    return tuple(out)

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

    k = _as_int(cfg.get("stoch_k", 12), 12)
    d = _as_int(cfg.get("stoch_d", 3), 3)
    s = _as_int(cfg.get("stoch_s", 3), 3)
    return f"stoch_k{k}_d{d}_s{s}"


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
def generate_filtered_signals(df_slice: pl.DataFrame, cfg: dict, df_main: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    if df_slice is None or not isinstance(df_slice, pl.DataFrame) or df_slice.height == 0:
        return pl.DataFrame([], schema=get_schema("signals"))

    df_slice = normalize_signals_times(df_slice, df_main=df_main)
    if "idx" in df_slice.columns:
        df_slice = df_slice.sort(["idx", "time_ns"])
    else:
        df_slice = df_slice.sort(["time_ns"])

    n = df_slice.height
    final_buy = np.ones(n, dtype=bool)
    final_sell = np.ones(n, dtype=bool)

    # --- 1. MA filter ---
    ma_int = _as_int(cfg.get("ma_int", 0), 0)
    ma_cols = _resolve_ma_cols(cfg, df_slice)

    if ma_int > 0 and ma_cols:
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
            if (ma_int >> i) & 1:
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

    # --- 2. Stochastic stateful gate ---
    if _as_bool(cfg.get("use_stochastic", False), False):
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

    # --- 3. Lookback filter ---
    lb_units = _as_int(cfg.get("entry_lookback_units", 0), 0)
    if lb_units > 0:
        brk_col = f"breakout_{lb_units}u"
        if brk_col in df_slice.columns:
            breakout_arr = (
                df_slice[brk_col]
                .fill_nan(0.0)
                .fill_null(0.0)
                .to_numpy()
                .astype(np.float64, copy=False)
            )
            final_buy &= breakout_arr >= 0.9
            final_sell &= breakout_arr <= 0.1

    # --- 4. BBW / regime amplitude filter ---
    if _as_bool(cfg.get("use_bbw", False), False):
        p = _as_int(cfg.get("bbw_periods", 0), 0)
        s = _as_float(cfg.get("bbw_std", 0), 0.0)
        threshold = _as_float(cfg.get("bbw_thresholds", 50), 50.0)
        bbw_col = f"bbw_p{p}_s{s}"

        if bbw_col in df_slice.columns:
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
    return out


# -----------------------------
# small util: compute selected ma_price_gap series name
# -----------------------------
def selected_gap_col_for_ma_int(ma_int: int) -> str:
    """
    Pick the first set bit (LSB = first MA column).
    If none set, fall back to 'ma_price_gap_a'.
    """
    try:
        mi = int(ma_int or 0)
    except Exception:
        mi = 0

    for i in range(4):
        if (mi >> i) & 1:
            return f"ma_price_gap_{chr(97 + i)}"
    return "ma_price_gap_a"


__all__ = [
    "normalize_signals_times",
    "get_ma_price_gaps_for_indices",
    "get_regime_amp_index_for_indices",
    "generate_filtered_signals",
    "selected_gap_col_for_ma_int"
]