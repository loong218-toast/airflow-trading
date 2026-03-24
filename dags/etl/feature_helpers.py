# etl/feature_helpers.py
from typing import List, Optional, Dict, Tuple, Any
import polars as pl
import numpy as np
import gc

from etl.schema import get_schema, enforce_schema

_STOCH_CACHE: Dict = {}
_LOOKBACK_CACHE: Dict = {}


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
    if isinstance(value, (int, float)):
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


def _as_bool_list(value: Any) -> list[bool]:
    return [_as_bool(x) for x in _as_list(value)]


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

# ---------- time/idx helpers ----------
def normalize_signals_times(df_signals: pl.DataFrame, df_main: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    if df_signals is None or df_signals.height == 0:
        return df_signals

    # 1) Force 'time' to Datetime (Naive) and standardize to Nanoseconds
    df = df_signals.with_columns(
        pl.col("time")
        .cast(pl.Datetime("ns"))  # Force ns here
        .dt.replace_time_zone(None)
    )

    # 2) Create 'time_ns' (Int64)
    df = df.with_columns(
        pl.col("time").cast(pl.Int64).alias("time_ns")
    )

    # 3) Alignment
    if df_main is not None:
        # CRITICAL: Ensure df_main is also ns before grabbing the array
        main_times = df_main["time_ns"].to_numpy(allow_copy=False)
        sig_times = df["time_ns"].to_numpy(allow_copy=False)

        # Use 'left' side search for entry signals (find the exact bar start)
        idxs = np.searchsorted(main_times, sig_times, side="left")

        df = df.with_columns(
            pl.Series("idx", idxs).clip(0, df_main.height - 1).cast(pl.Int64)
        )

    return df

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

    ma_periods = regime_cfg.get("ma_periods", run_cfg.get("ma_periods", []))
    ma_periods = _as_int_list(ma_periods)

    ma_int = _as_int(regime_cfg.get("ma_int", 0), 0)

    active_ma_cols = []
    for i in range(len(ma_periods)):
        if (ma_int >> i) & 1:
            col = f"kama_{chr(97 + i)}"
            if col in df_main.columns:
                active_ma_cols.append(col)

    if len(active_ma_cols) < 2:
        kama_cols = sorted([c for c in df_main.columns if c.startswith("kama_")])
        if len(kama_cols) >= 2:
            active_ma_cols = kama_cols[:2]

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

def add_volatility_index_features(df: pl.DataFrame, run_cfg: dict) -> pl.DataFrame:
    if df is None or df.height == 0:
        return df

    base_min = _as_int(run_cfg.get("BASE_MINUTES", 5), 5)

    use_signal_tf = _as_bool(run_cfg.get("volatility_use_signal_timeframe", False), False)
    modifier = _as_int(run_cfg.get("signal_timeframe_modifier", 3), 3) if use_signal_tf else 1
    effective_min = max(1, base_min * modifier)

    windows = {
        "rng_24h": int(round((24 * 60) / effective_min)),
        "rng_72h": int(round((72 * 60) / effective_min)),
        "rng_1w": int(round((7 * 24 * 60) / effective_min)),
        "rng_1m": int(round((30 * 24 * 60) / effective_min)),
    }

    exprs = []
    for col_name, win in windows.items():
        win = max(1, win)

        rolling_min = pl.col("close").rolling_min(window_size=win, min_periods=win)
        rolling_max = pl.col("close").rolling_max(window_size=win, min_periods=win)

        exprs.append(
            (
                ((rolling_max - rolling_min) / rolling_min)
                .fill_nan(0.0)
                .fill_null(0.0)
                .cast(pl.Float32)
                .alias(col_name)
            )
        )

    return df.with_columns(exprs)

def get_volatility_index_for_indices(
    df_main: pl.DataFrame,
    entry_idxs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract volatility index values at trade entry positions.
    Returns:
        rng_24h_entry, rng_72h_entry, rng_1w_entry, rng_1m_entry
    """
    if entry_idxs is None:
        entry_idxs = np.asarray([], dtype=np.int64)
    else:
        entry_idxs = np.asarray(entry_idxs, dtype=np.int64)

    n = entry_idxs.shape[0]
    zero = np.zeros(n, dtype=np.float32)

    if df_main is None or df_main.height == 0 or n == 0:
        return zero, zero.copy(), zero.copy(), zero.copy()

    cols = ["rng_24h", "rng_72h", "rng_1w", "rng_1m"]
    out = []

    for c in cols:
        if c not in df_main.columns:
            out.append(zero.copy())
            continue

        arr = (
            df_main[c]
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


# Precompute Function
def precompute_all_possible_features(df: pl.DataFrame, run_cfg: dict) -> pl.DataFrame:
    import math

    if df is None or df.height == 0:
        run_cfg["lookback_map"] = {}
        return df

    df = df.clone()

    base_min = _as_int(run_cfg.get("BASE_MINUTES", 5), 5)
    modifier = _as_int(run_cfg.get("signal_timeframe_modifier", 3), 3)
    unit_mult = max(1, modifier)

    # --- 1. MA Periods ---
    raw_ma_periods = run_cfg.get("ma_periods", [])
    ma_periods_used = sorted({x for x in _as_int_list(raw_ma_periods) if x > 0})

    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)

    for i, period in enumerate(ma_periods_used):
        ma_name = f"kama_{chr(97 + i)}"
        n = max(1, period * unit_mult)

        df = df.with_columns([
            pl.col("close").ewm_mean(span=n, adjust=False).alias(ma_name)
        ])

    # --- 2. Stochastic ---
    use_stoch_val = run_cfg.get("use_stochastic", False)
    should_compute_stoch = _as_bool(use_stoch_val, False)

    if should_compute_stoch:
        ks = _as_int_list(run_cfg.get("stoch_k", [12])) or [12]
        ds = _as_int_list(run_cfg.get("stoch_d", [3])) or [3]
        ss = _as_int_list(run_cfg.get("stoch_s", [3])) or [3]

        for k in ks:
            for d in ds:
                for s in ss:
                    k_window = max(1, k * unit_mult)
                    d_window = max(1, d * unit_mult)
                    s_window = max(1, s * unit_mult)

                    col_name = f"stoch_k{k}_d{d}_s{s}"

                    s_min = pl.col("close").rolling_min(k_window, min_periods=1)
                    s_max = pl.col("close").rolling_max(k_window, min_periods=1)

                    df = df.with_columns([
                        (100.0 * (pl.col("close") - s_min) / (s_max - s_min)).fill_nan(50.0).alias("_k")
                    ]).with_columns([
                        pl.col("_k").ewm_mean(span=d_window, adjust=False).alias("_d")
                    ]).with_columns([
                        pl.col("_d").ewm_mean(span=s_window, adjust=False).alias(col_name)
                    ]).drop(["_k", "_d"])

    # --- 3. Breakouts ---
    lookbacks = _as_int_list(run_cfg.get("entry_lookback_units", []))
    for lb_units in lookbacks:
        if lb_units < 0:
            continue

        periods = lb_units * unit_mult
        hi = pl.col("high").rolling_max(periods).shift(1)
        lo = pl.col("low").rolling_min(periods).shift(1)

        df = df.with_columns([
            ((pl.col("close") - lo) / (hi - lo)).cast(pl.Float32).alias(f"breakout_{lb_units}u")
        ])

        if lb_units == 0:
            df = df.with_columns([
                pl.lit(1.0).cast(pl.Float32).alias("breakout_0u")
            ])

    # --- 4. BBW ---
    bbw_periods = sorted({x for x in _as_int_list(run_cfg.get("bbw_periods", [])) if x > 0})
    bbw_std = _as_float_list(run_cfg.get("bbw_std", [2.5])) or [2.5]
    rel_lookback = 500

    for p in bbw_periods:
        for s in bbw_std:
            n = max(1, p * unit_mult)
            col_name = f"bbw_p{p}_s{s}"

            rolling_mean = pl.col("close").rolling_mean(n)
            rolling_std = pl.col("close").rolling_std(n, ddof=0)
            raw_bbw = (2 * s * rolling_std) / rolling_mean

            b_min = raw_bbw.rolling_min(window_size=rel_lookback)
            b_max = raw_bbw.rolling_max(window_size=rel_lookback)

            df = df.with_columns([
                (100 * (raw_bbw - b_min) / (b_max - b_min))
                .fill_nan(50)
                .clip(0, 100)
                .cast(pl.Float32)
                .alias(col_name)
            ])

    # --- 5. Warmup / lookback ---
    max_k = max(_as_int_list(run_cfg.get("stoch_k", [0])) or [0])
    max_ma = max(ma_periods_used) if ma_periods_used else 0
    max_lb = max(lookbacks) if lookbacks else 0
    max_bbw = max(bbw_periods) if bbw_periods else 0

    absolute_max_units = max(max_k, max_ma, max_lb, max_bbw)
    required_hours = math.ceil((absolute_max_units * unit_mult * base_min) / 60) + 2

    # --- 6. Era Generation ---
    start_str = _as_str(run_cfg.get("grid_start_date", ""))
    end_str = _as_str(run_cfg.get("grid_end_date", ""))
    if not start_str or not end_str:
        run_cfg["lookback_map"] = {}
        return df, run_cfg
        
    interval = f"{_as_int(run_cfg.get('sl_tp_interval_months', 6), 6)}mo"

    start_dt = pl.select(pl.lit(start_str).str.to_datetime(time_zone="UTC")).item()
    end_dt = pl.select(pl.lit(end_str).str.to_datetime(time_zone="UTC")).item()

    era_series = pl.datetime_range(
        start=start_dt, end=end_dt, interval=interval, eager=True
    ).dt.truncate("1mo")

    run_cfg["lookback_map"] = {
        dt.strftime("%Y-%m"): int(required_hours) for dt in era_series
    }

    return df, run_cfg

# -------- signal generator (lean) ----------
def generate_filtered_signals(df_slice: pl.DataFrame, cfg: dict, df_main: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    if df_slice is None or not isinstance(df_slice, pl.DataFrame) or df_slice.height == 0:
        return pl.DataFrame([], schema=get_schema("signals"))

    df_slice = normalize_signals_times(df_slice, df_main=df_main)

    cond_buy = pl.lit(True)
    cond_sell = pl.lit(True)

    # --- 1. Entry Lookback Logic ---
    lb_units = _as_int(cfg.get("entry_lookback_units", 0), 0)

    if lb_units > 0:
        brk_col = f"breakout_{lb_units}u"
        if brk_col in df_slice.columns:
            cond_buy = cond_buy & (pl.col(brk_col) >= 0.8).fill_null(False)
            cond_sell = cond_sell & (pl.col(brk_col) <= 0.2).fill_null(False)

    # --- 2. Stochastic Logic ---
    if _as_bool(cfg.get("use_stochastic", False), False):
        stoch_col = cfg.get("stoch_col")
        lower = _as_float(cfg.get("stoch_lower", 30), 30)
        upper = _as_float(cfg.get("stoch_upper", 70), 70)

        if stoch_col and stoch_col in df_slice.columns:
            cond_buy = cond_buy & (pl.col(stoch_col) < lower).fill_null(False)
            cond_sell = cond_sell & (pl.col(stoch_col) > upper).fill_null(False)

    # --- 3. MA Logic ---
    ma_int = _as_int(cfg.get("ma_int", 0), 0)
    ma_periods = _as_int_list(cfg.get("ma_periods", []))

    if ma_int > 0:
        ma_reversion = _as_bool(cfg.get("ma_reversion", False), False)

        for i in range(len(ma_periods)):
            if (ma_int >> i) & 1:
                kama_col_name = f"kama_{chr(97 + i)}"
                if kama_col_name in df_slice.columns:
                    if not ma_reversion:
                        cond_buy = cond_buy & (pl.col("close") > pl.col(kama_col_name)).fill_null(False)
                        cond_sell = cond_sell & (pl.col("close") < pl.col(kama_col_name)).fill_null(False)
                    else:
                        cond_buy = cond_buy & (pl.col("close") < pl.col(kama_col_name)).fill_null(False)
                        cond_sell = cond_sell & (pl.col("close") > pl.col(kama_col_name)).fill_null(False)

    # --- 4. BBW Threshold Gate ---
    if _as_bool(cfg.get("use_bbw", False), False):
        p = _as_int(cfg.get("bbw_periods", 0), 0)
        s = _as_float(cfg.get("bbw_std", 0), 0.0)
        threshold = _as_float(cfg.get("bbw_thresholds", 50), 50.0)

        bbw_col = f"bbw_p{p}_s{s}"

        if bbw_col in df_slice.columns:
            vol_gate = (pl.col(bbw_col) <= threshold).fill_null(False)
            cond_buy = cond_buy & vol_gate
            cond_sell = cond_sell & vol_gate

    df_buys = (
        df_slice.filter(cond_buy)
        .select([
            pl.col("idx"),
            pl.col("time_ns"),
            pl.lit(1, dtype=pl.Int8).alias("side"),
            pl.lit(_as_int(cfg.get("regime_id", 0), 0), dtype=pl.Int32).alias("regime_id")
        ])
    )

    df_sells = (
        df_slice.filter(cond_sell)
        .select([
            pl.col("idx"),
            pl.col("time_ns"),
            pl.lit(-1, dtype=pl.Int8).alias("side"),
            pl.lit(_as_int(cfg.get("regime_id", 0), 0), dtype=pl.Int32).alias("regime_id")
        ])
    )

    df_out = pl.concat([df_buys, df_sells]).sort("idx", "side")
    return df_out


# -------- small util: compute selected ma_price_gap series name --------
def selected_gap_col_for_ma_int(ma_int: int) -> str:
    """
    Pick the first set bit (LSB = ma_a). If none set, fall back to 'ma_price_gap_a' if present.
    """
    try:
        mi = int(ma_int or 0)
    except Exception:
        mi = 0
    for i in range(4):
        if (mi >> i) & 1:
            return f"ma_price_gap_{chr(97 + i)}"
    return "ma_price_gap_a"