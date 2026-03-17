# etl/feature_helpers.py
from typing import List, Optional, Dict, Tuple, Any
import polars as pl
import numpy as np
import gc

from etl.schema import get_schema, enforce_schema

# in-process caches (lightweight) — safe to keep for the lifetime of the worker
_STOCH_CACHE: Dict = {}
_LOOKBACK_CACHE: Dict = {}

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

# -------- helper: safely extract per-trade gap arrays (entry/exit) ----------
def get_ma_price_gaps_for_indices(
    df_main: pl.DataFrame,
    entry_idxs: np.ndarray,
    exit_idxs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    n_entry = entry_idxs.shape[0] if entry_idxs is not None else 0
    n_exit = exit_idxs.shape[0] if exit_idxs is not None else 0

    # Initialize results
    gap_a_entry, gap_b_entry = np.full(n_entry, np.nan, dtype=np.float32), np.full(n_entry, np.nan, dtype=np.float32)
    gap_a_exit, gap_b_exit = np.full(n_exit, np.nan, dtype=np.float32), np.full(n_exit, np.nan, dtype=np.float32)

    if df_main is None or df_main.height == 0:
        return gap_a_entry, gap_b_entry, gap_a_exit, gap_b_exit

    # 1. Grab raw prices and MAs (guaranteed to exist by precompute)
    close_arr = df_main["close"].to_numpy().astype(np.float32)
    
    for i, char in enumerate(['a', 'b']):
        ma_col = f"ma_{char}"
        if ma_col not in df_main.columns:
            continue
            
        ma_arr = df_main[ma_col].to_numpy().astype(np.float32)
        
        # 2. Calculate Gaps only for the needed indices to save memory
        # Entry Gaps
        valid_entry = (entry_idxs >= 0) & (entry_idxs < ma_arr.shape[0])
        if valid_entry.any():
            idx = entry_idxs[valid_entry]
            # (Price / MA) - 1
            gap_vals = (close_arr[idx] / ma_arr[idx]) - 1.0
            if i == 0: gap_a_entry[valid_entry] = gap_vals
            else: gap_b_entry[valid_entry] = gap_vals

        # Exit Gaps
        valid_exit = (exit_idxs >= 0) & (exit_idxs < ma_arr.shape[0])
        if valid_exit.any():
            idx = exit_idxs[valid_exit]
            gap_vals = (close_arr[idx] / ma_arr[idx]) - 1.0
            if i == 0: gap_a_exit[valid_exit] = gap_vals
            else: gap_b_exit[valid_exit] = gap_vals

    return gap_a_entry, gap_b_entry, gap_a_exit, gap_b_exit


# Precompute Function
def precompute_all_possible_features(df: pl.DataFrame, run_cfg: dict) -> pl.DataFrame:
    """
    Precompute feature columns (MA, stochastic, breakouts) and produce a per-era lookback_map (hours).
    The function mutates run_cfg by setting run_cfg['lookback_map'] to a dict:
       { "YYYY-MM": lookback_hours, ... }

    Requirements in run_cfg:
      - "BASE_MINUTES" (int)
      - "sl_tp_interval_months" (int)
      - "grid_start_date", "grid_end_date" (ISO datetimes)
      - "ma_periods" and "entry_lookback_h" are used to compute needed lookback
    """

    import math

    if df is None or df.height == 0:
        # still ensure run_cfg gets an empty map (explicit)
        run_cfg["lookback_map"] = {}
        return df

    df = df.clone()

    base_min = int(run_cfg.get("BASE_MINUTES", 5))
    bar_mult = max(1, 60 // base_min)

    # --- 1. MA Periods (Compute ALL as ma_a, ma_b, ma_c...) ---
    raw_ma_periods = run_cfg.get("ma_periods", []) or []
    if isinstance(raw_ma_periods, (int, float)):
        raw_ma_periods = [raw_ma_periods]

    ma_periods_used = sorted({int(x) for x in raw_ma_periods})

    ma_exprs = []
    for i, period in enumerate(ma_periods_used):
        ma_name = f"ma_{chr(97 + i)}"  # ma_a, ma_b, ma_c...
        window = max(1, int(period) * bar_mult)
        ma_exprs.append(
            pl.col("close").rolling_mean(window, min_periods=1)
            .shift(1).cast(pl.Float32).alias(ma_name)
        )
    if ma_exprs:
        df = df.with_columns(ma_exprs)

    # --- 2. Stochastic (pct_d_slow) ---
    use_stoch_val = run_cfg.get("use_stochastic", False)
    should_compute_stoch = use_stoch_val if isinstance(use_stoch_val, bool) else any(use_stoch_val)
    
    if should_compute_stoch:
        ks = run_cfg.get("stoch_k", [12])
        ds = run_cfg.get("stoch_d", [3])
        ss = run_cfg.get("stoch_s", [3])
        
        for k in ks:
            for d in ds:
                for s in ss:
                    # Consistent naming: stoch_k12_d3_s3
                    col_name = f"stoch_k{k}_d{d}_s{s}"
                    
                    s_min = pl.col("close").rolling_min(k, min_periods=1)
                    s_max = pl.col("close").rolling_max(k, min_periods=1)
                    
                    # We use a temporary scope to avoid column name collisions
                    df = df.with_columns([
                        (100.0 * (pl.col("close") - s_min) / (s_max - s_min)).fill_nan(50.0).alias("_k")
                    ]).with_columns([
                        pl.col("_k").rolling_mean(d, min_periods=1).alias("_d")
                    ]).with_columns([
                        pl.col("_d").rolling_mean(s, min_periods=1).alias(col_name)
                    ]).drop(["_k", "_d"])

    # --- 3. Breakouts (precompute for every requested entry_lookback_h) ---
    lookbacks = run_cfg.get("entry_lookback_h", [])
    if isinstance(lookbacks, int):
        lookbacks = [lookbacks]
    for lb_h in lookbacks:
        if lb_h < 0:
            continue
        periods = max(1, int((lb_h * 60) / base_min))
        hi = pl.col("high").rolling_max(periods).shift(1)
        lo = pl.col("low").rolling_min(periods).shift(1)
        df = df.with_columns([
            ((pl.col("close") - lo) / (hi - lo)).cast(pl.Float32).alias(f"breakout_{lb_h}h")
        ])

        if lb_h == 0:
            # Create a "Neutral" column. 
            # Note: This will NOT trigger trades with your current (>=1.0 / <=0.0) logic.
            df = df.with_columns([
                pl.lit(None).cast(pl.Float32).alias("breakout_0h")
            ])
            continue


    # 2. Warmup Math (The Single Source of Truth)
    raw_ma = run_cfg.get("ma_periods", [])
    if isinstance(raw_ma, (int, float)): raw_ma = [raw_ma]
    max_ma_val = max(raw_ma) if raw_ma else 0
    
    # Hours required for indicators to be valid
    ma_warmup_h = math.ceil((max_ma_val * bar_mult * base_min) / 60)
    
    raw_lb = run_cfg.get("entry_lookback_h", [])
    if isinstance(raw_lb, int): raw_lb = [raw_lb]
    max_entry_h = max(raw_lb) if raw_lb else 0

    lb_max = int(max(run_cfg.get("entry_lookback_h", [24])))
    ma_max_h = math.ceil((max(run_cfg.get("ma_periods", [200])) * base_min) / 60)
    required_hours = ma_max_h + lb_max + 1 # +1 for safety

    # 3. Era Generation (The Polars Way)
    # Extract string dates from cfg and create the range
    start_str = run_cfg["grid_start_date"]
    end_str = run_cfg["grid_end_date"]
    interval = f"{run_cfg.get('sl_tp_interval_months', 6)}mo"

    start_dt = pl.select(pl.lit(start_str).str.to_datetime(time_zone="UTC")).item()
    end_dt = pl.select(pl.lit(end_str).str.to_datetime(time_zone="UTC")).item()

    # Now create the range using the native datetime objects
    era_series = pl.datetime_range(
        start=start_dt,
        end=end_dt,
        interval=interval,
        eager=True
    ).dt.truncate("1mo")

    run_cfg["lookback_map"] = {
        dt.strftime("%Y-%m"): int(required_hours) 
        for dt in era_series
    }
    return df, run_cfg  # Return both!

# -------- signal generator (lean) ----------
def generate_filtered_signals(df_slice: pl.DataFrame, cfg: dict, df_main: Optional[pl.DataFrame] = None) -> pl.DataFrame:

    # Generate signals candle by candle first

    # 1. Early exit with the strict 4-column schema
    if df_slice is None or not isinstance(df_slice, pl.DataFrame) or df_slice.height == 0:
        return pl.DataFrame([], schema=CACHE_SIGNAL_SCHEMA)

    df_slice = normalize_signals_times(df_slice, df_main=df_main)

    # Initialize base conditions as True
    cond_buy = pl.lit(True)
    cond_sell = pl.lit(True)

    # --- 1. Entry Lookback Logic ---
    lb_h = int(cfg.get("entry_lookback_h", 0))
    if lb_h > 0:
        brk_col = f"breakout_{lb_h}h"
        if brk_col in df_slice.columns:
            # .fill_null(False) ensures that warmup periods don't trigger trades
            # AND don't break the logical chain.
            cond_buy = cond_buy & (pl.col(brk_col) >= 1.0).fill_null(False)
            cond_sell = cond_sell & (pl.col(brk_col) <= 0.0).fill_null(False)

    # --- 2. Stochastic Logic (CLEANED) ---
    if cfg.get("use_stochastic", False):
        stoch_col = cfg.get("stoch_col")
        lower = cfg.get("stoch_lower")
        upper = cfg.get("stoch_upper")
        
        # If the specific precomputed column exists, use it
        if stoch_col and stoch_col in df_slice.columns:
            cond_buy = cond_buy & (pl.col(stoch_col) < lower).fill_null(False)
            cond_sell = cond_sell & (pl.col(stoch_col) > upper).fill_null(False)

    # --- 3. MA Logic (Trend Following Only) ---
    ma_int = int(cfg.get("ma_int", 0))
    ma_periods = cfg.get("ma_periods", [])

    if ma_int > 0:
        ma_reversion = cfg.get("ma_reversion", False)
        for i in range(len(ma_periods)):
            if (ma_int >> i) & 1:
                ma_col_name = f"ma_{chr(97 + i)}"
                if ma_col_name in df_slice.columns:
                    if not ma_reversion:
                        cond_buy = cond_buy & (pl.col("close") > pl.col(ma_col_name)).fill_null(False)
                        cond_sell = cond_sell & (pl.col("close") < pl.col(ma_col_name)).fill_null(False)
                    else:
                        cond_buy = cond_buy & (pl.col("close") < pl.col(ma_col_name)).fill_null(False)
                        cond_sell = cond_sell & (pl.col("close") > pl.col(ma_col_name)).fill_null(False)

    # --- 4. Evaluate and Return Strict 4 Columns ---
    # We switch to lazy evaluation here for maximum Polars performance
    df_out = (
        df_slice.with_columns([
            pl.when(cond_buy).then(pl.lit(1))
              .when(cond_sell).then(pl.lit(-1))
              .otherwise(pl.lit(0))
              .alias("side"),
            pl.lit(cfg.get("regime_id", 0)).cast(pl.Int32).alias("regime_id")
        ])
        .filter(pl.col("side") != 0)
        .select(["idx", "time_ns", "side", "regime_id"]) # Ensure strict output
    )

    # --- 4. Evaluate both directions independently ---
    # Create Buy Signals
    df_buys = (
        df_slice.filter(cond_buy)
        .select([
            pl.col("idx"), 
            pl.col("time_ns"), 
            pl.lit(1, dtype=pl.Int8).alias("side"),
            pl.lit(cfg.get("regime_id", 0), dtype=pl.Int32).alias("regime_id")
        ])
    )

    # Create Sell Signals
    df_sells = (
        df_slice.filter(cond_sell)
        .select([
            pl.col("idx"), 
            pl.col("time_ns"), 
            pl.lit(-1, dtype=pl.Int8).alias("side"),
            pl.lit(cfg.get("regime_id", 0), dtype=pl.Int32).alias("regime_id")
        ])
    )

    # Vertical Concat to allow two signals at the same index/time
    df_out = pl.concat([df_buys, df_sells]).sort("idx", "side")

        # TEMPORARY DEBUG
    if df_out.height == 0:
        # Check if indicators even have data
        valid_brk = df_slice[f"breakout_{lb_h}h"].is_not_null().sum() if lb_h > 0 else "N/A"
    # Make sure to pass CACHE_SIGNAL_SCHEMA if enforce_schema expects a dict, 
    # or ensure get_schema("signals") in your schema.py is updated to these 4 columns.
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