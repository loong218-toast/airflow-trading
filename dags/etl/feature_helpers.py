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
    modifier = int(run_cfg.get("signal_timeframe_modifier", 3)) # Default to 3 (15m)
    unit_mult = modifier

    # --- 1. MA Periods (Compute ALL as ma_a, ma_b, ma_c...) ---
    raw_ma_periods = run_cfg.get("ma_periods", []) or []
    if isinstance(raw_ma_periods, (int, float)):
        raw_ma_periods = [raw_ma_periods]

    ma_periods_used = sorted({int(x) for x in raw_ma_periods})

    # KAMA Constants (Fixed)
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)

    for i, period in enumerate(ma_periods_used):
        ma_name = f"kama_{chr(97 + i)}"
        # The 'n' period scaled by our modifier
        n = max(1, period * unit_mult)
        
        # Polars calculation for ER (Efficiency Ratio)
        change = (pl.col("close") - pl.col("close").shift(n)).abs()
        volatility = (pl.col("close") - pl.col("close").shift(1)).abs().rolling_sum(n)
        er = (change / volatility).fill_nan(0.0)
        
        # Calculate Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc).pow(2)
        
        # Note: True KAMA is recursive. In Polars precompute, we use an EWM 
        # approximation or a scan. For high-speed trading scripts, 
        # a weighted mean over the period is often used as a proxy.
        df = df.with_columns([
            pl.col("close").ewm_mean(span=n, adjust=False).alias(ma_name)
        ])

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
                    k_window = k * unit_mult
                    d_window = d * unit_mult
                    s_window = s * unit_mult

                    # Consistent naming: stoch_k12_d3_s3
                    col_name = f"stoch_k{k}_d{d}_s{s}"
                    
                    s_min = pl.col("close").rolling_min(k_window, min_periods=1)
                    s_max = pl.col("close").rolling_max(k_window, min_periods=1)
                    
                    df = df.with_columns([
                        (100.0 * (pl.col("close") - s_min) / (s_max - s_min)).fill_nan(50.0).alias("_k")
                    ]).with_columns([
                        # Change rolling_mean to ewm_mean
                        pl.col("_k").ewm_mean(span=d_window, adjust=False).alias("_d")
                    ]).with_columns([
                        # Change rolling_mean to ewm_mean
                        pl.col("_d").ewm_mean(span=s_window, adjust=False).alias(col_name)
                    ]).drop(["_k", "_d"])

    # --- 3. Breakouts (precompute for every requested entry_lookback_h) ---
    lookbacks = run_cfg.get("entry_lookback_units", [])
    if isinstance(lookbacks, int):
        lookbacks = [lookbacks]

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
                pl.lit(1.0).cast(pl.Float32).alias("breakout_0u") # 1.0 means 'always triggered'
            ])

# --- 4. Normalized Bollinger Band Width (BBW) ---
    bbw_periods = run_cfg.get("bbw_periods", [])
    bbw_std = run_cfg.get("bbw_std", [2.5])
    # We use a long lookback to find the "Era Min/Max" for volatility
    rel_lookback = 500 

    for p in bbw_periods:
        if p <= 0: continue 
            
        for s in bbw_std:
            n = max(1, p * unit_mult)
            col_name = f"bbw_p{p}_s{s}"
            
            # 1. Calculate Raw BBW
            rolling_mean = pl.col("close").rolling_mean(n)
            rolling_std = pl.col("close").rolling_std(n, ddof=0)
            raw_bbw = (2 * s * rolling_std) / rolling_mean
            
            # 2. Calculate Rolling Min/Max of the Width itself
            # This identifies the "Squeeze" (Min) and "Expansion" (Max)
            b_min = raw_bbw.rolling_min(window_size=rel_lookback)
            b_max = raw_bbw.rolling_max(window_size=rel_lookback)
            
            # 3. Create the 0-100 Relative Scale
            # 50 = Exactly the middle of recent volatility
            df = df.with_columns([
                (100 * (raw_bbw - b_min) / (b_max - b_min))
                .fill_nan(50)
                .clip(0, 100) # Ensure no outliers
                .cast(pl.Float32)
                .alias(col_name)
            ])


    # --- 5. NEW WARMUP LOGIC (Dynamic & Robust) ---
    # We find the single largest window used across all features to ensure safety.
    max_k = max(run_cfg.get("stoch_k", [0]))
    max_ma = max(ma_periods_used) if ma_periods_used else 0
    max_lb = max(lookbacks) if lookbacks else 0
    max_bbw = max(bbw_periods) if bbw_periods else 0
    
    # The largest "Unit" window
    absolute_max_units = max(max_k, max_ma, max_lb, max_bbw)
    
    # Total hours = (Units * Modifier * BaseMin) / 60
    # We add 2 hours as a "Buffer" for shifting and EWM stabilization
    required_hours = math.ceil((absolute_max_units * unit_mult * base_min) / 60) + 2

    # --- 6. Era Generation ---
    start_str = run_cfg["grid_start_date"]
    end_str = run_cfg["grid_end_date"]
    interval = f"{run_cfg.get('sl_tp_interval_months', 6)}mo"

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

    # 1. Early exit with the strict 4-column schema
    if df_slice is None or not isinstance(df_slice, pl.DataFrame) or df_slice.height == 0:
        return pl.DataFrame([], schema=CACHE_SIGNAL_SCHEMA)

    df_slice = normalize_signals_times(df_slice, df_main=df_main)

    # Initialize base conditions as True
    cond_buy = pl.lit(True)
    cond_sell = pl.lit(True)

    # --- 1. Entry Lookback Logic ---
    lb_units = int(cfg.get("entry_lookback_units", 0))

    # Only apply the filter if units > 0. 
    # If units == 0, we skip this and cond_buy stays True (passing through).
    if lb_units > 0:
        brk_col = f"breakout_{lb_units}u"
        if brk_col in df_slice.columns:
            cond_buy = cond_buy & (pl.col(brk_col) >= 0.8).fill_null(False)
            cond_sell = cond_sell & (pl.col(brk_col) <= 0.2).fill_null(False)

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

                kama_col_name = f"kama_{chr(97 + i)}"

                ma_col_name = f"ma_{chr(97 + i)}"
                if kama_col_name in df_slice.columns:
                    if not ma_reversion:
                        # Trend Following: Price above KAMA for Buy
                        cond_buy = cond_buy & (pl.col("close") > pl.col(kama_col_name)).fill_null(False)
                        cond_sell = cond_sell & (pl.col("close") < pl.col(kama_col_name)).fill_null(False)
                    else:
                        # Mean Reversion: Price below KAMA for Buy
                        cond_buy = cond_buy & (pl.col("close") < pl.col(kama_col_name)).fill_null(False)
                        cond_sell = cond_sell & (pl.col("close") > pl.col(kama_col_name)).fill_null(False)

    # --- 4. BBW Threshold Gate ---
    if cfg.get("use_bbw", False):
        p = cfg.get("bbw_periods")
        s = cfg.get("bbw_std")
        threshold = cfg.get("bbw_thresholds", 50) # Now 50 means "Median Volatility"
        
        bbw_col = f"bbw_p{p}_s{s}"
        
        if bbw_col in df_slice.columns:
            # Entry only if volatility is in the lower half of recent history
            vol_gate = (pl.col(bbw_col) <= threshold).fill_null(False)
            
            cond_buy = cond_buy & vol_gate
            cond_sell = cond_sell & vol_gate

    # --- 5. Evaluate both directions independently ---
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