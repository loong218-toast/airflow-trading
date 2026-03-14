# etl/feature_helpers.py
from typing import List, Optional, Dict
import polars as pl
import numpy as np
import hashlib
import json
import os
import gc

from etl.schema import get_schema, enforce_schema

# in-process caches (lightweight) — safe to keep for the lifetime of the worker
_STOCH_CACHE: Dict = {}
_LOOKBACK_CACHE: Dict = {}

def normalize_signals_times(df_signals: pl.DataFrame, df_main: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    if df_signals is None or df_signals.height == 0:
        return df_signals

    # 1) Force 'time' to Datetime (Naive) and standardize to Nanoseconds
    df = df_signals.with_columns(
        pl.col("time")
        .cast(pl.Datetime("ns")) # Force ns here
        .dt.replace_time_zone(None)
    )

    # 2) Create 'time_ns' (Int64)
    df = df.with_columns(
        pl.col("time").cast(pl.Int64).alias("time_ns")
    )

    # 3) Alignment
    if df_main is not None:
        # CRITICAL: Ensure df_main is also ns before grabbing the array
        main_times = (
            df_main.select(pl.col("time_ns").cast(pl.Int64))
            .to_numpy(allow_copy=False)
            .flatten()
        )
        sig_times = df["time_ns"].to_numpy(allow_copy=False)
        
        # Use 'left' side search for entry signals (find the exact bar start)
        idxs = np.searchsorted(main_times, sig_times, side="left")
        
        df = df.with_columns(
            pl.Series("idx", idxs).clip(0, df_main.height - 1).cast(pl.Int64)
        )

    return df

# -------- precompute --------
def precompute_all_possible_features(
    df: pl.DataFrame,
    run_cfg: dict,
) -> pl.DataFrame:
    if df is None or df.height == 0:
        return df

    base_min = int(run_cfg.get("BASE_MINUTES", 5))
    bar_mult = 60 // base_min
    
    # --- 1) Compute MAs First (Essential for later steps) ---
    ma_exprs = []
    ma_periods = run_cfg.get("ma_periods", []) or []
    for i, period in enumerate(ma_periods[:4]):
        window = max(1, int(period) * bar_mult)
        ma_name = f"ma_{chr(97 + i)}"
        ma_exprs.append(
            pl.col("close").rolling_mean(window, min_periods=1)
            .shift(1).cast(pl.Float32).alias(ma_name)
        )
    
    if ma_exprs:
        df = df.with_columns(ma_exprs)

    # --- 2) Compute Gaps and Other Indicators ---
    secondary_exprs = []
    for i in range(len(ma_exprs)):
        ma_name = f"ma_{chr(97 + i)}"
        gap_name = f"ma_price_gap_{chr(97 + i)}"
        # This matches MASTER_SCHEMA exactly: ma_price_gap_a, ma_price_gap_b, etc.
        secondary_exprs.append(
            (pl.col("close") / pl.col(ma_name) - 1.0).cast(pl.Float32).alias(gap_name)
        )

    # Stochastic
    stoch = run_cfg.get("stochastic", {}) or {}
    if stoch.get("use_stochastic"):
        k_win = stoch.get("stoch_k", 12)
        d_win = stoch.get("stoch_d", 12)
        slow_win = stoch.get("stoch_slow", 8)
        
        s_min = pl.col("close").rolling_min(k_win, min_periods=1)
        s_max = pl.col("close").rolling_max(k_win, min_periods=1)
        
        # Note: We compute %K, %D, %DSlow sequentially
        df = df.with_columns([
            (100.0 * (pl.col("close") - s_min) / (s_max - s_min)).fill_nan(50.0).alias("%K")
        ])
        df = df.with_columns([
            pl.col("%K").rolling_mean(d_win, min_periods=1).alias("%D")
        ])
        secondary_exprs.append(pl.col("%D").rolling_mean(slow_win, min_periods=1).alias("%DSlow"))

    # Breakouts
    lookbacks = run_cfg.get("entry_lookback_h", [])
    if isinstance(lookbacks, int): lookbacks = [lookbacks]
    
    for lb_h in lookbacks:
        if lb_h <= 0: continue
        periods = max(1, int((lb_h * 60) / base_min))
        hi = pl.col("close").rolling_max(periods).shift(1)
        lo = pl.col("close").rolling_min(periods).shift(1)
        secondary_exprs.append(
            ((pl.col("close") - lo) / (hi - lo)).cast(pl.Float32).alias(f"breakout_{lb_h}h")
        )

    if secondary_exprs:
        df = df.with_columns(secondary_exprs)

    # --- 3) MA-to-MA Gaps (Last step) ---
    ma_gap_exprs = []
    pairs = [("ma_a", "ma_b", "gap_ma_a_b"), ("ma_b", "ma_c", "gap_ma_b_c"), ("ma_c", "ma_d", "gap_ma_c_d")]
    for left, right, name in pairs:
        if left in df.columns and right in df.columns:
            ma_gap_exprs.append(
                pl.when(pl.col(left).is_null() | pl.col(right).is_null())
                .then(pl.lit(None).cast(pl.Float32))
                .otherwise((pl.col(left) / pl.col(right) - 1.0).cast(pl.Float32))
                .alias(name)
            )

    if ma_gap_exprs:
        df = df.with_columns(ma_gap_exprs)

    gc.collect()
    return df

# -------- signal generator (optimized for lean caching) --------
def generate_filtered_signals(df_slice: pl.DataFrame, cfg: dict, df_main: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """
    Generate side signals using only precomputed columns that exist in df_slice.
    - MA bitmask (cfg['ma_int']) is interpreted as bits for ma_a..ma_d (LSB = ma_a).
      When ma_int == 0, no MA constraint applied.
    - ma_reversion flips (close > ma) to (close < ma) for selected bits.
    - If use_entry_lookback True and entry_lookback_h>0, uses breakout_{h}h strength precomputed on close.
    - If use_stochastic True, uses %DSlow if present (otherwise blocks).
    """
    if df_slice is None or not isinstance(df_slice, pl.DataFrame) or df_slice.height == 0:
        return pl.DataFrame([], schema=get_schema("signals"))

    df_slice = normalize_signals_times(df_slice, df_main=df_main)

    # build boolean expressions starting from True Expr
    cond_buy: pl.Expr = pl.lit(True)
    cond_sell: pl.Expr = pl.lit(True)

    # Stochastic
    if bool(cfg.get("use_stochastic", False)) or (isinstance(cfg.get("stochastic"), dict) and cfg.get("stochastic", {}).get("use_stochastic")):
        if "%DSlow" in df_slice.columns:
            cond_buy = cond_buy & (pl.col("%DSlow") < 80)
            cond_sell = cond_sell & (pl.col("%DSlow") > 20)
        else:
            # precompute missing -> block all (no signal)
            return pl.DataFrame([], schema=get_schema("signals"))

    # Entry lookback (breakout) — uses breakout_{h}h column computed on close
    lb_h = int(cfg.get("entry_lookback_h", 0) or 0)
    if bool(cfg.get("use_entry_lookback", False)) and lb_h > 0:
        bcol = f"breakout_{lb_h}h"
        if bcol in df_slice.columns:
            cond_buy = cond_buy & (pl.col(bcol) >= 1.0)
            cond_sell = cond_sell & (pl.col(bcol) <= 0.0)
        else:
            # if requested but not present -> no signals
            return pl.DataFrame([], schema=get_schema("signals"))

    # MA bitmask logic
    ma_int = int(cfg.get("ma_int", 0) or 0)
    ma_reversion = bool(cfg.get("ma_reversion", False))
    # iterate over up to 4 MAs (ma_a..ma_d), LSB = ma_a
    ma_seen = False
    for i in range(4):
        bit = (ma_int >> i) & 1
        if bit == 0:
            continue
        ma_seen = True
        mcol = f"ma_{chr(97 + i)}"
        if mcol not in df_slice.columns:
            # if MA missing and required, no signals for this regime
            return pl.DataFrame([], schema=get_schema("signals"))
        if not ma_reversion:
            cond_buy = cond_buy & (pl.col("close") > pl.col(mcol))
            cond_sell = cond_sell & (pl.col("close") < pl.col(mcol))
        else:
            cond_buy = cond_buy & (pl.col("close") < pl.col(mcol))
            cond_sell = cond_sell & (pl.col("close") > pl.col(mcol))

    # Build side column and filter non-zero
    side_expr = (
        pl.when(cond_buy).then(pl.lit(1))
        .when(cond_sell).then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
    )
    
    df_out = df_slice.with_columns(side_expr.alias("side")).filter(pl.col("side") != 0)

    if df_out.height == 0:
        return pl.DataFrame([], schema=get_schema("signals"))

    # --- THE CRITICAL CHANGE IS HERE ---
    # 2. ULTRA-LEAN CACHE: Attach ONLY regime_id and drop all other indicator bloat
    df_out = df_out.with_columns(
        pl.lit(cfg.get("regime_id", 0)).cast(pl.Int32).alias("regime_id")
    ).select([
        "idx",
        "time_ns",
        "side",
        "regime_id"
    ])

    return enforce_schema(df_out, "signals", strict=True)

# -------- small util: compute selected ma_price_gap series name --------
def selected_gap_col_for_ma_int(ma_int: int) -> str:
    """
    Pick the first set bit (LSB = ma_a). If none set, fall back to 'ma_price_gap_a' if present.
    This function only returns the column-name string; callers may check existence in df.
    """
    try:
        mi = int(ma_int or 0)
    except Exception:
        mi = 0
    for i in range(4):
        if (mi >> i) & 1:
            return f"ma_price_gap_{chr(97 + i)}"
    # no bits set -> default to ma_price_gap_a
    return "ma_price_gap_a"