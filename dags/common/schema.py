# common/schema.py
"""
Centralized Polars schema registry and helpers.

Exports:
- MASTER_SCHEMA, EQUITY_SCHEMA, SIGNAL_SCHEMA, BACKTEST_SCHEMA, DF_MAIN_SCHEMA
- SCHEMA_REGISTRY
- enforce_schema(df, schema_type, strict=False): cast/add missing cols, optional drop extras
- get_schema(schema_type)
- cast_to_master(data) / cast_to_schema(data, schema_type)
- validate_dataframe(df, schema_type) -> list of mismatch tuples (col, expected, actual)
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import polars as pl
import logging

_LOG = logging.getLogger(__name__)

# -------------------------
# Canonical schemas (polars dtypes)
# -------------------------
MASTER_SCHEMA: Dict[str, pl.DataType] = {
    "regime_id": pl.Int32,
    "era_int": pl.Int64,
    "side": pl.Int8,

    "exit_window_h": pl.Int32,
    "SL": pl.Float32,
    "TP": pl.Float32,

    "use_trailing_sl": pl.Boolean,
    "trailing_sl_pct": pl.Float32,
    "trailing_sl_interval": pl.Int32,
    "trailing_sl_stop_at_pos": pl.Boolean,

    "use_limit_entry": pl.Boolean,
    "limit_order_expiry_bars": pl.Int32,
    "trade_window_interval": pl.Int32,

    "total_pos": pl.Int32,
    "win_pos": pl.Int32,
    "balance": pl.Float32,
    "max_drawdown": pl.Float32,
    "max_consecutive_losses": pl.Int32,

    "signal_json": pl.String,
}

EQUITY_SCHEMA: Dict[str, pl.DataType] = {
    "regime_id": pl.Int32,
    "era_int": pl.Int64,
    "side": pl.Int8,
    "SL": pl.Float32,
    "TP": pl.Float32,
    "time_ns": pl.Int64,
    "entry_idx": pl.Int64,
    "exit_idx": pl.Int64,
    "pnl_pct": pl.Float32,
    "equity": pl.Float32,
}

TRADE_ML_SCHEMA: Dict[str, pl.DataType] = {
    "regime_id": pl.Int32,
    "era_int": pl.Int64,
    "side": pl.Int8,
    "SL": pl.Float32,
    "TP": pl.Float32,
    "SL_hit": pl.Float32,
    "TP_hit": pl.Float32,

    "use_limit_entry": pl.Boolean,
    "limit_order_expiry_bars": pl.Int32,
    "trade_window_interval": pl.Int32,

    "signal_idx": pl.Int64,
    "signal_time_ns": pl.Int64,
    "signal_price": pl.Float32,

    "order_idx": pl.Int64,
    "order_time_ns": pl.Int64,
    "order_price": pl.Float32,
    "order_mode": pl.Int8,

    "fill_status": pl.Int8,
    "entry_idx": pl.Int64,
    "entry_time_ns": pl.Int64,
    "entry_price": pl.Float32,

    "exit_idx": pl.Int64,
    "exit_time_ns": pl.Int64,
    "exit_price": pl.Float32,
    "exit_reason": pl.Int8,

    "fill_delay_bars": pl.Int32,
    "pnl_pct": pl.Float32,
}

CACHE_SIGNAL_SCHEMA: Dict[str, pl.DataType] = {
    "idx": pl.Int64,           # Pointer to the exact row in df_main
    "time_ns": pl.Int64,       # Universal timestamp for safety
    "side": pl.Int8,           # 1 for Buy, -1 for Sell
    "regime_id": pl.Int32,     # Linking ID for the regime
}

CACHE_BACKTEST_SCHEMA: Dict[str, pl.DataType] = {
    "sig_n": pl.Int64,
    "sig_min_ns": pl.Int64,
    "sig_max_ns": pl.Int64,
    "SL": pl.Float32,
    "TP": pl.Float32,
    "side": pl.Int8,
    "exit_window_h": pl.Int32,
    "entry_idx": pl.List(pl.Int64),
    "exit_idx": pl.List(pl.Int64),
    "entry_price": pl.List(pl.Float32),
    "exit_price": pl.List(pl.Float32),
    "ret": pl.List(pl.Float32),
    "exit_reason": pl.List(pl.Int8),
    "regime_id": pl.Int32,
    "use_trailing_sl": pl.Boolean,
    "trailing_sl_pct": pl.Float32,
    "trailing_sl_interval": pl.Int32,
    "trailing_sl_stop_at_pos": pl.Boolean,
    "use_limit_entry": pl.Boolean,
    "limit_order_expiry_bars": pl.Int32,
}

DF_MAIN_SCHEMA: Dict[str, pl.DataType] = {
    "pair": pl.Utf8,
    "market_type": pl.Utf8,
    "time": pl.Datetime("ns"),
    "time_ns": pl.Int64,
    "open": pl.Float32,
    "high": pl.Float32,
    "low": pl.Float32,
    "close": pl.Float32,
    "volume": pl.Float32,
    "funding_rate": pl.Float32,
    "spread": pl.Float32,
    "era_int": pl.Int64,
    "idx": pl.Int64,
}

# Registry
SCHEMA_REGISTRY: Dict[str, Dict[str, pl.DataType]] = {
    "master": MASTER_SCHEMA,
    "equity": EQUITY_SCHEMA,
    "trade_ml": TRADE_ML_SCHEMA,
    "signals": CACHE_SIGNAL_SCHEMA,
    "backtest": CACHE_BACKTEST_SCHEMA,
    "df_main": DF_MAIN_SCHEMA,
}

CLEAN_SCHEMA = DF_MAIN_SCHEMA

# Optional per-schema metadata (single place to adjust behavior)
SCHEMA_METADATA: Dict[str, Dict[str, Any]] = {
    "equity": {
        "key_columns": ["time_ns", "pnl_pct", "equity"],
        "min_non_null_fraction": 0.01,
    },
    "master": {
        "key_columns": ["balance", "total_pos", "max_drawdown"],
    },
   "trade_ml": {
        "key_columns": ["signal_idx", "entry_idx", "exit_idx", "pnl_pct"],
        "min_non_null_fraction": 0.01,
    },

}


def get_schema_key_columns(schema_type: str) -> List[str]:
    """
    Return the canonical 'key' columns for quick fragment detection.
    If not explicitly configured in SCHEMA_METADATA, fall back to a heuristic:
      - prefer columns with names like time_ns/pnl_pct/equity or first 3 numeric columns.
    """
    meta = SCHEMA_METADATA.get(schema_type, {})
    keys = meta.get("key_columns")
    if keys:
        return list(keys)

    # heuristic fallback
    schema = get_schema(schema_type)
    candidates = []
    for prefer in ("time_ns", "pnl_pct", "equity", "entry_idx", "exit_idx"):
        if prefer in schema:
            candidates.append(prefer)
    if candidates:
        return candidates[:3]

    # last-resort: first three columns from the canonical schema
    return list(schema.keys())[:3]


def _non_null_fraction(series: pl.Series) -> float:
    if series is None:
        return 0.0
    nulls = int(series.null_count())
    total = series.len()
    return 0.0 if total == 0 else float(total - nulls) / float(total)


def classify_fragment(df: Optional[pl.DataFrame], schema_type: str, min_non_null_fraction: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyze a polars DataFrame fragment and return attributes useful for deciding
    whether the fragment should be staged as `schema_type`.

    Returns a dict:
      {
        "is_like": bool,                       # passes heuristic check for schema_type
        "key_columns": [...],                  # key columns used for test
        "non_null_fractions": {col: frac},    # per-key-col non-null fraction
        "missing_key_columns": [...],          # missing key columns
        "other_schema_columns": [...],         # columns that match other canonical schemas (e.g. master)
      }
    """
    if df is None or df.height == 0:
        return {
            "is_like": False,
            "key_columns": [],
            "non_null_fractions": {},
            "missing_key_columns": [],
            "other_schema_columns": [],
        }

    keys = get_schema_key_columns(schema_type)
    if min_non_null_fraction is None:
        min_non_null_fraction = float(SCHEMA_METADATA.get(schema_type, {}).get("min_non_null_fraction", 0.01))

    non_null_fracs = {}
    missing = []
    for k in keys:
        if k not in df.columns:
            non_null_fracs[k] = 0.0
            missing.append(k)
        else:
            non_null_fracs[k] = _non_null_fraction(df[k])

    # determine if fragment is "like" this schema:
    # heuristic: at least one key column exists and has non-null fraction >= threshold,
    # and not all key columns are missing.
    any_key_exists = any(k in df.columns for k in keys)
    has_enough_non_null = any(frac >= min_non_null_fraction for frac in non_null_fracs.values())
    is_like = any_key_exists and has_enough_non_null

    # detect columns that belong to other canonical schemas (useful to detect master rows)
    other_cols = []
    master_cols = set(get_schema("master").keys())
    for c in df.columns:
        if c in master_cols:
            other_cols.append(c)

    return {
        "is_like": bool(is_like),
        "key_columns": keys,
        "non_null_fractions": non_null_fracs,
        "missing_key_columns": missing,
        "other_schema_columns": other_cols,
    }

# -------------------------
# Helpers
# -------------------------
def get_schema(schema_type: str) -> Dict[str, pl.DataType]:
    if schema_type not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema_type {schema_type}")
    return SCHEMA_REGISTRY[schema_type]

def enforce_schema(df: Optional[pl.DataFrame], schema_type: str, strict: bool = False) -> pl.DataFrame:
    target_schema = get_schema(schema_type)
    
    if df is None or df.height == 0:
        return pl.DataFrame([], schema=target_schema)

    current_schema = df.schema
    cast_exprs = []

    for col_name, target_dtype in target_schema.items():
        if col_name in df.columns:
            expr = pl.col(col_name)
            curr_dtype = current_schema[col_name]

            # ONLY apply numeric cleaning if the column is currently a Float or Int
            # This prevents the `is_not_nan` error on `null` dtypes.
            if curr_dtype in (pl.Float32, pl.Float64):
                expr = expr.fill_nan(None)
            
            if curr_dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
                if target_dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                    # Ensure we handle BOTH NaNs and Nulls before casting to Int
                    if curr_dtype in (pl.Float32, pl.Float64):
                        expr = expr.fill_nan(None)
                    expr = expr.fill_null(0).cast(target_dtype)

            # Direct cast is the safest way to move from `null` -> `Float32`
            cast_exprs.append(expr.cast(target_dtype).alias(col_name))
        else:
            # Missing: inject nulls
            cast_exprs.append(pl.lit(None).cast(target_dtype).alias(col_name))

    if not strict:
        for c in df.columns:
            if c not in target_schema:
                # Cast dynamic features to Float32 by default to keep memory low
                if df.schema[c] == pl.Float64:
                    cast_exprs.append(pl.col(c).cast(pl.Float32))
                else:
                    cast_exprs.append(pl.col(c))

    result = df.select(cast_exprs)
    
    # NEW DEBUG BLOCK
    if schema_type == "signals" and result.height > 0:
        null_count = result["side"].null_count()
        if null_count == result.height:
             _LOG.error(f"⚠️ SCHEMA CRASH: 'side' column is 100% NULL after enforcing {schema_type}!")

    return result

def cast_to_schema(data: Any, schema_type: str) -> pl.DataFrame:
    """
    Convert a dict or list-of-dicts (or a polars DataFrame) to a DataFrame conforming to schema_type.
    Usage:
      pl.DataFrame([raw_dict]).pipe(enforce_schema, "master")
    or
      cast_to_schema([raw_dict], "master")
    """
    if isinstance(data, pl.DataFrame):
        return enforce_schema(data, schema_type)
    try:
        df = pl.DataFrame(data)
    except Exception as e:
        _LOG.exception("cast_to_schema: failed to build DataFrame from data: %s", e)
        raise
    return enforce_schema(df, schema_type)
