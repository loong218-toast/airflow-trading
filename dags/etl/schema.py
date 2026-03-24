# etl/schema.py
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
    # 1. Identifiers & Temporal Context
    "regime_id": pl.Int32,
    "era_int": pl.Int64,
    "side": pl.Int8,

    # 2. Strategy Hyperparameters
    "ma_int": pl.Int32,
    "ma_reversion": pl.Boolean,
    "entry_lookback_units": pl.Int32,
    "exit_window_h": pl.Int32,
    "use_stochastic": pl.Boolean,
    "stoch_key": pl.String,   # "12-3-3 (20/80)" (Easy for Humans)
    "use_bbw": pl.Boolean,
    "bbw_periods": pl.Int32,
    "bbw_std": pl.Float32,
    "bbw_thresholds": pl.Int32,
    "SL": pl.Float32,
    "TP": pl.Float32,

    # 4. Performance Metrics
    "total_pos": pl.Int32,
    "win_pos": pl.Int32,
    "balance": pl.Float32,
    "max_drawdown": pl.Float32,
}

EQUITY_SCHEMA: Dict[str, pl.DataType] = {
    "regime_id": pl.Int32,
    "era_int": pl.Int64,
    "side": pl.Int8,
    "SL": pl.Float32,
    "TP": pl.Float32,
    "time_ns": pl.Int64,
    "entry_idx": pl.Int64,       # Individual Trade Detail
    "exit_idx": pl.Int64,        # Individual Trade Detail
    "pnl_pct": pl.Float32,
    "equity": pl.Float32,
    "ma_p_gap_a_entry": pl.Float32,
    "ma_p_gap_b_entry": pl.Float32,
    "ma_p_gap_a_exit":  pl.Float32,
    "ma_p_gap_b_exit": pl.Float32,

    "rng_24h_entry": pl.Float32,
    "rng_72h_entry": pl.Float32,
    "rng_1w_entry": pl.Float32,
    "rng_1m_entry": pl.Float32
}

CACHE_SIGNAL_SCHEMA: Dict[str, pl.DataType] = {
    "idx": pl.Int64,           # Pointer to the exact row in df_main
    "time_ns": pl.Int64,       # Universal timestamp for safety
    "side": pl.Int8,           # 1 for Buy, -1 for Sell
    "regime_id": pl.Int32,     # Linking ID for the regime
}

CACHE_BACKTEST_SCHEMA: Dict[str, pl.DataType] = {
    "sig_n": pl.Int64,        # Number of signals in this batch
    "sig_min_ns": pl.Int64,   # Hash/Signature of the signal batch
    "sig_max_ns": pl.Int64,   # Hash/Signature of the signal batch
    "SL": pl.Float32,         # The risk parameter used
    "TP": pl.Float32,         # The risk parameter used
    "side": pl.Int8,          # 1 for Buy, -1 for Sell
    "exit_window_h": pl.Int32,# The exit time parameter used
    "entry_idx": pl.List(pl.Int64),  # Array of actual trade entries
    "exit_idx": pl.List(pl.Int64),   # Array of actual trade exits
    "ret": pl.List(pl.Float32),      # Array of trade returns (use Float32 for RAM savings)
    "regime_id": pl.Int32,           # Link back to regime
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
    "signals": CACHE_SIGNAL_SCHEMA,
    "backtest": CACHE_BACKTEST_SCHEMA,
    "df_main": DF_MAIN_SCHEMA,
}

CLEAN_SCHEMA = DF_MAIN_SCHEMA

# Optional per-schema metadata (single place to adjust behavior)
SCHEMA_METADATA: Dict[str, Dict[str, Any]] = {
    # you can change key_columns in future if you want a different detection rule
    "equity": {
        "key_columns": ["time_ns", "pnl_pct", "equity"],
        # the fraction of non-null values (per-column) required to consider the fragment as trade-like
        "min_non_null_fraction": 0.01,
    },
    "master": {
        # master-specific columns used to detect master-like files accidentally written into equity
        "key_columns": ["balance", "total_pos", "max_drawdown"],
    },
    # other schemas may add metadata later...
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


def is_fragment_like_schema(df: Optional[pl.DataFrame], schema_type: str, min_non_null_fraction: Optional[float] = None) -> bool:
    return bool(classify_fragment(df, schema_type, min_non_null_fraction)["is_like"])

# -------------------------
# Helpers
# -------------------------
def get_schema(schema_type: str) -> Dict[str, pl.DataType]:
    if schema_type not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema_type {schema_type}")
    return SCHEMA_REGISTRY[schema_type]

def _is_int_dtype(dt: pl.DataType) -> bool:
    # simple containment check against known int dtypes
    return dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64)

def _is_float_dtype(dt: pl.DataType) -> bool:
    return dt in (pl.Float32, pl.Float64)

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

def cast_to_master(data: Any) -> pl.DataFrame:
    """Convenience wrapper for the master schema."""
    return cast_to_schema(data, "master")

def validate_dataframe(df: pl.DataFrame, schema_type: str) -> List[Tuple[str, Optional[pl.DataType], Optional[pl.DataType]]]:
    """
    Non-destructive validation helper.
    Returns a list of mismatches as tuples: (column, expected_dtype, actual_dtype)
    If a column is missing it's reported with actual_dtype=None.
    If there are no mismatches an empty list is returned.
    """
    schema = get_schema(schema_type)
    mismatches: List[Tuple[str, Optional[pl.DataType], Optional[pl.DataType]]] = []
    if df is None:
        for col, dtype in schema.items():
            mismatches.append((col, dtype, None))
        return mismatches

    for col, dtype in schema.items():
        if col not in df.columns:
            mismatches.append((col, dtype, None))
        else:
            actual = df.schema.get(col)
            # Compare canonical repr where possible
            if actual != dtype:
                mismatches.append((col, dtype, actual))
    return mismatches