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
    "entry_lookback_h": pl.Int32,
    "exit_window_h": pl.Int32,
    "SL": pl.Float32,
    "TP": pl.Float32,

    # 3. Market State / Features (Gaps)
    "ma_price_gap": pl.Float32,
    "ma_price_gap_a": pl.Float32,
    "ma_price_gap_b": pl.Float32,
    "ma_price_gap_c": pl.Float32,

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
    "pnl_pct": pl.Float32,
    "equity": pl.Float32,
}

CACHE_SIGNAL_SCHEMA: Dict[str, pl.DataType] = {
    "idx": pl.Int64,           # Pointer to the exact row in df_main
    "time_ns": pl.Int64,       # Universal timestamp for safety
    "side": pl.Int8,           # 1 for Buy, -1 for Sell
    "regime_id": pl.Int32,     # Linking ID for the regime
    "ma_int": pl.Int32,        # Bitmask for active MAs (00, 01, 10, 11 etc.)
    "ma_reversion": pl.Boolean # Strategy mode: True (Mean Rev), False (Trend)
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
                cast_exprs.append(pl.col(c))

    return df.select(cast_exprs)

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