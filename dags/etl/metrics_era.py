# metrics_era.py
from __future__ import annotations

import os
from typing import Sequence, Dict, Any, Optional

import pandas as pd
import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

amp_data = [
    {
        "era_int": 20230901,
        "era_close_time": "2023-09-01 23:55:00+00:00",
        "window_rows": 51840,
        "window_min_close": 25003.56,
        "window_max_close": 63690.12,
        "regime_amp_index": 1.547242,
        "regime_amp_pct": 154.724207
    },
    {
        "era_int": 20240301,
        "era_close_time": "2024-03-01 23:55:00+00:00",
        "window_rows": 51840,
        "window_min_close": 49338.49,
        "window_max_close": 73628.42,
        "regime_amp_index": 0.492312,
        "regime_amp_pct": 49.231199
    },
    {
        "era_int": 20240901,
        "era_close_time": "2024-09-01 23:55:00+00:00",
        "window_rows": 51840,
        "window_min_close": 52738.00,
        "window_max_close": 108984.96,
        "regime_amp_index": 1.066536,
        "regime_amp_pct": 106.653570
    },
    {
        "era_int": 20250301,
        "era_close_time": "2025-03-01 23:55:00+00:00",
        "window_rows": 51840,
        "window_min_close": 74610.00,
        "window_max_close": 124243.32,
        "regime_amp_index": 0.665237,
        "regime_amp_pct": 66.523683
    }
]


def calculate_optimal_tp(current_vol_idx, base_tp=1.4):
    # 1. Baseline regime amp (your established 'quiet' market floor)
    baseline_vol = 0.418

    # 2. Dynamic Calculation (Inverse Scaling)
    # If Vol is high, TP shrinks. If Vol is low, TP expands.
    scaled_tp = base_tp * (baseline_vol / current_vol_idx)

    # 3. Apply Professional Guardrails
    # - Floor (0.8): Protects against slippage/fees on your server
    # - Ceiling (2.4): Prevents 'hoping' for unrealistic moves
    final_tp = max(0.8, min(2.4, scaled_tp))

    # 4. Low-Vol Kill Switch
    # If Vol is too low (below 0.4), the strategy has no 'fuel' to hit targets
    is_active = current_vol_idx > 0.4

    return round(final_tp, 2) if is_active else None


# =============================
# CONFIG
# =============================

ERA_LIST = [
    20230901,
    20240301,
    20240901,
    20250301,
]

TIMEFRAME_MONTHS_LIST = [1, 3, 6]


# =============================
# DB
# =============================

def build_db_uri_from_env() -> str:
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_DB", "airflow")
    host = os.getenv("POSTGRES_HOST")

    if not host:
        host = "localhost" if os.name == "nt" else "postgres"

    if not user or not password:
        raise RuntimeError("POSTGRES_USER / POSTGRES_PASSWORD not set")

    return f"postgresql+psycopg2://{user}:{password}@{host}/{db}"


def get_engine_from_env(application_name: str = "metrics_era") -> Engine:
    uri = build_db_uri_from_env()
    sep = "&" if "?" in uri else "?"
    uri = f"{uri}{sep}application_name={application_name}"

    return create_engine(
        uri,
        pool_pre_ping=True,
        pool_size=3,
        max_overflow=5,
        pool_recycle=1800,
    )


# =============================
# TIME HELPERS (NS ONLY)
# =============================

def era_int_to_ns_bounds(era_int: int) -> tuple[int, int]:
    """
    era_int = YYYYMMDD
    Returns [start_ns, end_ns] for that UTC day.
    """
    start = pd.Timestamp(str(era_int), tz="UTC")
    end = start + pd.Timedelta(days=1)
    return int(start.value), int(end.value) - 1


def months_to_ns(months: int) -> int:
    """
    Approximate month offset using 30-day intervals.
    """
    return int(months * 30 * 24 * 60 * 60 * 1_000_000_000)


def estimate_5m_rows(months: int) -> int:
    """
    Rough forward-window row estimate for 5-minute BTC data.
    30 days/month * 24h/day * 12 bars/hour = 8640 bars/month.
    """
    return months * 30 * 24 * 12


# =============================
# REGIME AMPLITUDE
# =============================

def regime_amp_index_from_min_max(
    min_price: Optional[float],
    max_price: Optional[float],
) -> Optional[float]:
    if min_price is None or max_price is None:
        return None
    if min_price <= 0:
        return None
    return (max_price - min_price) / min_price


# =============================
# DATA LOAD
# =============================

def load_df_main_for_eras(
    engine: Engine,
    pair: str,
    market_type: str,
    eras: Sequence[int],
    forward_months_list: Sequence[int],
) -> pl.DataFrame:
    if not eras:
        return pl.DataFrame()

    era_dates = pd.to_datetime([str(e) for e in eras], utc=True)
    min_era = era_dates.min()
    max_era = era_dates.max()
    max_forward_months = max(forward_months_list) if forward_months_list else 0

    start_ns = int(min_era.value)
    end_ns = int(
        (max_era + pd.DateOffset(months=max_forward_months) + pd.Timedelta(days=1)).value
    )

    query = text("""
        SELECT
            time_ns,
            close
        FROM df_main
        WHERE pair = :pair
          AND market_type = :market_type
          AND time_ns >= :start_ns
          AND time_ns <  :end_ns
        ORDER BY time_ns ASC
    """)

    with engine.connect() as conn:
        df_pd = pd.read_sql_query(
            query,
            conn,
            params={
                "pair": pair,
                "market_type": market_type,
                "start_ns": start_ns,
                "end_ns": end_ns,
            },
        )

    if df_pd.empty:
        return pl.DataFrame()

    return (
        pl.from_pandas(df_pd)
        .with_columns([
            pl.col("time_ns").cast(pl.Int64),
            pl.col("close").cast(pl.Float64),
        ])
        .sort("time_ns")
    )


# =============================
# ERA CALC
# =============================

def calc_era_regime_amp(
    df: pl.DataFrame,
    era_int: int,
    forward_months: int,
) -> Dict[str, Any]:

    empty_row = {
        "era_int": era_int,
        "era_close_time": None,
        "timeframe_months": forward_months,
        "window_min_close": None,
        "window_max_close": None,
        "regime_amp_index": None,
    }

    if df.is_empty():
        return empty_row

    era_start_ns, era_end_ns = era_int_to_ns_bounds(era_int)

    era_df = df.filter(
        (pl.col("time_ns") >= era_start_ns)
        & (pl.col("time_ns") <= era_end_ns)
    )

    if era_df.is_empty():
        return empty_row

    era_close_ns = era_df.select(pl.col("time_ns").max()).item()

    window_end_ns = era_close_ns + months_to_ns(forward_months)

    window_df = df.filter(
        (pl.col("time_ns") >= era_close_ns)
        & (pl.col("time_ns") < window_end_ns)
    )

    if window_df.is_empty():
        return empty_row | {
            "era_close_time": pd.to_datetime(era_close_ns, utc=True),
        }

    min_close = float(window_df.select(pl.col("close").min()).item())
    max_close = float(window_df.select(pl.col("close").max()).item())
    amp = regime_amp_index_from_min_max(min_close, max_close)

    return {
        "era_int": era_int,
        "era_close_time": pd.to_datetime(era_close_ns, utc=True),
        "timeframe_months": forward_months,
        "window_min_close": min_close,
        "window_max_close": max_close,
        "regime_amp_index": amp,
    }


# =============================
# ATR
# =============================

def calc_atr_from_df(
    df: pl.DataFrame,
    period: int = 14,
) -> Optional[float]:
    """
    Close-only ATR proxy.
    TR = abs(close_t - close_{t-1})
    """
    if df.height <= period:
        return None

    tr = (
        df
        .with_columns(
            (pl.col("close") - pl.col("close").shift(1)).abs().alias("tr")
        )
        .drop_nulls()
    )

    if tr.height <= period:
        return None

    return float(tr.select(pl.col("tr").mean()).item())


def calc_era_atr(
    df: pl.DataFrame,
    era_int: int,
    forward_months: int,
    atr_period: int = 14,
) -> Dict[str, Any]:

    empty_row = {
        "era_int": era_int,
        "era_close_time": None,
        "timeframe_months": forward_months,
        "atr_period": atr_period,
        "atr": None,
    }

    if df.is_empty():
        return empty_row

    era_start_ns, era_end_ns = era_int_to_ns_bounds(era_int)

    era_df = df.filter(
        (pl.col("time_ns") >= era_start_ns)
        & (pl.col("time_ns") <= era_end_ns)
    )

    if era_df.is_empty():
        return empty_row

    era_close_ns = era_df.select(pl.col("time_ns").max()).item()

    window_end_ns = era_close_ns + months_to_ns(forward_months)

    window_df = df.filter(
        (pl.col("time_ns") >= era_close_ns)
        & (pl.col("time_ns") < window_end_ns)
    )

    if window_df.is_empty():
        return empty_row | {
            "era_close_time": pd.to_datetime(era_close_ns, utc=True),
        }

    atr = calc_atr_from_df(window_df, period=atr_period)

    return {
        "era_int": era_int,
        "era_close_time": pd.to_datetime(era_close_ns, utc=True),
        "timeframe_months": forward_months,
        "atr_period": atr_period,
        "atr": atr,
    }


# =============================
# TABLE BUILD
# =============================

def build_regime_amp_era_table(
    engine: Engine,
    pair: str,
    market_type: str,
    eras: Sequence[int],
    forward_months_list: Sequence[int],
) -> pd.DataFrame:

    df = load_df_main_for_eras(
        engine=engine,
        pair=pair,
        market_type=market_type,
        eras=eras,
        forward_months_list=forward_months_list,
    )

    rows = []
    for era in eras:
        for months in forward_months_list:
            rows.append(calc_era_regime_amp(df, era, months))

    out = pd.DataFrame(rows)
    return out.sort_values(["era_int", "timeframe_months"]).reset_index(drop=True)


def build_atr_era_table(
    engine: Engine,
    pair: str,
    market_type: str,
    eras: Sequence[int],
    forward_months_list: Sequence[int],
    atr_period: int = 14,
) -> pd.DataFrame:

    df = load_df_main_for_eras(
        engine=engine,
        pair=pair,
        market_type=market_type,
        eras=eras,
        forward_months_list=forward_months_list,
    )

    rows = []
    for era in eras:
        for months in forward_months_list:
            rows.append(calc_era_atr(df, era, months, atr_period))

    out = pd.DataFrame(rows)
    return out.sort_values(["era_int", "timeframe_months"]).reset_index(drop=True)


# =============================
# MAIN
# =============================

def main():
    engine = get_engine_from_env()

    pair = os.getenv("PAIR", "XXBTZUSD")
    market_type = os.getenv("MARKET_TYPE", "spot")

    regime_table = build_regime_amp_era_table(
        engine=engine,
        pair=pair,
        market_type=market_type,
        eras=ERA_LIST,
        forward_months_list=TIMEFRAME_MONTHS_LIST,
    )

    atr_table = build_atr_era_table(
        engine=engine,
        pair=pair,
        market_type=market_type,
        eras=ERA_LIST,
        forward_months_list=TIMEFRAME_MONTHS_LIST,
        atr_period=14,
    )

    print("\n" + "=" * 110)
    print(f" REGIME AMP + ATR ERA CHECK | pair={pair} | market_type={market_type} ")
    print("=" * 110)
    print(
        " Forward window hint (5m bars): "
        + " | ".join(f"{m}mo≈{estimate_5m_rows(m)} rows" for m in TIMEFRAME_MONTHS_LIST)
    )

    if regime_table.empty and atr_table.empty:
        print("No data.")
        return

    print("\n" + "-" * 110)
    print(" REGIME AMPLITUDE ")
    print("-" * 110)

    if regime_table.empty:
        print("No regime amp data.")
    else:
        print(
            regime_table[
                [
                    "era_int",
                    "era_close_time",
                    "timeframe_months",
                    "window_min_close",
                    "window_max_close",
                    "regime_amp_index",
                ]
            ].to_string(index=False)
        )

    print("\n" + "-" * 110)
    print(" ATR ")
    print("-" * 110)

    if atr_table.empty:
        print("No ATR data.")
    else:
        print(
            atr_table[
                [
                    "era_int",
                    "era_close_time",
                    "timeframe_months",
                    "atr_period",
                    "atr",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()