# research/metrics_era.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Dict, Any, Optional, List, Tuple

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

GRID_START_DATE = "2022-09-01T00:00:00Z"
GRID_END_DATE = "2025-09-15T23:59:59Z"

TIMEFRAME_MONTHS_LIST = [3]

OUTPUT_DIR = Path("/opt/airflow/airflow-trading/data_lake/Saved_results")
OUTPUT_FILE = OUTPUT_DIR / "amp_data.py"


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

def _parse_utc_dt(dt_in: Any) -> Optional[datetime]:
    if dt_in is None:
        return None

    if isinstance(dt_in, datetime):
        dt = dt_in
    elif isinstance(dt_in, str):
        s = dt_in[:-1] if dt_in.endswith("Z") else dt_in
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            try:
                dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
            except Exception:
                return None
    else:
        try:
            dt = pd.Timestamp(dt_in).to_pydatetime()
        except Exception:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt


def _dt_to_ns(dt_in: Any) -> Optional[int]:
    dt = _parse_utc_dt(dt_in)
    if dt is None:
        return None
    return int(pd.Timestamp(dt).value)

def split_period_windows_from_ns_df(
    df: pl.DataFrame,
    months: int,
    min_dt: Optional[datetime] = None,
    max_dt: Optional[datetime] = None,
    include_partial_tail: bool = False,
) -> List[Tuple[datetime, datetime]]:
    if months <= 0:
        return []

    if (min_dt is None or max_dt is None) and (df is None or df.height == 0):
        return []

    if min_dt is None or max_dt is None:
        try:
            first_ns = int(df[0, "time_ns"])
            last_ns = int(df[-1, "time_ns"])
            min_dt = min_dt or pd.to_datetime(first_ns, utc=True).to_pydatetime()
            max_dt = max_dt or pd.to_datetime(last_ns, utc=True).to_pydatetime()
        except Exception:
            return []

    min_dt = _parse_utc_dt(min_dt)
    max_dt = _parse_utc_dt(max_dt)
    if min_dt is None or max_dt is None:
        return []

    start = datetime(min_dt.year, min_dt.month, 1, tzinfo=timezone.utc)
    windows: List[Tuple[datetime, datetime]] = []
    cur = start

    while True:
        m = cur.month - 1 + int(months)
        y = cur.year + (m // 12)
        mm = (m % 12) + 1
        end = datetime(year=y, month=mm, day=1, tzinfo=timezone.utc)

        if end > max_dt:
            if include_partial_tail and cur < max_dt:
                windows.append((cur, max_dt))
            break

        windows.append((cur, end))
        cur = end

    return windows


def estimate_5m_rows(months: int) -> int:
    """
    Rough forward-window row estimate for 5-minute BTC data.
    30 days/month * 24h/day * 12 bars/hour = 8640 bars/month.
    """
    return months * 30 * 24 * 12

def _to_python_literal_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        if pd.isna(v):
            return None
        return v.to_pydatetime().isoformat(sep=" ")
    if isinstance(v, datetime):
        return v.isoformat(sep=" ")
    if pd.isna(v) if hasattr(pd, "isna") else False:
        return None
    return v


def write_amp_data_py(rows: List[Dict[str, Any]], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cleaned_rows: List[Dict[str, Any]] = []
    for row in rows:
        cleaned = {k: _to_python_literal_value(v) for k, v in row.items()}
        cleaned_rows.append(cleaned)

    content = "amp_data = " + repr(cleaned_rows) + "\n"
    out_file.write_text(content, encoding="utf-8")

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

def load_df_main_for_grid(
    engine: Engine,
    pair: str,
    market_type: str,
    grid_start_date: str,
    grid_end_date: str,
) -> pl.DataFrame:
    start_ns = _dt_to_ns(grid_start_date)
    end_ns = _dt_to_ns(grid_end_date)

    if start_ns is None or end_ns is None:
        return pl.DataFrame()

    query = text("""
        SELECT
            time_ns,
            close
        FROM df_main
        WHERE pair = :pair
          AND market_type = :market_type
          AND time_ns >= :start_ns
          AND time_ns <= :end_ns
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
# WINDOW CALC
# =============================

def calc_window_regime_amp(
    df: pl.DataFrame,
    window_start: datetime,
    window_end: datetime,
    timeframe_months: int,
) -> Dict[str, Any]:
    period_id = int(window_start.strftime("%Y%m%d"))

    empty_row = {
        "era_int": period_id,
        "era_close_time": window_end,
        "window_start_time": window_start,
        "window_end_time": window_end,
        "timeframe_months": timeframe_months,
        "window_rows": 0,
        "window_min_close": None,
        "window_max_close": None,
        "regime_amp_index": None,
    }

    if df.is_empty():
        return empty_row

    start_ns = int(pd.Timestamp(window_start).value)
    end_ns = int(pd.Timestamp(window_end).value)

    window_df = df.filter(
        (pl.col("time_ns") >= start_ns) &
        (pl.col("time_ns") < end_ns)
    )

    if window_df.is_empty():
        return empty_row

    min_close = float(window_df.select(pl.col("close").min()).item())
    max_close = float(window_df.select(pl.col("close").max()).item())
    amp = regime_amp_index_from_min_max(min_close, max_close)

    return {
        "era_int": period_id,
        "era_close_time": window_end,
        "window_start_time": window_start,
        "window_end_time": window_end,
        "timeframe_months": timeframe_months,
        "window_rows": window_df.height,
        "window_min_close": min_close,
        "window_max_close": max_close,
        "regime_amp_index": amp,
    }


def calc_window_atr(
    df: pl.DataFrame,
    window_start: datetime,
    window_end: datetime,
    timeframe_months: int,
    atr_period: int = 14,
) -> Dict[str, Any]:
    period_id = int(window_start.strftime("%Y%m%d"))

    empty_row = {
        "era_int": period_id,
        "era_close_time": window_end,
        "window_start_time": window_start,
        "window_end_time": window_end,
        "timeframe_months": timeframe_months,
        "atr_period": atr_period,
        "atr": None,
    }

    if df.is_empty():
        return empty_row

    start_ns = int(pd.Timestamp(window_start).value)
    end_ns = int(pd.Timestamp(window_end).value)

    window_df = df.filter(
        (pl.col("time_ns") >= start_ns) &
        (pl.col("time_ns") < end_ns)
    )

    if window_df.is_empty():
        return empty_row

    atr = calc_atr_from_df(window_df, period=atr_period)

    return {
        "era_int": period_id,
        "era_close_time": window_end,
        "window_start_time": window_start,
        "window_end_time": window_end,
        "timeframe_months": timeframe_months,
        "atr_period": atr_period,
        "atr": atr,
    }


# =============================
# ATR
# =============================

def calc_atr_from_df(
    df: pl.DataFrame,
    period: int = 14,
) -> Optional[float]:
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


# =============================
# TABLE BUILD
# =============================

def build_regime_amp_grid_table(
    engine: Engine,
    pair: str,
    market_type: str,
    grid_start_date: str,
    grid_end_date: str,
    forward_months_list: Sequence[int],
) -> pd.DataFrame:
    df = load_df_main_for_grid(
        engine=engine,
        pair=pair,
        market_type=market_type,
        grid_start_date=grid_start_date,
        grid_end_date=grid_end_date,
    )

    grid_start_dt = _parse_utc_dt(grid_start_date)
    grid_end_dt = _parse_utc_dt(grid_end_date)

    rows = []
    for months in forward_months_list:
        windows = split_period_windows_from_ns_df(
            df=df,
            months=months,
            min_dt=grid_start_dt,
            max_dt=grid_end_dt,
            include_partial_tail=True,
        )
        for window_start, window_end in windows:
            rows.append(calc_window_regime_amp(df, window_start, window_end, months))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["era_int", "timeframe_months"]).reset_index(drop=True)


def build_atr_grid_table(
    engine: Engine,
    pair: str,
    market_type: str,
    grid_start_date: str,
    grid_end_date: str,
    forward_months_list: Sequence[int],
    atr_period: int = 14,
) -> pd.DataFrame:
    df = load_df_main_for_grid(
        engine=engine,
        pair=pair,
        market_type=market_type,
        grid_start_date=grid_start_date,
        grid_end_date=grid_end_date,
    )

    grid_start_dt = _parse_utc_dt(grid_start_date)
    grid_end_dt = _parse_utc_dt(grid_end_date)

    rows = []
    for months in forward_months_list:
        windows = split_period_windows_from_ns_df(
            df=df,
            months=months,
            min_dt=grid_start_dt,
            max_dt=grid_end_dt,
            include_partial_tail=True,
        )
        for window_start, window_end in windows:
            rows.append(calc_window_atr(df, window_start, window_end, months, atr_period))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["era_int", "timeframe_months"]).reset_index(drop=True)


# =============================
# MAIN
# =============================

def main():
    engine = get_engine_from_env()

    pair = os.getenv("PAIR", "XXBTZUSD")
    market_type = os.getenv("MARKET_TYPE", "spot")

    regime_table = build_regime_amp_grid_table(
        engine=engine,
        pair=pair,
        market_type=market_type,
        grid_start_date=GRID_START_DATE,
        grid_end_date=GRID_END_DATE,
        forward_months_list=TIMEFRAME_MONTHS_LIST,
    )

    atr_table = build_atr_grid_table(
        engine=engine,
        pair=pair,
        market_type=market_type,
        grid_start_date=GRID_START_DATE,
        grid_end_date=GRID_END_DATE,
        forward_months_list=TIMEFRAME_MONTHS_LIST,
        atr_period=14,
    )

    print("\n" + "=" * 130)
    print(f" GRID REGIME AMP + ATR CHECK | pair={pair} | market_type={market_type} ")
    print("=" * 130)
    print(
        " Forward window hint (5m bars): "
        + " | ".join(f"{m}mo≈{estimate_5m_rows(m)} rows" for m in TIMEFRAME_MONTHS_LIST)
    )

    if regime_table.empty and atr_table.empty:
        print("No data.")
        return

    if not regime_table.empty:
        write_amp_data_py(
            regime_table[
                [
                    "era_int",
                    "era_close_time",
                    "window_rows",
                    "window_min_close",
                    "window_max_close",
                    "regime_amp_index",
                ]
            ].to_dict(orient="records"),
            OUTPUT_FILE,
        )
        print(f"\nSaved amp_data file to: {OUTPUT_FILE}")

    print("\n" + "-" * 130)
    print(" REGIME AMPLITUDE ")
    print("-" * 130)

    if regime_table.empty:
        print("No regime amp data.")
    else:
        print(
            regime_table[
                [
                    "era_int",
                    "window_start_time",
                    "window_end_time",
                    "timeframe_months",
                    "window_rows",
                    "window_min_close",
                    "window_max_close",
                    "regime_amp_index",
                ]
            ].to_string(index=False)
        )

    print("\n" + "-" * 130)
    print(" ATR ")
    print("-" * 130)

    if atr_table.empty:
        print("No ATR data.")
    else:
        print(
            atr_table[
                [
                    "era_int",
                    "window_start_time",
                    "window_end_time",
                    "timeframe_months",
                    "atr_period",
                    "atr",
                ]
            ].to_string(index=False)
        )

if __name__ == "__main__":
    main()