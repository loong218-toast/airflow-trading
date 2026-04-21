from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# =============================
# PATH / ENV
# =============================

BASE_DIR = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")


# =============================
# CONFIG
# =============================

LOOKBACK_DAYS = 730


def _default_utc_window(days_back: int = LOOKBACK_DAYS) -> tuple[str, str]:
    end_dt = datetime.now(timezone.utc).replace(microsecond=0)
    start_dt = end_dt - timedelta(days=days_back)
    start_s = start_dt.isoformat().replace("+00:00", "Z")
    end_s = end_dt.isoformat().replace("+00:00", "Z")
    return start_s, end_s


GRID_START_DATE, GRID_END_DATE = _default_utc_window(LOOKBACK_DAYS)

# Change this one line to switch instrument.
INSTRUMENT = "BTC"  # BTC | UK100 | AUDJPY | XAUUSD

INSTRUMENT_CONFIG = {
    "BTC": {
        "pair": "BTC",
        "mt5_symbol": "BTCUSD",  # change to your broker's exact MT5 crypto symbol if needed
        "target_pct_list": [0.2, 0.6, 0.8],
        "sl_pct_list": [0.2, 0.6, 0.8],
        "horizon_hours_list": [2, 3, 4, 8],
    },
    "UK100": {
        "pair": "UK100",
        "mt5_symbol": "UK100",
        "target_pct_list": [0.07, 0.13, 0.18],
        "sl_pct_list": [0.09, 0.13],
        "horizon_hours_list": [2, 3, 4, 8],
    },
    "AUDJPY": {
        "pair": "AUDJPY",
        "mt5_symbol": "AUDJPY",
        "target_pct_list": [0.07, 0.13, 0.18],
        "sl_pct_list": [0.09, 0.13],
        "horizon_hours_list": [2, 3, 4, 8],
    },
    "XAUUSD": {
        "pair": "XAUUSD",
        "mt5_symbol": "XAUUSD",
        "target_pct_list": [0.1, 0.2, 0.3],
        "sl_pct_list": [0.1, 0.2],
        "horizon_hours_list": [2, 3, 4, 8],
    },
}

if INSTRUMENT not in INSTRUMENT_CONFIG:
    raise ValueError(f"Unknown INSTRUMENT={INSTRUMENT!r}. Add it to INSTRUMENT_CONFIG.")

CFG = INSTRUMENT_CONFIG[INSTRUMENT]
PAIR = CFG["pair"]
MT5_SYMBOL = CFG["mt5_symbol"]

TARGET_PCT_LIST = CFG["target_pct_list"]
SL_PCT_LIST = CFG["sl_pct_list"]
HORIZON_HOURS_LIST = CFG["horizon_hours_list"]

# MT5 connection settings from env (credentials / terminal connection only)
MT5_PATH = None  # optionally set to r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_TIMEOUT = int(os.getenv("MT5_TIMEOUT", "60000"))
MT5_PORTABLE = os.getenv("MT5_PORTABLE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}

MT5_CHUNK_DAYS = 90

OUTPUT_DIR = BASE_DIR / "data_lake" / "Saved_results" / "price_reversion_checks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MINUTE_NS = 60 * 1_000_000_000
HOUR_NS = 60 * MINUTE_NS
FIVE_MIN_NS = 5 * MINUTE_NS


# =============================
# TIME HELPERS
# =============================

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


def _safe_name(value: str) -> str:
    return (
        value.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


# =============================
# DATA LOAD - MT5
# =============================

def _init_mt5_from_env():
    import MetaTrader5 as mt5

    init_kwargs: Dict[str, Any] = {
        "timeout": MT5_TIMEOUT,
        "portable": MT5_PORTABLE,
    }

    if MT5_PATH:
        init_kwargs["path"] = MT5_PATH

    if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
        init_kwargs["login"] = int(MT5_LOGIN)
        init_kwargs["password"] = MT5_PASSWORD
        init_kwargs["server"] = MT5_SERVER

    ok = mt5.initialize(**init_kwargs)
    if not ok:
        err = mt5.last_error()
        raise RuntimeError(f"MT5 initialize() failed: {err}")

    return mt5


def _fetch_mt5_rates_range_chunked(
    mt5,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    chunk_days: int = 90,
) -> pd.DataFrame:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"MT5 symbol_select({symbol!r}) failed: {mt5.last_error()}")

    chunks: List[pd.DataFrame] = []
    cur = start_dt

    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=chunk_days), end_dt)

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, cur, chunk_end)
        if rates is not None and len(rates) > 0:
            chunks.append(pd.DataFrame(rates))

        cur = chunk_end

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    if "time" not in df.columns:
        return pd.DataFrame()

    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    # MT5 timestamps are UTC seconds since epoch.
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ns"] = df["time"].astype("int64")

    for col in ("high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    needed = ["time_ns", "high", "low", "close"]
    for col in needed:
        if col not in df.columns:
            return pd.DataFrame()

    df = df.dropna(subset=needed).reset_index(drop=True)
    return df[needed]


def load_df_mt5_for_grid(
    symbol: str,
    grid_start_date: str,
    grid_end_date: str,
    chunk_days: int = 90,
) -> pl.DataFrame:
    start_dt = _parse_utc_dt(grid_start_date)
    end_dt = _parse_utc_dt(grid_end_date)

    if start_dt is None or end_dt is None:
        return pl.DataFrame()

    mt5 = _init_mt5_from_env()
    try:
        df_pd = _fetch_mt5_rates_range_chunked(
            mt5=mt5,
            symbol=symbol,
            start_dt=start_dt,
            end_dt=end_dt,
            chunk_days=chunk_days,
        )
    finally:
        mt5.shutdown()

    if df_pd.empty:
        return pl.DataFrame()

    return (
        pl.from_pandas(df_pd)
        .with_columns(
            [
                pl.col("time_ns").cast(pl.Int64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
            ]
        )
        .sort("time_ns")
    )


# =============================
# HELPERS
# =============================

def _mean_median(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr))


def _pct_from_ratio(r: Optional[float]) -> Optional[float]:
    if r is None:
        return None
    return float(r * 100.0)


def _net_expectancy_pct(
    target_first_rate_pct: float,
    target_pct: float,
    sl_first_rate_pct: float,
    sl_pct: float,
) -> float:
    tp_r = target_first_rate_pct / 100.0
    sl_r = sl_first_rate_pct / 100.0
    return (tp_r * target_pct) - (sl_r * sl_pct)


def _segment_ends_from_time_ns(time_ns: np.ndarray, bar_ns: int = FIVE_MIN_NS) -> np.ndarray:
    n = int(time_ns.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.int64)

    diffs = np.diff(time_ns)
    breaks = np.flatnonzero(diffs != bar_ns) + 1

    starts = np.r_[0, breaks]
    ends = np.r_[breaks, n]

    seg_end = np.empty(n, dtype=np.int64)
    for s, e in zip(starts, ends):
        seg_end[s:e] = e
    return seg_end


def make_output_file(instrument: str, pair: str, symbol: str) -> Path:
    safe_instrument = _safe_name(instrument)
    safe_pair = _safe_name(pair)
    safe_symbol = _safe_name(symbol)
    return OUTPUT_DIR / f"price_reversion_check_{safe_instrument}_{safe_pair}_{safe_symbol}.csv"


# =============================
# ANALYSIS
# =============================

def analyze_target_sl_survival(
    df: pl.DataFrame,
    target_pct_list: Sequence[float],
    sl_pct_list: Sequence[float],
    horizon_hours_list: Sequence[int],
    instrument: str,
    source_symbol: str,
    pair: str,
    conservative_sl_first: bool = True,
) -> pd.DataFrame:
    if df.is_empty():
        return pd.DataFrame()

    time_ns = df.get_column("time_ns").to_numpy()
    high = df.get_column("high").to_numpy()
    low = df.get_column("low").to_numpy()
    close = df.get_column("close").to_numpy()

    n = int(close.shape[0])
    if n < 2:
        return pd.DataFrame()

    horizons = sorted({int(h) for h in horizon_hours_list if int(h) > 0})
    if not horizons:
        return pd.DataFrame()

    target_list = sorted({float(x) for x in target_pct_list if float(x) > 0.0})
    sl_list = sorted({float(x) for x in sl_pct_list if float(x) > 0.0})

    horizon_end_by_h: Dict[int, np.ndarray] = {}
    for h in horizons:
        cutoff_ns = time_ns + np.int64(h * HOUR_NS)
        horizon_end_by_h[h] = np.searchsorted(time_ns, cutoff_ns, side="right")

    segment_end_exclusive = _segment_ends_from_time_ns(time_ns, FIVE_MIN_NS)

    rows: List[Dict[str, Any]] = []
    anchors_total = (n - 1) * 2  # long + short per anchor

    for target_pct in target_list:
        target_ratio = float(target_pct) / 100.0

        for sl_pct in sl_list:
            sl_ratio = float(sl_pct) / 100.0

            stats_by_h: Dict[int, Dict[str, Any]] = {
                h: {
                    "target_minutes": [],
                    "sl_minutes": [],
                    "first_event_minutes": [],
                    "target_first_count": 0,
                    "sl_first_count": 0,
                    "censored_count": 0,
                    "target_first_then_sl_count": 0,
                }
                for h in horizons
            }

            for anchor_idx in range(n - 1):
                anchor_price = float(close[anchor_idx])
                if not np.isfinite(anchor_price) or anchor_price <= 0.0:
                    continue

                anchor_time = int(time_ns[anchor_idx])
                seg_end_abs = int(segment_end_exclusive[anchor_idx])
                if seg_end_abs <= anchor_idx + 1:
                    continue

                future_high_full = high[anchor_idx + 1:seg_end_abs]
                future_low_full = low[anchor_idx + 1:seg_end_abs]
                future_time_full = time_ns[anchor_idx + 1:seg_end_abs]

                for side in (1, -1):
                    if side == 1:
                        target_price = anchor_price * (1.0 + target_ratio)
                        sl_price = anchor_price * (1.0 - sl_ratio)
                        target_hits_full = np.flatnonzero(future_high_full >= target_price)
                        sl_hits_full = np.flatnonzero(future_low_full <= sl_price)
                    else:
                        target_price = anchor_price * (1.0 - target_ratio)
                        sl_price = anchor_price * (1.0 + sl_ratio)
                        target_hits_full = np.flatnonzero(future_low_full <= target_price)
                        sl_hits_full = np.flatnonzero(future_high_full >= sl_price)

                    target_rel_full = int(target_hits_full[0]) if target_hits_full.size else -1
                    sl_rel_full = int(sl_hits_full[0]) if sl_hits_full.size else -1

                    for h in horizons:
                        horizon_end_abs = int(horizon_end_by_h[h][anchor_idx])
                        if horizon_end_abs <= anchor_idx + 1:
                            continue

                        # Do not bridge gaps / session breaks.
                        horizon_end_abs = min(horizon_end_abs, seg_end_abs)

                        local_len = horizon_end_abs - (anchor_idx + 1)
                        if local_len <= 0:
                            continue

                        target_in = target_rel_full != -1 and target_rel_full < local_len
                        sl_in = sl_rel_full != -1 and sl_rel_full < local_len

                        if not target_in and not sl_in:
                            stats_by_h[h]["censored_count"] += 1
                            continue

                        if target_in and sl_in:
                            if target_rel_full < sl_rel_full:
                                target_first = True
                            elif sl_rel_full < target_rel_full:
                                target_first = False
                            else:
                                target_first = not conservative_sl_first
                        elif target_in:
                            target_first = True
                        else:
                            target_first = False

                        if target_first:
                            stats_by_h[h]["target_first_count"] += 1
                            target_min = (int(future_time_full[target_rel_full]) - anchor_time) / MINUTE_NS
                            stats_by_h[h]["target_minutes"].append(target_min)
                            stats_by_h[h]["first_event_minutes"].append(target_min)

                            if sl_in and sl_rel_full > target_rel_full:
                                stats_by_h[h]["target_first_then_sl_count"] += 1
                        else:
                            stats_by_h[h]["sl_first_count"] += 1
                            sl_min = (int(future_time_full[sl_rel_full]) - anchor_time) / MINUTE_NS
                            stats_by_h[h]["sl_minutes"].append(sl_min)
                            stats_by_h[h]["first_event_minutes"].append(sl_min)

            for h in horizons:
                s = stats_by_h[h]
                target_first_count = int(s["target_first_count"])
                sl_first_count = int(s["sl_first_count"])
                censored_count = int(s["censored_count"])
                resolved_count = target_first_count + sl_first_count

                avg_target, med_target = _mean_median(s["target_minutes"])
                avg_sl, med_sl = _mean_median(s["sl_minutes"])
                avg_first, med_first = _mean_median(s["first_event_minutes"])

                target_first_rate_pct = _pct_from_ratio(target_first_count / anchors_total if anchors_total else None)
                sl_first_rate_pct = _pct_from_ratio(sl_first_count / anchors_total if anchors_total else None)
                target_first_then_sl_rate_pct = _pct_from_ratio(
                    s["target_first_then_sl_count"] / target_first_count if target_first_count else None
                )
                resolved_rate_pct = _pct_from_ratio(resolved_count / anchors_total if anchors_total else None)
                censored_rate_pct = _pct_from_ratio(censored_count / anchors_total if anchors_total else None)
                target_given_resolved_rate_pct = _pct_from_ratio(
                    target_first_count / resolved_count if resolved_count else None
                )

                rows.append(
                    {
                        "instrument": instrument,
                        "symbol": source_symbol,
                        "pair": pair,
                        "target_pct": float(target_pct),
                        "sl_pct": float(sl_pct),
                        "horizon_hours": int(h),
                        "anchors_total": anchors_total,
                        "target_first_count": target_first_count,
                        "sl_first_count": sl_first_count,
                        "target_first_then_sl_count": int(s["target_first_then_sl_count"]),
                        "resolved_count": resolved_count,
                        "censored_count": censored_count,
                        "target_first_rate_pct": target_first_rate_pct,
                        "sl_first_rate_pct": sl_first_rate_pct,
                        "target_first_then_sl_rate_pct": target_first_then_sl_rate_pct,
                        "resolved_rate_pct": resolved_rate_pct,
                        "censored_rate_pct": censored_rate_pct,
                        "target_given_resolved_rate_pct": target_given_resolved_rate_pct,
                        "net_expectancy_pct": _net_expectancy_pct(
                            target_first_rate_pct or 0.0,
                            float(target_pct),
                            sl_first_rate_pct or 0.0,
                            float(sl_pct),
                        ),
                        "avg_minutes_to_target": avg_target,
                        "median_minutes_to_target": med_target,
                        "avg_minutes_to_sl": avg_sl,
                        "median_minutes_to_sl": med_sl,
                        "avg_minutes_to_first_event": avg_first,
                        "median_minutes_to_first_event": med_first,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["instrument", "target_pct", "sl_pct", "horizon_hours"]).reset_index(drop=True)


# =============================
# MAIN
# =============================

def main() -> None:
    source_symbol = MT5_SYMBOL

    df = load_df_mt5_for_grid(
        symbol=source_symbol,
        grid_start_date=GRID_START_DATE,
        grid_end_date=GRID_END_DATE,
        chunk_days=MT5_CHUNK_DAYS,
    )

    if df.is_empty():
        print("No data returned for the requested grid.")
        return

    summary_df = analyze_target_sl_survival(
        df=df,
        target_pct_list=TARGET_PCT_LIST,
        sl_pct_list=SL_PCT_LIST,
        horizon_hours_list=HORIZON_HOURS_LIST,
        instrument=INSTRUMENT,
        source_symbol=source_symbol,
        pair=PAIR,
    )

    summary_df = summary_df.round(4)

    output_file = make_output_file(INSTRUMENT, PAIR, source_symbol)

    print("\n" + "=" * 150)
    print(f" TARGET vs SL SURVIVAL SCAN | instrument={INSTRUMENT} | symbol={source_symbol} | pair={PAIR}")
    print("=" * 150)
    print(f"Grid start: {GRID_START_DATE}")
    print(f"Grid end  : {GRID_END_DATE}")
    print(f"Rows loaded: {df.height:,}")
    print(f"Target Pct : {', '.join(f'{x:.1f}%' for x in TARGET_PCT_LIST)}")
    print(f"SL Pcts    : {', '.join(f'{x:.1f}%' for x in SL_PCT_LIST)}")
    print(f"Horizons   : {', '.join(f'{h}h' for h in HORIZON_HOURS_LIST)}")
    print(f"MT5 symbol : {source_symbol}")

    if summary_df.empty:
        print("No summary produced.")
        return

    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary CSV to: {output_file}")

    print("\n" + "-" * 150)
    print(" SUMMARY ")
    print("-" * 150)
    with pd.option_context("display.max_columns", None, "display.width", 300):
        print(
            summary_df[
                [
                    "instrument",
                    "symbol",
                    "pair",
                    "target_pct",
                    "sl_pct",
                    "horizon_hours",
                    "anchors_total",
                    "target_first_count",
                    "sl_first_count",
                    "target_first_then_sl_count",
                    "resolved_count",
                    "censored_count",
                    "target_first_rate_pct",
                    "sl_first_rate_pct",
                    "target_first_then_sl_rate_pct",
                    "resolved_rate_pct",
                    "censored_rate_pct",
                    "target_given_resolved_rate_pct",
                    "net_expectancy_pct",
                    "avg_minutes_to_target",
                    "median_minutes_to_target",
                    "avg_minutes_to_sl",
                    "median_minutes_to_sl",
                    "avg_minutes_to_first_event",
                    "median_minutes_to_first_event",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
