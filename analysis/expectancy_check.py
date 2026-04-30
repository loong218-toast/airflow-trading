# expectancy_check.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import polars as pl

import pythoncom

try:
    pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
except pythoncom.com_error:
    pass

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# =============================
# PATH / ENV
# =============================

BASE_DIR = Path(__file__).resolve()
while BASE_DIR.name != "airflow-trading":
    BASE_DIR = BASE_DIR.parent

if load_dotenv is not None:
    load_dotenv()


# =============================
# CONFIG
# =============================

LOOKBACK_DAYS = 180
ENTRY_BUCKET_HOURS = 4
USE_ENTRY_BUCKET_HOURS = False
RISK_PCT = 0.005

USE_ROLLING_VOLATILITY_CHECK = False
VOL_LOOKBACK_DAYS = 7
VOL_LOW_CUTOFF = 0.33
VOL_HIGH_CUTOFF = 0.66

USE_MA_FILTER = True
MA_TYPE = "ema"          # "ema" or "sma"
MA_PERIOD_BARS = 32
MA_WARMUP_DAYS = 10

if 24 % ENTRY_BUCKET_HOURS != 0:
    raise ValueError("ENTRY_BUCKET_HOURS must divide 24 exactly.")

MYT = timezone(timedelta(hours=8), name="MYT")


def _default_utc_window(days_back: int = LOOKBACK_DAYS) -> tuple[str, str]:
    end_dt = datetime.now(timezone.utc).replace(microsecond=0)
    start_dt = end_dt - timedelta(days=days_back)
    return (
        start_dt.isoformat().replace("+00:00", "Z"),
        end_dt.isoformat().replace("+00:00", "Z"),
    )


GRID_START_DATE, GRID_END_DATE = _default_utc_window(LOOKBACK_DAYS)

INSTRUMENT = "UK100"

INSTRUMENT_CONFIG = {
    "BTC": {
        "pair": "BTC",
        "mt5_symbol": "BTCUSD",
        "tp_range": {"min": 0.3, "max": 2.4, "step": 0.1},
        "sl_range": {"min": 0.3, "max": 2.4, "step": 0.1},
        "horizon_hours_list": [24],
    },
    "UK100": {
        "pair": "UK100",
        "mt5_symbol": "UK100",
        "tp_range": {"min": 0.05, "max": 0.22, "step": 0.01},
        "sl_range": {"min": 0.05, "max": 0.22, "step": 0.01},
        "horizon_hours_list": [3],
    },
    "AUDJPY": {
        "pair": "AUDJPY",
        "mt5_symbol": "AUDJPY",
        "tp_range": {"min": 0.03, "max": 0.16, "step": 0.01},
        "sl_range": {"min": 0.03, "max": 0.16, "step": 0.01},
        "horizon_hours_list": [8],
    },
    "USDCHF": {
        "pair": "USDCHF",
        "mt5_symbol": "USDCHF",
        "tp_range": {"min": 0.25, "max": 0.25, "step": 0.05},
        "sl_range": {"min": 0.6, "max": 0.6, "step": 0.05},
        "horizon_hours_list": [8, 16, 24, 48, 72, 168],
    },
    "XAUUSD": {
        "pair": "XAUUSD",
        "mt5_symbol": "XAUUSD",
        "tp_range": {"min": 0.1, "max": 0.8, "step": 0.1},
        "sl_range": {"min": 0.1, "max": 0.8, "step": 0.1},
        "horizon_hours_list": [24],
    },
}

if INSTRUMENT not in INSTRUMENT_CONFIG:
    raise ValueError(f"Unknown INSTRUMENT={INSTRUMENT!r}. Add it to INSTRUMENT_CONFIG.")

CFG = INSTRUMENT_CONFIG[INSTRUMENT]
PAIR = CFG["pair"]
MT5_SYMBOL = CFG["mt5_symbol"]

RANDOMIZE_ENTRY_PRICE = True
RANDOM_ENTRY_SEED = 12121212
RANDOM_ENTRY_MODE = "hybrid"  # "full", "body", "wick", "hybrid"
RANDOM_ENTRY_BODY_PROB = 0.70

def _float_range_inclusive(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if stop < start:
        raise ValueError("stop must be >= start")

    values: List[float] = []
    cur = float(start)
    eps = step / 10_000.0

    while cur <= stop + eps:
        values.append(round(cur, 10))
        cur += step

    return values


TARGET_PCT_LIST = _float_range_inclusive(CFG["tp_range"]["min"], CFG["tp_range"]["max"], CFG["tp_range"]["step"])
SL_PCT_LIST = _float_range_inclusive(CFG["sl_range"]["min"], CFG["sl_range"]["max"], CFG["sl_range"]["step"])
HORIZON_HOURS_LIST = CFG["horizon_hours_list"]

MT5_PATH = None
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_TIMEOUT = int(os.getenv("MT5_TIMEOUT", "60000"))
MT5_PORTABLE = os.getenv("MT5_PORTABLE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}

MT5_CHUNK_DAYS = 90

OUTPUT_BASE_DIR = BASE_DIR / "data_lake" / "Saved_results" / "expectancy_checks"
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = OUTPUT_BASE_DIR / INSTRUMENT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS_TO_REMOVE = [
    "symbol",
    "pair",
    "ma_candidate_count",
    "ma_pass_count",
    "ma_pass_rate_pct",
    "avg_entry_vs_ma_pct",
    "median_entry_vs_ma_pct",
    "net_expectancy_risk_pct",
    "horizon_eligible_count",
    "horizon_eligible_rate_pct",
]

MINUTE_NS = 60 * 1_000_000_000
HOUR_NS = 60 * MINUTE_NS


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

    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _safe_name(value: str) -> str:
    return (
        value.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def _bucket_start_hour_from_time_ns_myt(time_ns: int, bucket_hours: int = ENTRY_BUCKET_HOURS) -> int:
    ts_utc = pd.Timestamp(int(time_ns), unit="ns", tz="UTC")
    ts_myt = ts_utc.tz_convert(MYT)
    return int((ts_myt.hour // bucket_hours) * bucket_hours)


def _bucket_label_from_start_hour(start_hour: int, bucket_hours: int = ENTRY_BUCKET_HOURS) -> str:
    end_hour = start_hour + bucket_hours
    return f"{start_hour:02d}:00-{end_hour:02d}:00 MYT"


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
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

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
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ns"] = df["time"].astype("int64")

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    needed = ["time_ns", "open", "high", "low", "close"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()

    return df.dropna(subset=needed).reset_index(drop=True)[needed]


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
                pl.col("open").cast(pl.Float64),
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


def _net_expectancy_risk_pct(
    target_first_rate_pct: float,
    target_pct: float,
    sl_first_rate_pct: float,
    sl_pct: float,
    risk_pct: float = RISK_PCT,
) -> float:
    if sl_pct <= 0:
        raise ValueError("sl_pct must be > 0")

    tp_rate = target_first_rate_pct / 100.0
    sl_rate = sl_first_rate_pct / 100.0
    rr_multiple = float(target_pct) / float(sl_pct)

    win_pct = risk_pct * rr_multiple * 100.0
    loss_pct = risk_pct * 100.0
    return (tp_rate * win_pct) - (sl_rate * loss_pct)


def _market_tag(instrument: str, pair: str, symbol: str) -> str:
    for value in (pair, instrument, symbol):
        if value:
            return _safe_name(str(value))
    return "market"


def make_output_file(instrument: str, pair: str, symbol: str, bucket_hours: Optional[int], use_bucket_hours: bool) -> Path:
    market = _market_tag(instrument, pair, symbol)
    bucket_part = f"{int(bucket_hours)}h_myt_buckets" if use_bucket_hours else "no_myt_buckets"
    return OUTPUT_DIR / f"expectancy_scan_{market}_{bucket_part}.csv"


def _heatmap_annot_size(pivot: pd.DataFrame) -> int:
    biggest = max(pivot.shape)
    if biggest <= 8:
        return 9
    if biggest <= 10:
        return 8
    if biggest <= 12:
        return 7
    if biggest <= 15:
        return 6
    if biggest <= 20:
        return 5
    return 4


# =============================
# VOLATILITY / MA HELPERS
# =============================

def _compute_true_range_np(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )


def _compute_vol_categories(
    time_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    if not USE_ROLLING_VOLATILITY_CHECK:
        return np.full(int(close.shape[0]), "all_vol", dtype=object)

    tr = _compute_true_range_np(high, low, close)
    idx = pd.to_datetime(time_ns, unit="ns", utc=True)
    s = pd.Series(tr, index=idx)

    roll = s.rolling(window=f"{int(VOL_LOOKBACK_DAYS)}D", closed="left")
    q_low = roll.quantile(VOL_LOW_CUTOFF).to_numpy(dtype=np.float64, copy=False)
    q_high = roll.quantile(VOL_HIGH_CUTOFF).to_numpy(dtype=np.float64, copy=False)

    out = np.full(len(close), "", dtype=object)
    warmup_cutoff = idx[0] + pd.Timedelta(days=VOL_LOOKBACK_DAYS)
    valid = idx >= warmup_cutoff
    tr_np = tr.astype(np.float64, copy=False)

    low_mask = valid & np.isfinite(q_low) & (tr_np <= q_low)
    med_mask = valid & np.isfinite(q_low) & np.isfinite(q_high) & (tr_np > q_low) & (tr_np <= q_high)
    high_mask = valid & np.isfinite(q_high) & (tr_np > q_high)

    out[low_mask] = "low_vol"
    out[med_mask] = "med_vol"
    out[high_mask] = "high_vol"
    return out


def _compute_ma_values(close: np.ndarray) -> np.ndarray:
    s = pd.Series(close, dtype="float64")
    if MA_TYPE.lower().strip() == "ema":
        ma = s.ewm(span=MA_PERIOD_BARS, adjust=False, min_periods=MA_PERIOD_BARS).mean()
    elif MA_TYPE.lower().strip() == "sma":
        ma = s.rolling(window=MA_PERIOD_BARS, min_periods=MA_PERIOD_BARS).mean()
    else:
        raise ValueError("MA_TYPE must be 'ema' or 'sma'")
    return ma.shift(1).to_numpy(dtype=np.float64, copy=False)


def _ma_pass(side: int, price: float, ma_value: float) -> bool:
    if not np.isfinite(ma_value):
        return False
    return (price < ma_value) if side == 1 else (price > ma_value)

def _rng_for_anchor(anchor_idx: int) -> np.random.Generator:
    return np.random.default_rng(RANDOM_ENTRY_SEED + (anchor_idx + 1) * 1_000_003)


def _sample_entry_price(
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    anchor_idx: int,
) -> float:
    if not all(np.isfinite(x) for x in (open_px, high_px, low_px, close_px)):
        return np.nan

    lo_candle = float(low_px)
    hi_candle = float(high_px)
    body_lo = min(float(open_px), float(close_px))
    body_hi = max(float(open_px), float(close_px))

    if hi_candle < lo_candle:
        lo_candle, hi_candle = hi_candle, lo_candle

    rng = _rng_for_anchor(anchor_idx)
    mode = RANDOM_ENTRY_MODE.lower().strip()

    if mode == "body":
        lo, hi = body_lo, body_hi

    elif mode == "wick":
        wick_ranges: List[Tuple[float, float]] = []
        if lo_candle < body_lo:
            wick_ranges.append((lo_candle, body_lo))
        if body_hi < hi_candle:
            wick_ranges.append((body_hi, hi_candle))

        if not wick_ranges:
            lo, hi = lo_candle, hi_candle
        else:
            lengths = np.array([max(0.0, hi - lo) for lo, hi in wick_ranges], dtype=np.float64)
            total = float(lengths.sum())
            if total <= 0.0:
                lo, hi = lo_candle, hi_candle
            else:
                pick = rng.choice(len(wick_ranges), p=lengths / total)
                lo, hi = wick_ranges[int(pick)]

    elif mode == "hybrid":
        use_body = rng.random() < RANDOM_ENTRY_BODY_PROB and body_hi > body_lo
        if use_body:
            lo, hi = body_lo, body_hi
        else:
            lo, hi = lo_candle, hi_candle

    else:  # "full"
        lo, hi = lo_candle, hi_candle

    if hi < lo:
        lo, hi = hi, lo

    if hi == lo:
        return float(lo)

    return float(rng.uniform(lo, hi))

# =============================
# PLOTS
# =============================

def save_net_expectancy_tp_sl_plot(
    summary_df: pd.DataFrame,
    instrument: str,
    pair: str,
    symbol: str,
    include_entry_bucket_hours: bool,
) -> List[Path]:
    if summary_df.empty:
        return []
    if sns is None:
        raise ImportError("seaborn is required for this plot function, but it is not installed.")

    market = _market_tag(instrument, pair, symbol)
    saved_files: List[Path] = []

    group_cols: List[str] = []
    if "filter_mode" in summary_df.columns and summary_df["filter_mode"].nunique(dropna=True) > 1:
        group_cols.append("filter_mode")
    if "vol_category" in summary_df.columns and summary_df["vol_category"].nunique(dropna=True) > 1:
        group_cols.append("vol_category")
    if include_entry_bucket_hours and "entry_bucket_label" in summary_df.columns:
        group_cols.append("entry_bucket_label")

    groups = [(k, g.copy()) for k, g in summary_df.groupby(group_cols, sort=True)] if group_cols else [("ALL", summary_df.copy())]

    for key, group in groups:
        horizons = sorted(group["horizon_hours"].dropna().unique().tolist())
        if not horizons:
            continue

        vmin = float(group["net_expectancy_pct"].min())
        vmax = float(group["net_expectancy_pct"].max())
        label = _safe_name(_bucket_label_from_start_hour(0) if not group_cols else _safe_name(str(key)))

        fig, axes = plt.subplots(1, len(horizons), figsize=(7.5 * len(horizons), 6.5), squeeze=False, constrained_layout=True)
        axes_row = axes[0]
        first_mappable = None

        for ax, h in zip(axes_row, horizons):
            sub = group[group["horizon_hours"] == h].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            pivot = sub.pivot_table(
                index="sl_pct",
                columns="target_pct",
                values="net_expectancy_pct",
                aggfunc="mean",
            ).sort_index(ascending=False)

            hm = sns.heatmap(
                pivot,
                ax=ax,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                annot=False,
                linewidths=0.5,
                linecolor="white",
                cbar=False,
                square=True,
            )

            if first_mappable is None and hm.collections:
                first_mappable = hm.collections[0]

            title_bits = [market]
            if group_cols:
                title_bits.append(str(key))
            title_bits.append(f"{int(h)}h")
            ax.set_title(" | ".join(title_bits), fontsize=12, pad=10)
            ax.set_xlabel("Target %")
            ax.set_ylabel("SL %")
            ax.tick_params(axis="x", labelrotation=0, labelsize=8)
            ax.tick_params(axis="y", labelrotation=0, labelsize=8)

        if first_mappable is not None:
            fig.colorbar(first_mappable, ax=axes_row.tolist(), shrink=0.9, pad=0.02, label="Net expectancy (%)")

        out_file = OUTPUT_DIR / f"expectancy_heatmap_{market}_{label}.png"
        fig.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(out_file)

    return saved_files


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
    time_bucket_hours: int = ENTRY_BUCKET_HOURS,
    include_entry_bucket_hours: bool = USE_ENTRY_BUCKET_HOURS,
    conservative_sl_first: bool = True,
    use_ma_filter: bool = USE_MA_FILTER,
) -> pd.DataFrame:
    if df.is_empty():
        return pd.DataFrame()

    if include_entry_bucket_hours and 24 % time_bucket_hours != 0:
        raise ValueError("time_bucket_hours must divide 24 exactly.")

    time_ns = df.get_column("time_ns").to_numpy()
    high = df.get_column("high").to_numpy()
    low = df.get_column("low").to_numpy()
    close = df.get_column("close").to_numpy()

    open_px = df.get_column("open").to_numpy()

    n = int(close.shape[0])
    if n < 2:
        return pd.DataFrame()

    horizons = sorted({int(h) for h in horizon_hours_list if int(h) > 0})
    target_list = sorted({float(x) for x in target_pct_list if float(x) > 0.0})
    sl_list = sorted({float(x) for x in sl_pct_list if float(x) > 0.0})
    vol_category_by_idx = _compute_vol_categories(time_ns, high, low, close)
    ma_values = _compute_ma_values(close)

    if USE_ROLLING_VOLATILITY_CHECK:
        vol_categories = ["low_vol", "med_vol", "high_vol"]
        print(f"Volatility check enabled: {VOL_LOOKBACK_DAYS}D rolling TR percentile")
        print(f"Volatility warmup skipped bars: {int(np.sum(vol_category_by_idx == '')):,}")
    else:
        vol_categories = ["all_vol"]

    horizon_end_by_h: Dict[int, np.ndarray] = {}
    for h in horizons:
        horizon_end_by_h[h] = np.searchsorted(time_ns, time_ns + np.int64(h * HOUR_NS), side="right")

    bucket_starts = list(range(0, 24, time_bucket_hours)) if include_entry_bucket_hours else [0]
    filter_modes = [("all_trades", False)] + ([("ma_filter", True)] if use_ma_filter else [])

    rows: List[Dict[str, Any]] = []

    for filter_mode, apply_ma_filter in filter_modes:
        for target_pct in target_list:
            target_ratio = float(target_pct) / 100.0

            for sl_pct in sl_list:
                sl_ratio = float(sl_pct) / 100.0

                stats_by_vol: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]] = {
                    vol_cat: {
                        b: {
                            h: {
                                "target_minutes": [],
                                "sl_minutes": [],
                                "first_event_minutes": [],
                                "horizon_exit_r": [],
                                "forced_exit_r_all": [],
                                "mae_r": [],
                                "mfe_r": [],
                                "ma_candidate_count": 0,
                                "ma_pass_count": 0,
                                "entry_vs_ma_pct": [],
                                "horizon_eligible_count": 0,
                                "target_first_count": 0,
                                "sl_first_count": 0,
                                "censored_count": 0,
                                "target_first_then_sl_count": 0,
                                "anchors_total": 0,
                            }
                            for h in horizons
                        }
                        for b in bucket_starts
                    }
                    for vol_cat in vol_categories
                }

                for anchor_idx in range(n - 1):
                    if RANDOMIZE_ENTRY_PRICE:
                        anchor_price = _sample_entry_price(
                            open_px=float(open_px[anchor_idx]),
                            high_px=float(high[anchor_idx]),
                            low_px=float(low[anchor_idx]),
                            close_px=float(close[anchor_idx]),
                            anchor_idx=anchor_idx,
                        )
                    else:
                        anchor_price = float(close[anchor_idx])
                    if not np.isfinite(anchor_price) or anchor_price <= 0.0:
                        continue

                    vol_category = vol_category_by_idx[anchor_idx]
                    if vol_category == "" or vol_category not in stats_by_vol:
                        continue

                    ma_value = float(ma_values[anchor_idx]) if anchor_idx < len(ma_values) else np.nan
                    entry_vs_ma_pct = ((anchor_price - ma_value) / ma_value) * 100.0 if np.isfinite(ma_value) and ma_value != 0 else np.nan

                    bucket_start_hour = (
                        _bucket_start_hour_from_time_ns_myt(int(time_ns[anchor_idx]), time_bucket_hours)
                        if include_entry_bucket_hours
                        else 0
                    )

                    future_high_full = high[anchor_idx + 1:]
                    future_low_full = low[anchor_idx + 1:]
                    future_time_full = time_ns[anchor_idx + 1:]

                    for side in (1, -1):
                        ma_ok = _ma_pass(side, anchor_price, ma_value)

                        for h in horizons:
                            s = stats_by_vol[vol_category][bucket_start_hour][h]
                            # candidate seen (optional, diagnostic only)
                            s["ma_candidate_count"] += 1

                            if apply_ma_filter and not ma_ok:
                                continue

                            # NOW this trade exists
                            s["anchors_total"] += 1
                            if np.isfinite(entry_vs_ma_pct):
                                s["entry_vs_ma_pct"].append(float(entry_vs_ma_pct))
                            if ma_ok:
                                s["ma_pass_count"] += 1

                            if apply_ma_filter and not ma_ok:
                                continue

                            if side == 1:
                                target_price = anchor_price * (1.0 + target_ratio)
                                sl_price = anchor_price * (1.0 - sl_ratio)
                            else:
                                target_price = anchor_price * (1.0 - target_ratio)
                                sl_price = anchor_price * (1.0 + sl_ratio)

                            horizon_end_abs = int(horizon_end_by_h[h][anchor_idx])
                            if horizon_end_abs <= anchor_idx + 1:
                                continue

                            local_len = horizon_end_abs - (anchor_idx + 1)
                            if local_len <= 0:
                                continue

                            s["horizon_eligible_count"] += 1
                            future_high_local = future_high_full[:local_len]
                            future_low_local = future_low_full[:local_len]
                            future_time_local = future_time_full[:local_len]

                            if side == 1:
                                target_hits = np.flatnonzero(future_high_local >= target_price)
                                sl_hits = np.flatnonzero(future_low_local <= sl_price)
                            else:
                                target_hits = np.flatnonzero(future_low_local <= target_price)
                                sl_hits = np.flatnonzero(future_high_local >= sl_price)

                            target_rel = int(target_hits[0]) if target_hits.size else -1
                            sl_rel = int(sl_hits[0]) if sl_hits.size else -1

                            target_in = target_rel != -1
                            sl_in = sl_rel != -1

                            if not target_in and not sl_in:
                                s["censored_count"] += 1
                                exit_idx = local_len - 1
                                exit_price = float(close[anchor_idx + 1 + exit_idx])
                                forced_pnl_pct = ((exit_price - anchor_price) / anchor_price) if side == 1 else ((anchor_price - exit_price) / anchor_price)
                                forced_exit_r = forced_pnl_pct / sl_ratio
                                s["horizon_exit_r"].append(float(forced_exit_r))
                                s["forced_exit_r_all"].append(float(forced_exit_r))
                                exc_high = future_high_local[: exit_idx + 1]
                                exc_low = future_low_local[: exit_idx + 1]
                            else:
                                if target_in and sl_in:
                                    if target_rel < sl_rel:
                                        target_first = True
                                    elif sl_rel < target_rel:
                                        target_first = False
                                    else:
                                        target_first = not conservative_sl_first
                                elif target_in:
                                    target_first = True
                                else:
                                    target_first = False

                                if target_first:
                                    s["target_first_count"] += 1
                                    first_rel = target_rel
                                    target_min = (int(future_time_local[target_rel]) - int(time_ns[anchor_idx])) / MINUTE_NS
                                    s["target_minutes"].append(target_min)
                                    s["first_event_minutes"].append(target_min)
                                    if sl_in and sl_rel > target_rel:
                                        s["target_first_then_sl_count"] += 1
                                else:
                                    s["sl_first_count"] += 1
                                    first_rel = sl_rel
                                    sl_min = (int(future_time_local[sl_rel]) - int(time_ns[anchor_idx])) / MINUTE_NS
                                    s["sl_minutes"].append(sl_min)
                                    s["first_event_minutes"].append(sl_min)

                                exc_high = future_high_local[: first_rel + 1]
                                exc_low = future_low_local[: first_rel + 1]
                                exit_idx = target_rel if target_first else sl_rel
                                exit_price = float(close[anchor_idx + 1 + exit_idx])
                                forced_pnl_pct = ((exit_price - anchor_price) / anchor_price) if side == 1 else ((anchor_price - exit_price) / anchor_price)
                                forced_exit_r = forced_pnl_pct / sl_ratio
                                s["forced_exit_r_all"].append(float(forced_exit_r))

                            if side == 1:
                                adverse_pct = max(0.0, (anchor_price - float(np.min(exc_low))) / anchor_price)
                                favorable_pct = max(0.0, (float(np.max(exc_high)) - anchor_price) / anchor_price)
                            else:
                                adverse_pct = max(0.0, (float(np.max(exc_high)) - anchor_price) / anchor_price)
                                favorable_pct = max(0.0, (anchor_price - float(np.min(exc_low))) / anchor_price)

                            s["mae_r"].append(float(adverse_pct / sl_ratio))
                            s["mfe_r"].append(float(favorable_pct / sl_ratio))

                for vol_category in vol_categories:
                    for bucket_start_hour in bucket_starts:
                        bucket_label = (
                            _bucket_label_from_start_hour(bucket_start_hour, time_bucket_hours)
                            if include_entry_bucket_hours
                            else "ALL HOURS"
                        )

                        for h in horizons:
                            s = stats_by_vol[vol_category][bucket_start_hour][h]
                            target_first_count = int(s["target_first_count"])
                            sl_first_count = int(s["sl_first_count"])
                            resolved_count = target_first_count + sl_first_count
                            anchors_total = int(s["anchors_total"])

                            avg_target, med_target = _mean_median(s["target_minutes"])
                            avg_sl, med_sl = _mean_median(s["sl_minutes"])
                            avg_first, med_first = _mean_median(s["first_event_minutes"])
                            avg_horizon_exit_r, med_horizon_exit_r = _mean_median(s["horizon_exit_r"])
                            avg_forced_exit_r, med_forced_exit_r = _mean_median(s["forced_exit_r_all"])
                            avg_mae_r, med_mae_r = _mean_median(s["mae_r"])
                            avg_mfe_r, med_mfe_r = _mean_median(s["mfe_r"])

                            target_first_rate_pct = _pct_from_ratio(target_first_count / anchors_total if anchors_total else None)
                            sl_first_rate_pct = _pct_from_ratio(sl_first_count / anchors_total if anchors_total else None)
                            resolved_rate_pct = _pct_from_ratio(resolved_count / anchors_total if anchors_total else None)
                            censored_rate_pct = _pct_from_ratio((int(s["censored_count"]) / anchors_total) if anchors_total else None)
                            target_first_then_sl_rate_pct = _pct_from_ratio(
                                s["target_first_then_sl_count"] / target_first_count if target_first_count else None
                            )
                            target_given_resolved_rate_pct = _pct_from_ratio(
                                target_first_count / resolved_count if resolved_count else None
                            )

                            horizon_exit_positive_rate_pct = _pct_from_ratio(
                                (sum(1 for r in s["horizon_exit_r"] if r > 0.0) / len(s["horizon_exit_r"])) if s["horizon_exit_r"] else None
                            )
                            forced_exit_positive_rate_pct = _pct_from_ratio(
                                (sum(1 for r in s["forced_exit_r_all"] if r > 0.0) / len(s["forced_exit_r_all"])) if s["forced_exit_r_all"] else None
                            )

                            ma_candidate_count = int(s["ma_candidate_count"])
                            ma_pass_count = int(s["ma_pass_count"])
                            ma_pass_rate_pct = _pct_from_ratio(ma_pass_count / ma_candidate_count if ma_candidate_count else None)
                            avg_entry_vs_ma_pct, med_entry_vs_ma_pct = _mean_median(s["entry_vs_ma_pct"])

                            row = {
                                "instrument": instrument,
                                "filter_mode": filter_mode,
                                "ma_type": MA_TYPE.lower(),
                                "ma_period_bars": int(MA_PERIOD_BARS),
                                "use_ma_filter": bool(apply_ma_filter),
                                "vol_category": vol_category,
                                "risk_pct": float(RISK_PCT),
                                "target_pct": float(target_pct),
                                "sl_pct": float(sl_pct),
                                "horizon_hours": int(h),
                                "anchors_total": anchors_total,
                                "target_first_count": target_first_count,
                                "sl_first_count": sl_first_count,
                                "target_first_then_sl_count": int(s["target_first_then_sl_count"]),
                                "resolved_count": resolved_count,
                                "censored_count": int(s["censored_count"]),
                                "target_first_rate_pct": target_first_rate_pct,
                                "sl_first_rate_pct": sl_first_rate_pct,
                                "target_first_then_sl_rate_pct": target_first_then_sl_rate_pct,
                                "resolved_rate_pct": resolved_rate_pct,
                                "censored_rate_pct": censored_rate_pct,
                                "target_given_resolved_rate_pct": target_given_resolved_rate_pct,
                                "net_expectancy_pct": _net_expectancy_risk_pct(
                                    target_first_rate_pct or 0.0,
                                    float(target_pct),
                                    sl_first_rate_pct or 0.0,
                                    float(sl_pct),
                                    risk_pct=RISK_PCT,
                                ),
                                "avg_minutes_to_target": avg_target,
                                "median_minutes_to_target": med_target,
                                "avg_minutes_to_sl": avg_sl,
                                "median_minutes_to_sl": med_sl,
                                "avg_minutes_to_first_event": avg_first,
                                "median_minutes_to_first_event": med_first,
                                "avg_horizon_exit_r": avg_horizon_exit_r,
                                "median_horizon_exit_r": med_horizon_exit_r,
                                "horizon_exit_positive_rate_pct": horizon_exit_positive_rate_pct,
                                "avg_forced_exit_r": avg_forced_exit_r,
                                "median_forced_exit_r": med_forced_exit_r,
                                "forced_exit_positive_rate_pct": forced_exit_positive_rate_pct,
                                "avg_mae_r": avg_mae_r,
                                "median_mae_r": med_mae_r,
                                "avg_mfe_r": avg_mfe_r,
                                "median_mfe_r": med_mfe_r,
                                "ma_candidate_count": ma_candidate_count,
                                "ma_pass_count": ma_pass_count,
                                "ma_pass_rate_pct": ma_pass_rate_pct,
                                "avg_entry_vs_ma_pct": avg_entry_vs_ma_pct,
                                "median_entry_vs_ma_pct": med_entry_vs_ma_pct,
                                "horizon_eligible_count": int(s["horizon_eligible_count"]),
                                "horizon_eligible_rate_pct": _pct_from_ratio(
                                    s["horizon_eligible_count"] / anchors_total if anchors_total else None
                                ),
                            }

                            if include_entry_bucket_hours:
                                row.update(
                                    {
                                        "entry_bucket_start_hour": int(bucket_start_hour),
                                        "entry_bucket_hours": int(time_bucket_hours),
                                        "entry_bucket_label": bucket_label,
                                    }
                                )

                            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    sort_cols = ["instrument", "filter_mode", "vol_category", "target_pct", "sl_pct", "horizon_hours"]
    if include_entry_bucket_hours:
        sort_cols = ["instrument", "filter_mode", "vol_category", "entry_bucket_start_hour", "target_pct", "sl_pct", "horizon_hours"]

    out = out.sort_values(sort_cols).reset_index(drop=True)

    front_cols = [
        "instrument",
        "filter_mode",
        "ma_type",
        "ma_period_bars",
        "use_ma_filter",
        "vol_category",
        "entry_bucket_label" if include_entry_bucket_hours else None,
        "entry_bucket_start_hour" if include_entry_bucket_hours else None,
        "entry_bucket_hours" if include_entry_bucket_hours else None,
        "target_pct",
        "sl_pct",
        "horizon_hours",
    ]
    front_cols = [c for c in front_cols if c is not None and c in out.columns]

    remaining = [c for c in out.columns if c not in front_cols]
    return out[front_cols + remaining]


# =============================
# MAIN
# =============================

def main() -> None:
    source_symbol = MT5_SYMBOL
    bucket_mode = USE_ENTRY_BUCKET_HOURS
    time_bucket_hours = ENTRY_BUCKET_HOURS if bucket_mode else 24

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
        time_bucket_hours=time_bucket_hours,
        include_entry_bucket_hours=bucket_mode,
        use_ma_filter=USE_MA_FILTER,
    )

    if summary_df.empty:
        print("No summary produced.")
        return

    summary_df = summary_df.round(4).sort_values(
        ["instrument", "filter_mode", "vol_category", "target_pct", "sl_pct", "horizon_hours"]
    ).reset_index(drop=True)

    output_file = make_output_file(INSTRUMENT, PAIR, source_symbol, ENTRY_BUCKET_HOURS, bucket_mode)

    print("\n" + "=" * 140)
    print(
        f" TARGET vs SL SURVIVAL SCAN | instrument={INSTRUMENT} | symbol={source_symbol} | pair={PAIR} | "
        f"bucket={'ON' if bucket_mode else 'OFF'}"
    )
    print("=" * 140)
    print(f"Grid start: {GRID_START_DATE}")
    print(f"Grid end  : {GRID_END_DATE}")
    print(f"Rows loaded: {df.height:,}")
    print(f"Target Pct : {', '.join(f'{x:.1f}%' for x in TARGET_PCT_LIST)}")
    print(f"SL Pcts    : {', '.join(f'{x:.1f}%' for x in SL_PCT_LIST)}")
    print(f"Horizons   : {', '.join(f'{h}h' for h in HORIZON_HOURS_LIST)}")
    print(f"MT5 symbol : {source_symbol}")
    print(f"Volatility : {'ON' if USE_ROLLING_VOLATILITY_CHECK else 'OFF'} | lookback={VOL_LOOKBACK_DAYS}D")
    print(f"MA filter  : {'ON' if USE_MA_FILTER else 'OFF'} | {MA_TYPE.upper()}({MA_PERIOD_BARS})")

    export_df = summary_df.drop(columns=COLUMNS_TO_REMOVE, errors="ignore")
    export_df.to_csv(output_file, index=False)
    print(f"Saved summary CSV to: {output_file}")

    if "filter_mode" in summary_df.columns:
        print("\n=== FILTER MODE COMPARISON ===")
        mode_view = summary_df.groupby("filter_mode", dropna=False).agg(
            trades=("horizon_hours", "count"),
            avg_expectancy=("net_expectancy_pct", "mean"),
            median_expectancy=("net_expectancy_pct", "median"),
            avg_ma_pass_rate=("ma_pass_rate_pct", "mean"),
            avg_mae_r=("avg_mae_r", "mean"),
            avg_mfe_r=("avg_mfe_r", "mean"),
        ).reset_index()
        print(mode_view.to_string(index=False))

    if not bucket_mode:
        plot_files = save_net_expectancy_tp_sl_plot(
            summary_df=summary_df,
            instrument=INSTRUMENT,
            pair=PAIR,
            symbol=source_symbol,
            include_entry_bucket_hours=bucket_mode,
        )
        if plot_files:
            print("\nSaved plot file(s):")
            for p in plot_files:
                print(p)
    else:
        print("\nHeatmap generation skipped (USE_ENTRY_BUCKET_HOURS = True)")


if __name__ == "__main__":
    main()