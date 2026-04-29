
from __future__ import annotations

import pythoncom

try:
    pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
except pythoncom.com_error:
    # COM already initialized with a different model — this is OK
    pass

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

# =============================
# CONFIG
# =============================

LOOKBACK_DAYS = 180
ENTRY_BUCKET_HOURS = 4
USE_ENTRY_BUCKET_HOURS = False
RISK_PCT = 0.005  # 0.5% equity risk per trade

if 24 % ENTRY_BUCKET_HOURS != 0:
    raise ValueError("ENTRY_BUCKET_HOURS must divide 24 exactly.")

MYT = timezone(timedelta(hours=8), name="MYT")


def _default_utc_window(days_back: int = LOOKBACK_DAYS) -> tuple[str, str]:
    end_dt = datetime.now(timezone.utc).replace(microsecond=0)
    start_dt = end_dt - timedelta(days=days_back)
    start_s = start_dt.isoformat().replace("+00:00", "Z")
    end_s = end_dt.isoformat().replace("+00:00", "Z")
    return start_s, end_s


GRID_START_DATE, GRID_END_DATE = _default_utc_window(LOOKBACK_DAYS)

# Change this one line to switch instrument.
INSTRUMENT = "USDCHF"  # BTC | UK100 | AUDJPY | XAUUSD

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
        "tp_range": {"min": 0.06, "max": 0.25, "step": 0.01},
        "sl_range": {"min": 0.06, "max": 0.25, "step": 0.01},
        "horizon_hours_list": [3],
    },
    "AUDJPY": {
        "pair": "AUDJPY",
        "mt5_symbol": "AUDJPY",
        "tp_range": {"min": 0.03, "max": 0.18, "step": 0.01},
        "sl_range": {"min": 0.03, "max": 0.18, "step": 0.01},
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

def _float_range_inclusive(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if stop < start:
        raise ValueError("stop must be >= start")

    values: List[float] = []
    cur = float(start)

    # small epsilon so the last value is included when it should be
    eps = step / 10_000.0

    while cur <= stop + eps:
        values.append(round(cur, 10))
        cur += step

    return values

TARGET_PCT_LIST = _float_range_inclusive(
    CFG["tp_range"]["min"],
    CFG["tp_range"]["max"],
    CFG["tp_range"]["step"],
)

SL_PCT_LIST = _float_range_inclusive(
    CFG["sl_range"]["min"],
    CFG["sl_range"]["max"],
    CFG["sl_range"]["step"],
)

HORIZON_HOURS_LIST = CFG["horizon_hours_list"]

# MT5 connection settings from env (credentials / terminal connection only)
MT5_PATH = None  # optionally set to r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_TIMEOUT = int(os.getenv("MT5_TIMEOUT", "60000"))
MT5_PORTABLE = os.getenv("MT5_PORTABLE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}

MT5_CHUNK_DAYS = 90

OUTPUT_DIR = BASE_DIR / "data_lake" / "Saved_results" / "expectancy_checks"
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

    # Reward is scaled relative to stop size.
    # Example: tp=0.8, sl=0.2 => 4R reward
    rr_multiple = float(target_pct) / float(sl_pct)

    win_pct = risk_pct * rr_multiple * 100.0   # convert to percent units
    loss_pct = risk_pct * 100.0                # stop-out loss in percent units

    return (tp_rate * win_pct) - (sl_rate * loss_pct)


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

def _market_tag(instrument: str, pair: str, symbol: str) -> str:
    for value in (pair, instrument, symbol):
        if value:
            return _safe_name(str(value))
    return "market"


def _bucket_tag(bucket_hours: Optional[int], use_bucket_hours: bool) -> str:
    if use_bucket_hours:
        return f"{int(bucket_hours)}h_myt_buckets"
    return "no_myt_buckets"

def make_output_file(
    instrument: str,
    pair: str,
    symbol: str,
    bucket_hours: Optional[int],
    use_bucket_hours: bool,
) -> Path:
    market = _market_tag(instrument, pair, symbol)
    bucket_part = _bucket_tag(bucket_hours, use_bucket_hours)
    return OUTPUT_DIR / f"expectancy_scan_{market}_{bucket_part}.csv"

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

    if include_entry_bucket_hours and "entry_bucket_label" in summary_df.columns:
        bucket_groups = [
            (str(bucket_label), group.copy())
            for bucket_label, group in summary_df.groupby("entry_bucket_label", sort=True)
        ]
    else:
        bucket_groups = [("ALL HOURS", summary_df.copy())]

    for bucket_label, group in bucket_groups:
        group = group.copy()
        horizons = sorted(group["horizon_hours"].dropna().unique().tolist())
        if not horizons:
            continue

        vmin = float(group["net_expectancy_pct"].min())
        vmax = float(group["net_expectancy_pct"].max())

        n_plots = len(horizons)
        fig, axes = plt.subplots(
            1,
            n_plots,
            figsize=(7.5 * n_plots, 6.5),
            squeeze=False,
            constrained_layout=True,
        )
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

            annot_size = _heatmap_annot_size(pivot)

            hm = sns.heatmap(
                pivot,
                ax=ax,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                annot=False,
                fmt=".3f",
                annot_kws={"size": annot_size},
                linewidths=0.5,
                linecolor="white",
                cbar=False,
                square=True,
            )

            if first_mappable is None and hm.collections:
                first_mappable = hm.collections[0]

            ax.set_title(f"{market} | {bucket_label} | {int(h)}h", fontsize=12, pad=10)
            ax.set_xlabel("Target %")
            ax.set_ylabel("SL %")
            ax.tick_params(axis="x", labelrotation=0, labelsize=8)
            ax.tick_params(axis="y", labelrotation=0, labelsize=8)

        if first_mappable is not None:
            fig.colorbar(
                first_mappable,
                ax=axes_row.tolist(),
                shrink=0.9,
                pad=0.02,
                label="Net expectancy (%)",
            )

        safe_bucket = _safe_name(bucket_label)
        out_file = OUTPUT_DIR / f"expectancy_heatmap_{market}_{safe_bucket}.png"
        fig.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(out_file)

    return saved_files

def _heatmap_annot_size(pivot: pd.DataFrame) -> int:
    n_rows, n_cols = pivot.shape
    biggest = max(n_rows, n_cols)

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

def save_tp_sl_combo_report(summary_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for h, sub in summary_df.groupby("horizon_hours"):
        sub = sub.dropna(subset=["target_pct", "sl_pct", "net_expectancy_pct"]).copy()
        if len(sub) < 4:
            continue

        tp = sub["target_pct"].to_numpy(dtype=float)
        sl = sub["sl_pct"].to_numpy(dtype=float)
        y = sub["net_expectancy_pct"].to_numpy(dtype=float)

        # 2D surface with interaction + curvature
        X = np.column_stack([
            np.ones(len(sub)),
            tp,
            sl,
            tp * sl,
            tp ** 2,
            sl ** 2,
        ])

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta

        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rows.append(
            {
                "horizon_hours": int(h),
                "intercept": float(beta[0]),
                "tp_coef": float(beta[1]),
                "sl_coef": float(beta[2]),
                "tp_sl_interaction_coef": float(beta[3]),
                "tp_sq_coef": float(beta[4]),
                "sl_sq_coef": float(beta[5]),
                "surface_r2": r2,
                "spearman_tp_expectancy": sub["target_pct"].corr(sub["net_expectancy_pct"], method="spearman"),
                "spearman_sl_expectancy": sub["sl_pct"].corr(sub["net_expectancy_pct"], method="spearman"),
            }
        )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "horizon_hours",
                "intercept",
                "tp_coef",
                "sl_coef",
                "tp_sl_interaction_coef",
                "tp_sq_coef",
                "sl_sq_coef",
                "surface_r2",
                "spearman_tp_expectancy",
                "spearman_sl_expectancy",
            ]
        )
        empty.to_csv(out_path, index=False)
        return empty

    report = (
        pd.DataFrame(rows)
        .sort_values("horizon_hours")
        .reset_index(drop=True)
    )

    report.to_csv(out_path, index=False)
    return report

def save_tp_sl_correlation_report(summary_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for h, sub in summary_df.groupby("horizon_hours"):
        rows.append(
            {
                "horizon_hours": int(h),
                "spearman_tp_expectancy": sub["target_pct"].corr(sub["net_expectancy_pct"], method="spearman"),
                "spearman_sl_expectancy": sub["sl_pct"].corr(sub["net_expectancy_pct"], method="spearman"),
                "spearman_tp_sl": sub["target_pct"].corr(sub["sl_pct"], method="spearman"),
                "pearson_tp_expectancy": sub["target_pct"].corr(sub["net_expectancy_pct"], method="pearson"),
                "pearson_sl_expectancy": sub["sl_pct"].corr(sub["net_expectancy_pct"], method="pearson"),
            }
        )

    report = pd.DataFrame(rows).sort_values("horizon_hours").reset_index(drop=True)
    report.to_csv(out_path, index=False)
    return report

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
) -> pd.DataFrame:
    if df.is_empty():
        return pd.DataFrame()

    if include_entry_bucket_hours and 24 % time_bucket_hours != 0:
        raise ValueError("time_bucket_hours must divide 24 exactly.")

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

    if include_entry_bucket_hours:
        bucket_starts = list(range(0, 24, time_bucket_hours))
    else:
        bucket_starts = [0]

    rows: List[Dict[str, Any]] = []

    for target_pct in target_list:
        target_ratio = float(target_pct) / 100.0

        for sl_pct in sl_list:
            sl_ratio = float(sl_pct) / 100.0

            stats_by_bucket: Dict[int, Dict[int, Dict[str, Any]]] = {
                b: {
                    h: {
                        "target_minutes": [],
                        "sl_minutes": [],
                        "first_event_minutes": [],
                        "horizon_exit_r": [],        # only trades still open at horizon
                        "forced_exit_r_all": [],     # all eligible trades closed at horizon
                        "horizon_eligible_count": 0,  # trades with enough future data for this horizon
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

            for anchor_idx in range(n - 1):
                anchor_price = float(close[anchor_idx])
                if not np.isfinite(anchor_price) or anchor_price <= 0.0:
                    continue

                bucket_start_hour = (
                    _bucket_start_hour_from_time_ns_myt(int(time_ns[anchor_idx]), time_bucket_hours)
                    if include_entry_bucket_hours
                    else 0
                )

                future_high_full = high[anchor_idx + 1:]
                future_low_full = low[anchor_idx + 1:]
                future_time_full = time_ns[anchor_idx + 1:]

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
                        s = stats_by_bucket[bucket_start_hour][h]
                        s["anchors_total"] += 1

                        horizon_end_abs = int(horizon_end_by_h[h][anchor_idx])
                        if horizon_end_abs <= anchor_idx + 1:
                            continue

                        local_len = horizon_end_abs - (anchor_idx + 1)
                        if local_len <= 0:
                            continue

                        # ✅ SAFE DEBUG — anchor_idx is defined here
                        # if anchor_idx in (1000, 5000, 10000) and h in (24, 25, 26):
                        #     print(
                        #         f"[DEBUG] anchor={anchor_idx}",
                        #         f"h={h}",
                        #         f"horizon_end_abs={horizon_end_abs}",
                        #         f"seg_end_abs={seg_end_abs}",
                        #         f"local_len={local_len}",
                        #         f"cutoff_time={pd.Timestamp(int(time_ns[anchor_idx] + h * HOUR_NS), unit='ns', tz='UTC')}",
                        #     )

                        s["horizon_eligible_count"] += 1

                        # Forced exit at horizon for ALL eligible trades
                        horizon_exit_idx = horizon_end_abs - 1
                        horizon_exit_price = float(close[horizon_exit_idx])

                        if side == 1:
                            forced_pnl_pct = (horizon_exit_price - anchor_price) / anchor_price
                        else:
                            forced_pnl_pct = (anchor_price - horizon_exit_price) / anchor_price

                        forced_exit_r = forced_pnl_pct / sl_ratio
                        s["forced_exit_r_all"].append(float(forced_exit_r))

                        target_in = target_rel_full != -1 and target_rel_full < local_len
                        sl_in = sl_rel_full != -1 and sl_rel_full < local_len

                        if not target_in and not sl_in:
                            s["censored_count"] += 1

                            # This is the unresolved-only horizon exit metric
                            s["horizon_exit_r"].append(float(forced_exit_r))
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
                            stats_by_bucket[bucket_start_hour][h]["target_first_count"] += 1
                            target_min = (int(future_time_full[target_rel_full]) - int(time_ns[anchor_idx])) / MINUTE_NS
                            stats_by_bucket[bucket_start_hour][h]["target_minutes"].append(target_min)
                            stats_by_bucket[bucket_start_hour][h]["first_event_minutes"].append(target_min)

                            if sl_in and sl_rel_full > target_rel_full:
                                stats_by_bucket[bucket_start_hour][h]["target_first_then_sl_count"] += 1
                        else:
                            stats_by_bucket[bucket_start_hour][h]["sl_first_count"] += 1
                            sl_min = (int(future_time_full[sl_rel_full]) - int(time_ns[anchor_idx])) / MINUTE_NS
                            stats_by_bucket[bucket_start_hour][h]["sl_minutes"].append(sl_min)
                            stats_by_bucket[bucket_start_hour][h]["first_event_minutes"].append(sl_min)

            for bucket_start_hour in bucket_starts:
                bucket_label = (
                    _bucket_label_from_start_hour(bucket_start_hour, time_bucket_hours)
                    if include_entry_bucket_hours
                    else "ALL HOURS"
                )

                for h in horizons:
                    s = stats_by_bucket[bucket_start_hour][h]
                    target_first_count = int(s["target_first_count"])
                    sl_first_count = int(s["sl_first_count"])
                    censored_count = int(s["censored_count"])
                    resolved_count = target_first_count + sl_first_count
                    anchors_total = int(s["anchors_total"])

                    avg_target, med_target = _mean_median(s["target_minutes"])
                    avg_sl, med_sl = _mean_median(s["sl_minutes"])
                    avg_first, med_first = _mean_median(s["first_event_minutes"])

                    avg_horizon_exit_r, med_horizon_exit_r = _mean_median(s["horizon_exit_r"])
                    avg_forced_exit_r, med_forced_exit_r = _mean_median(s["forced_exit_r_all"])

                    horizon_exit_positive_rate_pct = _pct_from_ratio(
                        (
                            sum(1 for r in s["horizon_exit_r"] if r > 0.0) / len(s["horizon_exit_r"])
                            if s["horizon_exit_r"]
                            else None
                        )
                    )

                    forced_exit_positive_rate_pct = _pct_from_ratio(
                        (
                            sum(1 for r in s["forced_exit_r_all"] if r > 0.0) / len(s["forced_exit_r_all"])
                            if s["forced_exit_r_all"]
                            else None
                        )
                    )

                    horizon_eligible_rate_pct = _pct_from_ratio(
                        s["horizon_eligible_count"] / anchors_total if anchors_total else None
                    )

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

                    net_expectancy_risk_pct = _net_expectancy_risk_pct(
                        target_first_rate_pct or 0.0,
                        float(target_pct),
                        sl_first_rate_pct or 0.0,
                        float(sl_pct),
                        risk_pct=RISK_PCT,
                    )

                    row = {
                        "instrument": instrument,
                        "symbol": source_symbol,
                        "pair": pair,
                        "risk_pct": float(RISK_PCT),
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
                        "net_expectancy_risk_pct": net_expectancy_risk_pct,
                        "net_expectancy_pct": net_expectancy_risk_pct,
                        "avg_minutes_to_target": avg_target,
                        "median_minutes_to_target": med_target,
                        "avg_minutes_to_sl": avg_sl,
                        "median_minutes_to_sl": med_sl,
                        "avg_minutes_to_first_event": avg_first,
                        "median_minutes_to_first_event": med_first,
                        "horizon_eligible_count": int(s["horizon_eligible_count"]),
                        "horizon_eligible_rate_pct": horizon_eligible_rate_pct,
                        "avg_horizon_exit_r": avg_horizon_exit_r,
                        "median_horizon_exit_r": med_horizon_exit_r,
                        "horizon_exit_positive_rate_pct": horizon_exit_positive_rate_pct,
                        "avg_forced_exit_r": avg_forced_exit_r,
                        "median_forced_exit_r": med_forced_exit_r,
                        "forced_exit_positive_rate_pct": forced_exit_positive_rate_pct,
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

    sort_cols = ["instrument", "target_pct", "sl_pct", "horizon_hours"]
    if include_entry_bucket_hours:
        sort_cols = ["instrument", "entry_bucket_start_hour", "target_pct", "sl_pct", "horizon_hours"]

    out = out.sort_values(sort_cols).reset_index(drop=True)

    if include_entry_bucket_hours:
        front_cols = [
            "instrument",
            "symbol",
            "pair",
            "entry_bucket_label",
            "entry_bucket_start_hour",
            "entry_bucket_hours",
            "target_pct",
            "sl_pct",
            "horizon_hours",
        ]
    else:
        front_cols = [
            "instrument",
            "symbol",
            "pair",
            "target_pct",
            "sl_pct",
            "horizon_hours",
        ]

    remaining = [c for c in out.columns if c not in front_cols]
    out = out[front_cols + remaining]

    return out


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
    )

    summary_df = summary_df.round(4)

    sort_cols = ["instrument", "symbol", "pair", "target_pct", "sl_pct", "horizon_hours"]
    if bucket_mode:
        sort_cols = ["instrument", "entry_bucket_label", "target_pct", "sl_pct", "horizon_hours"]

    summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)

    output_file = make_output_file(INSTRUMENT, PAIR, source_symbol, ENTRY_BUCKET_HOURS, bucket_mode)

    bucket_text = f"{ENTRY_BUCKET_HOURS}h MYT" if bucket_mode else "OFF"

    print("\n" + "=" * 160)
    print(
        f" TARGET vs SL SURVIVAL SCAN | instrument={INSTRUMENT} | symbol={source_symbol} | pair={PAIR} | bucket={bucket_text}"
    )
    print("=" * 160)
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

    plot_files = []

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

    combo_file = OUTPUT_DIR / f"expectancy_combo_{_market_tag(INSTRUMENT, PAIR, source_symbol)}.csv"
    combo_df = save_tp_sl_combo_report(summary_df, combo_file)
    print(f"Saved combo report to: {combo_file}")
    print(combo_df.to_string(index=False))

if __name__ == "__main__":
    main()
