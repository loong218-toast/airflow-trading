# metatrader_trades.py

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import MetaTrader5 as mt5


# =========================
# SETTINGS
# =========================

HISTORY_DAYS = 365
BAR_TIMEFRAME = mt5.TIMEFRAME_M5

# Per-symbol rules
# tp_pct, sl_pct, spread_pct are in percent units
# last_n_trades = -1 means keep all trades for that symbol
# MIX_BUY_SELL=True lets each symbol define separate buy/sell rules.
MIX_BUY_SELL = True
SYMBOL_RULES: Dict[str, Dict[str, Any]] = {
    "FTSE100": {
        "buy": {
            "tp_pct": 0.28,
            "sl_pct": 0.35,
            "spread_pct": 0.007,
        },
        "sell": {
            "tp_pct": 0.28,
            "sl_pct": 0.35,
            "spread_pct": 0.007,
        },
        "last_n_trades": -1,
    },
    "AUDJPY": {
        "buy": {
            "tp_pct": 0.40,
            "sl_pct": 0.90,
            "spread_pct": 0.007,
        },
        "sell": {
            "tp_pct": 0.40,
            "sl_pct": 0.90,
            "spread_pct": 0.007,
        },
        "last_n_trades": -1,
    },
    "USDCHF": {
        "buy": {
            "tp_pct": 0.30,
            "sl_pct": 0.80,
            "spread_pct": 0.005,
        },
        "sell": {
            "tp_pct": 0.30,
            "sl_pct": 0.80,
            "spread_pct": 0.005,
        },
        "last_n_trades": -1,
    },
}


SYMBOL_FILTER: Optional[str] = None
MAGIC_FILTER: Optional[int] = None

# If both TP and SL are inside the same 5m candle, count it as SL.
CONSERVATIVE_SAME_BAR_SL_FIRST = True

BASE_DIR = Path(r"C:\Users\Owner\airflow-trading\data_lake")
TRADE_DATA_DIR = BASE_DIR / "mt5_trade_data"
TRADE_DATA_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_DIR = BASE_DIR / "Saved_results" / "mt5_symbol_tp_sl_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

RAW_DEALS_FILE = TRADE_DATA_DIR / "mt5_deals_history.csv"
CLOSED_TRADES_FILE = TRADE_DATA_DIR / "mt5_closed_trades.csv"
OVERVIEW_FILE = ANALYSIS_DIR / "overall_summary.csv"
SUMMARY_FILE = ANALYSIS_DIR / "symbol_summary.csv"

# Split detail outputs
PNL_DETAIL_FILE = ANALYSIS_DIR / "trade_level_pnl_summary.csv"
EXCURSION_DETAIL_FILE = ANALYSIS_DIR / "trade_level_mae_mfe_summary.csv"

RESET_MIN_BAR_INDEX = 1          # ignore the entry bar itself when looking for reset
RESET_TOL_PCT = 0              # 0.0 means exact touch of entry; raise slightly if needed

# =========================
# HELPERS
# =========================

def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def init_mt5() -> None:
    ok = mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")


def ensure_utc(ts: Any) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _safe_metric(metrics: Any, key: str) -> float:
    if not isinstance(metrics, dict):
        return np.nan
    value = metrics.get(key, np.nan)
    try:
        return float(value) if pd.notna(value) else np.nan
    except Exception:
        return np.nan


def pct_to_ratio(pct_value: float) -> float:
    return float(pct_value) / 100.0


def round_price(x: Any) -> Any:
    if pd.isna(x):
        return x
    return round(float(x), 3)


def round_pct(x: Any) -> Any:
    if pd.isna(x):
        return x
    return round(float(x), 3)

def _round_summary_numbers(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if c == "symbol":
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(decimals)
    return out

def _side_move_pct(side: int, entry_price: float, exit_price: float) -> float:
    if not np.isfinite(entry_price) or entry_price <= 0.0 or not np.isfinite(exit_price):
        return np.nan
    if side == 1:
        return ((exit_price - entry_price) / entry_price) * 100.0
    return ((entry_price - exit_price) / entry_price) * 100.0


def _infer_tp_sl_from_exit(side: int, entry_price: float, exit_price: float) -> tuple[float, float]:
    move_pct = _side_move_pct(side, entry_price, exit_price)
    if not np.isfinite(move_pct):
        return np.nan, np.nan

    move_pct = abs(float(move_pct))
    if side == 1:
        return (move_pct, np.nan) if exit_price >= entry_price else (np.nan, move_pct)
    else:
        return (move_pct, np.nan) if exit_price <= entry_price else (np.nan, move_pct)


def rule_for(symbol: str) -> Optional[Dict[str, float]]:
    return SYMBOL_RULES.get(symbol)


def symbol_last_n(symbol: str) -> int:
    rule = rule_for(symbol)
    if rule is None:
        return -1
    return int(rule.get("last_n_trades", -1))

def trade_rule_for(symbol: str, side: int) -> Optional[Dict[str, Any]]:
    rule = rule_for(symbol)
    if rule is None:
        return None

    if not MIX_BUY_SELL or ("tp_pct" in rule and "sl_pct" in rule and "spread_pct" in rule):
        return rule

    side_key = "buy" if side == 1 else "sell"
    side_rule = rule.get(side_key, {})
    if not isinstance(side_rule, dict):
        side_rule = {}

    merged: Dict[str, Any] = {}
    for key in ("tp_pct", "sl_pct", "spread_pct", "last_n_trades"):
        if key in rule and key not in {"buy", "sell"}:
            merged[key] = rule[key]
    merged.update({k: v for k, v in side_rule.items() if v is not None})
    return merged

def _actual_exit_ratio(entry_price: float, exit_price: float) -> float:
    if not np.isfinite(entry_price) or entry_price <= 0.0 or not np.isfinite(exit_price):
        return np.nan
    return abs(float(exit_price) - float(entry_price)) / float(entry_price)


def append_csv_dedup(df: pd.DataFrame, file_path: Path, key_cols: List[str]) -> None:
    if df.empty:
        return

    df = df.copy()

    if file_path.exists():
        old = pd.read_csv(file_path)
        old = old.reindex(columns=df.columns)
        combined = pd.concat([old, df], ignore_index=True)
    else:
        combined = df.copy()

    combined = combined.reindex(columns=df.columns)

    keys = [c for c in key_cols if c in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")
        combined = combined.sort_values(keys).reset_index(drop=True)
    else:
        combined = combined.drop_duplicates(keep="last").reset_index(drop=True)

    combined.to_csv(file_path, index=False)


def floor_to_m5(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.floor("5min")


def filter_last_n_per_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    parts: List[pd.DataFrame] = []

    for symbol, g in df.groupby("symbol", dropna=False):
        n = symbol_last_n(str(symbol))
        g = g.sort_values(["entry_time", "position_id"]).reset_index(drop=True)

        if n == -1:
            parts.append(g)
        else:
            if n <= 0:
                raise ValueError(f"last_n_trades for {symbol} must be -1 or a positive integer")
            parts.append(g.tail(n))

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if out.empty:
        return out

    return out.sort_values(["entry_time", "position_id"]).reset_index(drop=True)


def _safe_mean_median(series: pd.Series) -> tuple[Optional[float], Optional[float]]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, None
    return float(s.mean()), float(s.median())


def _trade_won_by_price(side: int, entry_price: float, exit_price: float) -> bool:
    move_pct = _side_move_pct(side, entry_price, exit_price)
    return bool(np.isfinite(move_pct) and move_pct > 0.0)


# =========================
# MT5 DATA
# =========================

def _deals_to_dataframe(deals: Any) -> pd.DataFrame:
    if deals is None or len(deals) == 0:
        return pd.DataFrame()

    first = deals[0]

    if hasattr(first, "_asdict"):
        df = pd.DataFrame.from_records([d._asdict() for d in deals])
    else:
        df = pd.DataFrame(deals)

    if df.empty:
        return df

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    if "time_msc" in df.columns:
        df["time_msc"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)

    return df


def load_deals(days: int) -> pd.DataFrame:
    end_dt = utc_now()
    start_dt = end_dt - timedelta(days=days)

    deals = mt5.history_deals_get(start_dt, end_dt)
    if deals is None:
        raise RuntimeError(f"history_deals_get() failed: {mt5.last_error()}")

    return _deals_to_dataframe(deals)


def closed_positions_from_deals(deals_df: pd.DataFrame) -> pd.DataFrame:
    if deals_df.empty:
        return pd.DataFrame()

    df = deals_df.copy()

    if "position_id" not in df.columns and "position" in df.columns:
        df["position_id"] = df["position"]

    needed = {"time", "price", "symbol", "type", "entry"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing MT5 deal columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    if "position_id" not in df.columns:
        raise RuntimeError(f"No position_id/position column found. Available columns: {list(df.columns)}")

    if SYMBOL_FILTER is not None:
        df = df[df["symbol"] == SYMBOL_FILTER].copy()

    if MAGIC_FILTER is not None and "magic" in df.columns:
        df = df[df["magic"] == MAGIC_FILTER].copy()

    if df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for pos_id, g in df.groupby("position_id", dropna=True):
        g = g.sort_values("time").copy()

        entry_g = g[g["entry"] == mt5.DEAL_ENTRY_IN].copy()
        if entry_g.empty:
            continue

        exit_g = g[g["entry"].isin([mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY])].copy()

        entry_deal = entry_g.iloc[0]
        exit_deal = exit_g.iloc[-1] if not exit_g.empty else g.iloc[-1]

        deal_type = int(entry_deal["type"])
        if deal_type == mt5.DEAL_TYPE_BUY:
            side = 1
        elif deal_type == mt5.DEAL_TYPE_SELL:
            side = -1
        else:
            continue

        volume = float(entry_deal["volume"]) if "volume" in entry_deal.index else np.nan

        fee = float(g["fee"].sum()) if "fee" in g.columns else 0.0
        commission = float(g["commission"].sum()) if "commission" in g.columns else 0.0
        swap = float(g["swap"].sum()) if "swap" in g.columns else 0.0
        gross_profit = float(g["profit"].sum()) if "profit" in g.columns else 0.0
        net_actual_pnl = gross_profit + commission + swap + fee

        rows.append(
            {
                "position_id": int(pos_id) if pd.notna(pos_id) else None,
                "symbol": str(entry_deal["symbol"]),
                "side": side,
                "volume": volume,
                "entry_time": pd.to_datetime(entry_deal["time"], utc=True),
                "entry_price": float(entry_deal["price"]),
                "exit_time": pd.to_datetime(exit_deal["time"], utc=True),
                "exit_price": float(exit_deal["price"]),
                "gross_profit": gross_profit,
                "commission": commission,
                "swap": swap,
                "fee": fee,
                "net_actual_pnl": net_actual_pnl,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["entry_time", "position_id"]).reset_index(drop=True)


def fetch_bars(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, BAR_TIMEFRAME, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["time", "open", "high", "low", "close"]].dropna().reset_index(drop=True)


def build_bars_cache(trades_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}
    now_utc = utc_now()

    for symbol, g in trades_df.groupby("symbol"):
        start_ts = ensure_utc(g["entry_time"].min())
        start = floor_to_m5(start_ts)
        bars = fetch_bars(symbol, start.to_pydatetime(), now_utc)
        cache[str(symbol)] = bars

    return cache


# =========================
# SIMULATION
# =========================

def calc_profit_proxy(symbol: str, side: int, volume: float, entry_price: float, exit_price: float) -> Optional[float]:
    action = mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL

    try:
        if not mt5.symbol_select(symbol, True):
            return None

        profit = mt5.order_calc_profit(action, symbol, volume, entry_price, exit_price)
        if profit is None:
            return None

        return float(profit)
    except Exception:
        return None


def simulate_trade_m5(
    symbol: str,
    side: int,
    volume: float,
    entry_time: pd.Timestamp,
    entry_price: float,
    gross_profit_actual: float,
    commission: float,
    swap: float,
    fee: float,
    tp_pct: float,
    sl_pct: float,
    spread_pct: float,
    bars: pd.DataFrame,
) -> Dict[str, Any]:
    tp_ratio = pct_to_ratio(tp_pct)
    sl_ratio = pct_to_ratio(sl_pct)
    spread_ratio = pct_to_ratio(spread_pct)

    entry_dt = ensure_utc(entry_time)
    entry_floor = floor_to_m5(entry_dt)

    replay = bars[bars["time"] >= entry_floor].copy()
    replay = replay.sort_values("time").reset_index(drop=True)

    if side == 1:
        ref_entry = entry_price * (1.0 + spread_ratio / 2.0)
        tp_price = ref_entry * (1.0 + tp_ratio)
        sl_price = ref_entry * (1.0 - sl_ratio)
    else:
        ref_entry = entry_price * (1.0 - spread_ratio / 2.0)
        tp_price = ref_entry * (1.0 - tp_ratio)
        sl_price = ref_entry * (1.0 + sl_ratio)

    sim_close_type = "not_hit_in_data"
    sim_exit_price_used = np.nan
    sim_exit_time = pd.NaT

    if not replay.empty:
        for _, bar in replay.iterrows():
            high = float(bar["high"])
            low = float(bar["low"])

            if side == 1:
                tp_hit = high >= tp_price
                sl_hit = low <= sl_price
            else:
                tp_hit = low <= tp_price
                sl_hit = high >= sl_price

            if tp_hit and sl_hit:
                sim_exit_price_used = sl_price
                sim_close_type = "sl"
                sim_exit_time = pd.Timestamp(bar["time"])
                break

            if tp_hit:
                sim_exit_price_used = tp_price
                sim_close_type = "tp"
                sim_exit_time = pd.Timestamp(bar["time"])
                break

            if sl_hit:
                sim_exit_price_used = sl_price
                sim_close_type = "sl"
                sim_exit_time = pd.Timestamp(bar["time"])
                break

    sim_gross_profit = np.nan
    sim_net_pnl = np.nan
    sim_pnl_pct = np.nan

    if sim_close_type in {"tp", "sl"}:
        if side == 1:
            sim_pnl_pct = ((sim_exit_price_used - entry_price) / entry_price) * 100.0
        else:
            sim_pnl_pct = ((entry_price - sim_exit_price_used) / entry_price) * 100.0

        sim_gross_profit = calc_profit_proxy(symbol, side, volume, entry_price, float(sim_exit_price_used))
        if sim_gross_profit is None:
            sim_gross_profit = gross_profit_actual

        sim_net_pnl = sim_gross_profit + commission + swap + fee

    return {
        "ref_entry_price": round_price(ref_entry),
        "tp_price": round_price(tp_price),
        "sl_price": round_price(sl_price),
        "sim_exit_price_used": round_price(sim_exit_price_used),
        "sim_exit_time": sim_exit_time,
        "sim_close_type": sim_close_type,
        "sim_pnl_pct": round_pct(sim_pnl_pct),
        "sim_gross_profit": round_price(sim_gross_profit),
        "sim_net_pnl": round_price(sim_net_pnl),
    }


def simulate_trade_m5_avg_volume(
    symbol: str,
    side: int,
    volume: float,
    avg_volume: float,
    entry_time: pd.Timestamp,
    entry_price: float,
    gross_profit_actual: float,
    commission: float,
    swap: float,
    fee: float,
    tp_pct: float,
    sl_pct: float,
    spread_pct: float,
    bars: pd.DataFrame,
) -> Dict[str, Any]:
    base = simulate_trade_m5(
        symbol=symbol,
        side=side,
        volume=volume,
        entry_time=entry_time,
        entry_price=entry_price,
        gross_profit_actual=gross_profit_actual,
        commission=commission,
        swap=swap,
        fee=fee,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        spread_pct=spread_pct,
        bars=bars,
    )

    avg_scale = (avg_volume / volume) if (volume and volume > 0) else np.nan

    if base["sim_close_type"] not in {"tp", "sl"} or pd.isna(base["sim_exit_price_used"]):
        return {
            **base,
            "avg_volume_used": round_price(avg_volume),
            "sim_avgvol_gross_profit": np.nan,
            "sim_avgvol_pnl": np.nan,
        }

    sim_exit_price_used = float(base["sim_exit_price_used"])

    sim_avgvol_gross_profit = calc_profit_proxy(symbol, side, avg_volume, entry_price, sim_exit_price_used)
    if sim_avgvol_gross_profit is None:
        if pd.notna(avg_scale):
            sim_avgvol_gross_profit = float(base["sim_gross_profit"]) * avg_scale
        else:
            sim_avgvol_gross_profit = float(base["sim_gross_profit"])

    if pd.notna(avg_scale):
        avg_commission = commission * avg_scale
        avg_swap = swap * avg_scale
        avg_fee = fee * avg_scale
    else:
        avg_commission = commission
        avg_swap = swap
        avg_fee = fee

    sim_avgvol_pnl = sim_avgvol_gross_profit + avg_commission + avg_swap + avg_fee

    return {
        **base,
        "avg_volume_used": round_price(avg_volume),
        "sim_avgvol_gross_profit": round_price(sim_avgvol_gross_profit),
        "sim_avgvol_pnl": round_price(sim_avgvol_pnl),
    }


# =========================
# SAVE HELPERS
# =========================

def save_trade_history(deals_df: pd.DataFrame, closed_df: pd.DataFrame) -> None:
    if not deals_df.empty:
        raw = deals_df.copy()
        for col in ["price", "profit", "commission", "swap", "fee"]:
            if col in raw.columns:
                raw[col] = raw[col].apply(round_price)
        append_csv_dedup(raw, RAW_DEALS_FILE, key_cols=["ticket"] if "ticket" in raw.columns else ["time", "symbol"])

    if not closed_df.empty:
        closed = closed_df.copy()
        for col in ["gross_profit", "commission", "swap", "fee", "net_actual_pnl"]:
            if col in closed.columns:
                closed[col] = closed[col].apply(round_price)
        append_csv_dedup(closed, CLOSED_TRADES_FILE, key_cols=["position_id"])


def _trade_pnl_export_columns() -> list[str]:
    return [
        "position_id",
        "symbol",
        "side",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "volume",
        "avg_volume_used",
        "actual_net_pnl",
        "actual_avgvol_pnl",
        "sim_net_pnl",
        "sim_avgvol_pnl",
        "actual_trade_won",
        "sim_trade_won",
        "actual_entry_revisit_mae_r",
        "actual_entry_revisit_mfe_r",
        "sim_entry_revisit_mae_r",
        "sim_entry_revisit_mfe_r",
    ]


def _prepare_trade_pnl_export(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    for col in ["avg_volume_used", "actual_avgvol_pnl", "sim_avgvol_pnl"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)

    cols = [c for c in _trade_pnl_export_columns() if c in out.columns]
    return out.loc[:, cols]


def _prepare_symbol_summary_export(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if "symbol" in out.columns:
        cols = ["symbol"] + [c for c in out.columns if c != "symbol"]
        out = out.loc[:, cols]
    return out


def _excursion_stats_from_path(
    side: int,
    entry_price: float,
    highs: np.ndarray,
    lows: np.ndarray,
    tp_ratio: float,
    sl_ratio: float,
) -> Dict[str, Any]:
    """Pre-resolution MAE/MFE.

    This version does not use the old entry-revisit gate.
    It measures the maximum adverse/favorable excursion on the path that the
    caller already sliced to end at the first resolution event:
    - TP hit
    - SL hit
    - or censor/time exit

    That fixes:
    - Case B: almost hit TP, then reversed to SL
    - Case C: TP hit, then later price moved again (post-resolution bars are
      not included because the caller should stop the path at the first exit)
    """
    nan_out = {"mae_r": np.nan, "mfe_r": np.nan, "had_reset": False}

    if not np.isfinite(entry_price) or entry_price <= 0.0:
        return nan_out
    if tp_ratio <= 0.0 or sl_ratio <= 0.0:
        return nan_out
    if highs.size == 0 or lows.size == 0:
        return nan_out

    n = int(min(len(highs), len(lows)))
    if n <= 0:
        return nan_out

    highs = np.asarray(highs[:n], dtype=np.float64)
    lows = np.asarray(lows[:n], dtype=np.float64)

    if side == 1:
        mae_raw = np.maximum(0.0, entry_price - lows)
        mfe_raw = np.maximum(0.0, highs - entry_price)
    else:
        mae_raw = np.maximum(0.0, highs - entry_price)
        mfe_raw = np.maximum(0.0, entry_price - lows)

    if mae_raw.size == 0 or mfe_raw.size == 0:
        return nan_out

    mae_max_raw = float(np.nanmax(mae_raw))
    mfe_max_raw = float(np.nanmax(mfe_raw))

    mae_r = (mae_max_raw / entry_price) / sl_ratio if sl_ratio > 0.0 else np.nan
    mfe_r = (mfe_max_raw / entry_price) / tp_ratio if tp_ratio > 0.0 else np.nan

    return {
        "mae_r": float(mae_r) if np.isfinite(mae_r) else np.nan,
        "mfe_r": float(mfe_r) if np.isfinite(mfe_r) else np.nan,
        "had_reset": True,
    }


def _summary_stats(df: pd.DataFrame) -> Dict[str, float]:
    trades = len(df)
    if trades == 0:
        return {
            "trades": 0,
            "actual_win_rate": np.nan,
            "sim_win_rate": np.nan,
            "actual_net_expectancy": np.nan,
            "sim_net_expectancy": np.nan,
            "actual_avgvol_net_expectancy": np.nan,
            "sim_avgvol_net_expectancy": np.nan,
        }

    actual_wins = pd.to_numeric(df["actual_trade_won"], errors="coerce").fillna(0.0)
    sim_wins = pd.to_numeric(df["sim_trade_won"], errors="coerce").fillna(0.0)

    actual_pnl = pd.to_numeric(df["actual_net_pnl"], errors="coerce").fillna(0.0)
    sim_pnl = pd.to_numeric(df["sim_net_pnl"], errors="coerce").fillna(0.0)

    actual_avgvol_pnl = pd.to_numeric(df["actual_avgvol_pnl"], errors="coerce").fillna(0.0)
    sim_avgvol_pnl = pd.to_numeric(df["sim_avgvol_pnl"], errors="coerce").fillna(0.0)

    return {
        "trades": int(trades),
        "actual_win_rate": float(actual_wins.mean() * 100.0),
        "sim_win_rate": float(sim_wins.mean() * 100.0),
        "actual_net_expectancy": float(actual_pnl.mean()),
        "sim_net_expectancy": float(sim_pnl.mean()),
        "actual_avgvol_net_expectancy": float(actual_avgvol_pnl.mean()),
        "sim_avgvol_net_expectancy": float(sim_avgvol_pnl.mean()),
    }


def run() -> pd.DataFrame:
    init_mt5()
    try:
        deals_df = load_deals(HISTORY_DAYS)
        all_trades_ever = closed_positions_from_deals(deals_df)
        save_trade_history(deals_df, all_trades_ever)

        analysis_trades = all_trades_ever[all_trades_ever["symbol"].isin(SYMBOL_RULES.keys())].copy()
        analysis_trades = analysis_trades.sort_values(["entry_time", "position_id"]).reset_index(drop=True)
        trades_df = filter_last_n_per_symbol(analysis_trades)
        if trades_df.empty:
            return pd.DataFrame()

        bars_cache = build_bars_cache(trades_df)
        avg_volume_map = all_trades_ever.groupby("symbol")["volume"].mean().to_dict()
        results: List[Dict[str, Any]] = []

        for _, tr in trades_df.iterrows():
            symbol = str(tr["symbol"])
            side = int(tr["side"])
            rule = trade_rule_for(symbol, side)
            symbol_bars = bars_cache.get(symbol)
            if symbol_bars is None or symbol_bars.empty or rule is None:
                continue

            if not all(k in rule for k in ("tp_pct", "sl_pct", "spread_pct")):
                continue

            entry_time = ensure_utc(tr["entry_time"])
            exit_time = ensure_utc(tr["exit_time"])
            entry_floor = floor_to_m5(entry_time)
            entry_px = float(tr["entry_price"])
            vol = float(tr["volume"])
            avg_vol = float(avg_volume_map.get(symbol, vol))

            tp_ratio = pct_to_ratio(rule["tp_pct"])
            sl_ratio = pct_to_ratio(rule["sl_pct"])

            sim = simulate_trade_m5_avg_volume(
                symbol,
                side,
                vol,
                avg_vol,
                entry_time,
                entry_px,
                tr["gross_profit"],
                tr["commission"],
                tr["swap"],
                tr["fee"],
                rule["tp_pct"],
                rule["sl_pct"],
                rule["spread_pct"],
                symbol_bars,
            )

            act_path = symbol_bars[(symbol_bars["time"] >= entry_floor) & (symbol_bars["time"] <= exit_time)].copy()
            act_exc = _excursion_stats_from_path(
                side,
                entry_px,
                act_path["high"].values,
                act_path["low"].values,
                tp_ratio,
                sl_ratio,
            )

            sim_exit_t = ensure_utc(sim["sim_exit_time"]) if pd.notna(sim["sim_exit_time"]) else exit_time
            sim_path = symbol_bars[(symbol_bars["time"] >= entry_floor) & (symbol_bars["time"] <= sim_exit_t)].copy()
            sim_exc = _excursion_stats_from_path(
                side,
                entry_px,
                sim_path["high"].values,
                sim_path["low"].values,
                tp_ratio,
                sl_ratio,
            )

            actual_won = _trade_won_by_price(side, entry_px, float(tr["exit_price"]))

            results.append(
                {
                    "position_id": tr["position_id"],
                    "symbol": symbol,
                    "side": side,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": entry_px,
                    "exit_price": float(tr["exit_price"]),
                    "volume": vol,
                    "avg_volume_used": avg_vol,
                    "actual_net_pnl": float(tr["net_actual_pnl"]),
                    "actual_avgvol_pnl": float(tr["net_actual_pnl"]) * (avg_vol / vol) if vol > 0 else np.nan,
                    "sim_net_pnl": sim["sim_net_pnl"],
                    "sim_avgvol_pnl": sim["sim_avgvol_pnl"],
                    "actual_trade_won": actual_won,
                    "sim_trade_won": sim["sim_close_type"] == "tp",
                    "actual_entry_revisit_mae_r": round_price(act_exc["mae_r"]),
                    "actual_entry_revisit_mfe_r": round_price(act_exc["mfe_r"]),
                    "sim_entry_revisit_mae_r": round_price(sim_exc["mae_r"]),
                    "sim_entry_revisit_mfe_r": round_price(sim_exc["mfe_r"]),
                }
            )

        res_df = pd.DataFrame(results)
        if res_df.empty:
            return res_df

        pnl_export_df = _prepare_trade_pnl_export(res_df)
        append_csv_dedup(pnl_export_df, PNL_DETAIL_FILE, ["position_id"])

        exc_export_cols = [
            "position_id",
            "symbol",
            "side",
            "entry_time",
            "exit_time",
            "actual_trade_won",
            "sim_trade_won",
            "actual_entry_revisit_mae_r",
            "actual_entry_revisit_mfe_r",
            "sim_entry_revisit_mae_r",
            "sim_entry_revisit_mfe_r",
        ]
        append_csv_dedup(res_df.reindex(columns=exc_export_cols), EXCURSION_DETAIL_FILE, ["position_id"])

        ov = _summary_stats(res_df)
        ov.update(
            {
                "actual_total_net_pnl": float(pd.to_numeric(res_df["actual_net_pnl"], errors="coerce").fillna(0.0).sum()),
                "sim_total_net_pnl": float(pd.to_numeric(res_df["sim_net_pnl"], errors="coerce").fillna(0.0).sum()),
                "actual_total_avgvol_net_pnl": float(pd.to_numeric(res_df["actual_avgvol_pnl"], errors="coerce").fillna(0.0).sum()),
                "sim_total_avgvol_net_pnl": float(pd.to_numeric(res_df["sim_avgvol_pnl"], errors="coerce").fillna(0.0).sum()),
            }
        )
        _round_summary_numbers(pd.DataFrame([ov])).to_csv(OVERVIEW_FILE, index=False)

        s_rows = []
        for sym, g in res_df.groupby("symbol"):
            stats = _summary_stats(g)
            stats.update(
                {
                    "symbol": sym,
                    "avg_vol": float(pd.to_numeric(g["avg_volume_used"], errors="coerce").mean()),
                    "actual_avg_vol": float(pd.to_numeric(g["volume"], errors="coerce").mean()),
                    "actual_total_net_pnl": float(pd.to_numeric(g["actual_net_pnl"], errors="coerce").fillna(0.0).sum()),
                    "sim_total_net_pnl": float(pd.to_numeric(g["sim_net_pnl"], errors="coerce").fillna(0.0).sum()),
                    "actual_total_avgvol_net_pnl": float(pd.to_numeric(g["actual_avgvol_pnl"], errors="coerce").fillna(0.0).sum()),
                    "sim_total_avgvol_net_pnl": float(pd.to_numeric(g["sim_avgvol_pnl"], errors="coerce").fillna(0.0).sum()),
                }
            )
            s_rows.append(stats)

        summary_export = _prepare_symbol_summary_export(pd.DataFrame(s_rows))
        _round_summary_numbers(summary_export).to_csv(SUMMARY_FILE, index=False)

        return res_df
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run()
