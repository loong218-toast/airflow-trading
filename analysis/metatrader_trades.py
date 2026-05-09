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
SYMBOL_RULES: Dict[str, Dict[str, float]] = {
    "UK100": {"tp_pct": 0.07, "sl_pct": 0.18, "spread_pct": 0.01, "last_n_trades": 15},
    "AUDJPY": {"tp_pct": 0.15, "sl_pct": 0.20, "spread_pct": 0.01, "last_n_trades": 10},
    "USDCHF": {"tp_pct": 0.25, "sl_pct": 0.60, "spread_pct": 0.02, "last_n_trades": -1},
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

def _actual_exit_ratio(entry_price: float, exit_price: float) -> float:
    if not np.isfinite(entry_price) or entry_price <= 0.0 or not np.isfinite(exit_price):
        return np.nan
    return abs(float(exit_price) - float(entry_price)) / float(entry_price)


def append_csv_dedup(df: pd.DataFrame, file_path: Path, key_cols: List[str]) -> None:
    if df.empty:
        return

    if file_path.exists():
        old = pd.read_csv(file_path)
        combined = pd.concat([old, df], ignore_index=True)
    else:
        combined = df.copy()

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

def _excursion_stats_from_path(
    side: int,
    entry_price: float,
    highs: np.ndarray,
    lows: np.ndarray,
    tp_ratio: float,
    sl_ratio: float,
    exclude_last_bar: bool = True,
) -> Dict[str, float]:
    nan_out = {"pre_reset_mae_r": np.nan, "pre_reset_tp_r": np.nan, "had_reset": False}

    if not np.isfinite(entry_price) or entry_price <= 0.0 or highs.size < 2:
        return nan_out

    # Horizon: All bars except the very last one
    lookback_limit = len(highs) - 1 

    eligible_mae_raw = []
    eligible_mfe_raw = []

    for i in range(lookback_limit):
        # 1. Identify Reset (Did price return to entry at index i or any time after?)
        if side == 1: # BUY
            has_reset = any(highs[i:lookback_limit] >= entry_price)
            current_mae_raw = entry_price - lows[i]
            current_mfe_raw = highs[i] - entry_price
        else: # SELL
            # For SELL, reset is price dropping back DOWN to entry
            has_reset = any(lows[i:lookback_limit] <= entry_price)
            current_mae_raw = highs[i] - entry_price
            current_mfe_raw = entry_price - lows[i]

        if has_reset:
            eligible_mae_raw.append(max(0.0, current_mae_raw))
            eligible_mfe_raw.append(max(0.0, current_mfe_raw))

    if not eligible_mae_raw:
        return nan_out

    # Pick the best (max) from all eligible bars
    return {
        "pre_reset_mae_r": float((max(eligible_mae_raw) / entry_price) / sl_ratio) if sl_ratio > 0 else np.nan,
        "pre_reset_tp_r": float((max(eligible_mfe_raw) / entry_price) / tp_ratio) if tp_ratio > 0 else np.nan,
        "had_reset": True
    }


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

    bar_outcome = "not_hit_in_data"
    sim_exit_price_used = np.nan
    sim_close_type = "not_hit_in_data"
    sim_exit_time = pd.NaT

    if not replay.empty:
        for _, bar in replay.iterrows():
            open_px = float(bar["open"])
            close_px = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])

            same_open_close = abs(open_px - close_px) <= 1e-12

            if side == 1:
                tp_hit = high >= tp_price
                sl_hit = low <= sl_price
            else:
                tp_hit = low <= tp_price
                sl_hit = high >= sl_price

            if tp_hit and sl_hit:
                bar_outcome = "sl_same_bar"
                sim_exit_price_used = sl_price
                sim_close_type = "sl"
                sim_exit_time = pd.Timestamp(bar["time"])
                break

            if tp_hit:
                bar_outcome = "tp_first"
                sim_exit_price_used = tp_price
                sim_close_type = "tp"
                sim_exit_time = pd.Timestamp(bar["time"])
                break

            if sl_hit:
                bar_outcome = "sl_first"
                sim_exit_price_used = sl_price
                sim_close_type = "sl"
                sim_exit_time = pd.Timestamp(bar["time"])
                break

            if same_open_close and (tp_hit or sl_hit):
                bar_outcome = "sl_same_bar"
                sim_exit_price_used = sl_price
                sim_close_type = "sl"
                sim_exit_time = pd.Timestamp(bar["time"])
                break

    sim_gross_profit = np.nan
    sim_net_pnl = np.nan
    net_delta = np.nan
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
        net_delta = sim_net_pnl - (gross_profit_actual + commission + swap + fee)

    return {
        "ref_entry_price": round_price(ref_entry),
        "tp_price": round_price(tp_price),
        "sl_price": round_price(sl_price),
        "sim_exit_price_used": round_price(sim_exit_price_used),
        "sim_exit_time": sim_exit_time,
        "sim_close_type": sim_close_type,
        "bar_outcome": bar_outcome,
        "sim_pnl_pct": round_pct(sim_pnl_pct),
        "sim_gross_profit": round_price(sim_gross_profit),
        "sim_net_pnl": round_price(sim_net_pnl),
        "net_delta": round_price(net_delta),
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
            "sim_avgvol_net_pnl": np.nan,
            "net_delta_avgvol": np.nan,
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

    sim_avgvol_net_pnl = sim_avgvol_gross_profit + avg_commission + avg_swap + avg_fee
    net_delta_avgvol = sim_avgvol_net_pnl - (gross_profit_actual + commission + swap + fee)

    return {
        **base,
        "avg_volume_used": round_price(avg_volume),
        "sim_avgvol_gross_profit": round_price(sim_avgvol_gross_profit),
        "sim_avgvol_net_pnl": round_price(sim_avgvol_net_pnl),
        "net_delta_avgvol": round_price(net_delta_avgvol),
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

# =========================
# SAVE HELPERS
# =========================

def _build_trade_table_columns() -> tuple[list[str], list[str]]:
    pnl_cols = [
        "position_id",
        "symbol",
        "side",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "volume",
        "avg_volume_used",
        "tp_pct",
        "sl_pct",
        "spread_pct",
        "actual_net_pnl",
        "actual_trade_won",
        "actual_tp_pct",
        "actual_sl_pct",
        "sim_net_pnl",
        "sim_avgvol_net_pnl",
        "sim_trade_won",
        "sim_tp_pct",
        "sim_sl_pct",
        "net_delta",
        "net_delta_avgvol",
        "sim_close_type",
        "bar_outcome",
    ]

    excursion_cols = [
        "position_id",
        "symbol",
        "side",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "volume",
        "avg_volume_used",
        "tp_pct",
        "sl_pct",
        "spread_pct",
        "actual_trade_won",
        "sim_trade_won",
        "actual_pre_reset_mae_r",
        "actual_pre_reset_mfe_r",
        "actual_post_reset_mae_r",
        "actual_post_reset_mfe_r",
        "actual_mae_win_r",
        "actual_mae_loss_r",
        "actual_mfe_win_r",
        "actual_mfe_loss_r",
        "sim_pre_reset_mae_r",
        "sim_pre_reset_mfe_r",
        "sim_post_reset_mae_r",
        "sim_post_reset_mfe_r",
        "sim_mae_win_r",
        "sim_mae_loss_r",
        "sim_mfe_win_r",
        "sim_mfe_loss_r",
        "sim_close_type",
        "bar_outcome",
    ]

    return pnl_cols, excursion_cols


# =========================
# MAIN
# =========================

def run() -> pd.DataFrame:
    init_mt5()
    try:
        deals_df = load_deals(HISTORY_DAYS)
        if deals_df.empty:
            print("No deals found.")
            return pd.DataFrame()

        trades_df = closed_positions_from_deals(deals_df)
        if trades_df.empty:
            print("No closed positions found.")
            return pd.DataFrame()

        trades_df = filter_last_n_per_symbol(trades_df)
        save_trade_history(deals_df, trades_df)

        bars_cache = build_bars_cache(trades_df)

        avg_volume_by_symbol = (
            trades_df.groupby("symbol")["volume"].mean().to_dict()
            if not trades_df.empty and "volume" in trades_df.columns
            else {}
        )

        def _metrics_for_path(
            path: pd.DataFrame,
            side: int,
            entry_price: float,
            tp_ratio: float,
            sl_ratio: float,
            exclude_last_bar: bool,
        ) -> Dict[str, float]:
            if path.empty:
                return {
                    "pre_reset_mae_r": np.nan,
                    "pre_reset_tp_r": np.nan,
                    "post_reset_mae_r": np.nan,
                    "post_reset_tp_r": np.nan,
                    "first_reset_idx": np.nan,
                    "had_reset": False,
                }

            highs = path["high"].to_numpy(dtype=np.float64, copy=False)
            lows = path["low"].to_numpy(dtype=np.float64, copy=False)

            return _excursion_stats_from_path(
                side=side,
                entry_price=entry_price,
                highs=highs,
                lows=lows,
                tp_ratio=tp_ratio,
                sl_ratio=sl_ratio,
                exclude_last_bar=exclude_last_bar,
            )

        results: List[Dict[str, Any]] = []
        skipped_no_rule = 0
        skipped_no_data = 0  # <--- Add this counter

        for idx, tr in trades_df.iterrows():
            if idx % 10 == 0:
                print(f"Processing trade {idx + 1}/{len(trades_df)}...")

            symbol = str(tr["symbol"])
            
            # --- ADD THIS SKIP LOGIC HERE ---
            symbol_bars = bars_cache.get(symbol)
            if symbol_bars is None or symbol_bars.empty:
                print(f"Skipping {symbol} (Trade {tr.get('position_id')}): No M5 bars found in MT5 history.")
                skipped_no_data += 1
                continue
            # --------------------------------

            rule = rule_for(symbol)
            if rule is None:
                skipped_no_rule += 1
                continue

            entry_time = ensure_utc(tr["entry_time"])
            exit_time = ensure_utc(tr["exit_time"])
            entry_price = float(tr["entry_price"])
            exit_price = float(tr["exit_price"])
            side = int(tr["side"])
            volume = float(tr["volume"]) if pd.notna(tr["volume"]) else np.nan
            avg_volume = float(avg_volume_by_symbol.get(symbol, volume))

            tp_ratio = pct_to_ratio(rule["tp_pct"])
            sl_ratio = pct_to_ratio(rule["sl_pct"])

            actual_net_pnl = float(tr["net_actual_pnl"])

            # Price-based win/loss for excursion masking
            actual_trade_won = _trade_won_by_price(
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
            )

            # Actual trade excursion: normalize by the actual entry->exit price distance
            actual_exit_ratio = _actual_exit_ratio(entry_price, exit_price)

            # 1. Calculate Pure Price Distance (Denominator)
            # This ensures (114.406 - 114.252) = 0.154 is your 1.0R benchmark
            actual_price_dist = abs(exit_price - entry_price)
            actual_price_ratio = actual_price_dist / entry_price if entry_price > 0 else 0.0001
            if actual_price_ratio == 0: actual_price_ratio = 0.0001

            actual_path = bars_cache.get(symbol, pd.DataFrame())
            
            # --- ADD THIS SAFETY CHECK ---
            if not actual_path.empty and "time" in actual_path.columns:
                actual_path = actual_path[
                    (actual_path["time"] >= entry_time) & (actual_path["time"] <= exit_time)
                ].copy()
            else:
                actual_path = pd.DataFrame() # Ensure it's an empty DF if no data
            # ------------------------------
            actual_path = actual_path[
                (actual_path["time"] >= entry_time) & (actual_path["time"] <= exit_time)
            ].copy()

            # 2. Compute Actual Metrics 
            # We set exclude_last_bar=True to ensure the "Full SL" move isn't counted as an "Almost SL"
            actual_metrics = _metrics_for_path(
                path=actual_path,
                side=side,
                entry_price=entry_price,
                tp_ratio=actual_price_ratio, # Denominator: 0.154
                sl_ratio=actual_price_ratio, # Denominator: 0.154
                exclude_last_bar=True,       # IMPORTANT: Separates "Full SL" from "Almost SL"
            )

            if tr["position_id"] == 545210010:
                print(f"--- DETAILED MATH CHECK ---")
                # Look at the path excluding the last bar
                path_to_check = actual_path.iloc[:-1]
                
                for i, row in path_to_check.iterrows():
                    if row['high'] >= 114.37: # Check peaks near your area
                        # Check if a reset happened AFTER this bar but before the end
                        future_lows = path_to_check.iloc[i:]['low']
                        reset_found = any(future_lows <= 114.252)
                        print(f"Bar Time: {row['time']}, High: {row['high']}, Reset Found: {reset_found}")

            sim = simulate_trade_m5_avg_volume(
                symbol=symbol,
                side=side,
                volume=volume,
                avg_volume=avg_volume,
                entry_time=entry_time,
                entry_price=entry_price,
                gross_profit_actual=float(tr["gross_profit"]),
                commission=float(tr["commission"]),
                swap=float(tr["swap"]),
                fee=float(tr["fee"]),
                tp_pct=rule["tp_pct"],
                sl_pct=rule["sl_pct"],
                spread_pct=rule["spread_pct"],
                bars=bars_cache.get(symbol, pd.DataFrame()),
            )

            sim_trade_won = bool(sim["sim_close_type"] == "tp")

            actual_tp_pct, actual_sl_pct = _infer_tp_sl_from_exit(
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
            )

            sim_exit_price = sim.get("sim_exit_price_used", np.nan)
            sim_tp_pct, sim_sl_pct = _infer_tp_sl_from_exit(
                side=side,
                entry_price=entry_price,
                exit_price=float(sim_exit_price) if pd.notna(sim_exit_price) else np.nan,
            )

            sim_metrics = {
                "pre_reset_mae_r": np.nan,
                "pre_reset_tp_r": np.nan,
                "post_reset_mae_r": np.nan,
                "post_reset_tp_r": np.nan,
                "first_reset_idx": np.nan,
                "had_reset": False,
            }

            if sim["sim_close_type"] in {"tp", "sl"} and not pd.isna(sim["sim_exit_time"]):
                sim_path = bars_cache.get(symbol, pd.DataFrame())
                
                # --- ADD THIS SAFETY CHECK ---
                if not sim_path.empty and "time" in sim_path.columns:
                    sim_path = sim_path[
                        (sim_path["time"] >= entry_time) &
                        (sim_path["time"] <= ensure_utc(sim["sim_exit_time"]))
                    ].copy()
                else:
                    sim_path = pd.DataFrame()
                # ------------------------------

                sim_metrics = _metrics_for_path(
                    path=sim_path,
                    side=side,
                    entry_price=entry_price,
                    tp_ratio=tp_ratio,
                    sl_ratio=sl_ratio,
                    exclude_last_bar=True,
                )

            actual_pre_reset_mae_r = np.nan
            actual_pre_reset_tp_r = np.nan
            if actual_metrics["had_reset"]:
                if actual_trade_won: # If trade was a Win, show MFE (TP R)
                    actual_pre_reset_tp_r = round_price(actual_metrics["pre_reset_tp_r"])
                else: # If trade was a Loss, show MAE
                    actual_pre_reset_mae_r = round_price(actual_metrics["pre_reset_mae_r"])

            # --- Logic for SIM (Show Both) ---
            sim_pre_reset_mae_r = round_price(sim_metrics.get("pre_reset_mae_r", np.nan))
            sim_pre_reset_tp_r = round_price(sim_metrics.get("pre_reset_tp_r", np.nan))

            results.append(
                {
                    "position_id": int(tr["position_id"]) if pd.notna(tr.get("position_id", np.nan)) else np.nan,
                    "symbol": symbol,
                    "side": side,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "volume": volume,
                    "avg_volume_used": avg_volume,
                    "tp_pct": rule["tp_pct"],
                    "sl_pct": rule["sl_pct"],
                    "spread_pct": rule["spread_pct"],
                    "actual_net_pnl": actual_net_pnl,
                    "actual_trade_won": actual_trade_won,
                    "actual_tp_pct": actual_tp_pct,
                    "actual_sl_pct": actual_sl_pct,

                    "actual_pre_reset_mae_r": actual_pre_reset_mae_r,
                    "actual_pre_reset_tp_r": actual_pre_reset_tp_r,

                    # Sim: keep both
                    "sim_pre_reset_mae_r": round_price(_safe_metric(sim_metrics, "pre_reset_mae_r")),
                    "sim_pre_reset_tp_r": round_price(_safe_metric(sim_metrics, "pre_reset_tp_r")),
                    "sim_post_reset_mae_r": round_price(_safe_metric(sim_metrics, "post_reset_mae_r")),
                    "sim_post_reset_tp_r": round_price(_safe_metric(sim_metrics, "post_reset_tp_r")),

                    "sim_net_pnl": sim["sim_net_pnl"],
                    "sim_avgvol_net_pnl": sim["sim_avgvol_net_pnl"],
                    "sim_trade_won": sim_trade_won,
                    "sim_tp_pct": sim_tp_pct,
                    "sim_sl_pct": sim_sl_pct,
                    "net_delta": sim["net_delta"],
                    "net_delta_avgvol": sim["net_delta_avgvol"],
                    "sim_close_type": sim["sim_close_type"],
                    "bar_outcome": sim["bar_outcome"],
                }
            )

        out = pd.DataFrame(results)
        if out.empty:
            print("No results produced.")
            return out

        pnl_cols = [
            "position_id",
            "symbol",
            "side",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "volume",
            "avg_volume_used",
            "tp_pct",
            "sl_pct",
            "spread_pct",
            "actual_net_pnl",
            "actual_trade_won",
            "actual_tp_pct",
            "actual_sl_pct",
            "sim_net_pnl",
            "sim_avgvol_net_pnl",
            "sim_trade_won",
            "sim_tp_pct",
            "sim_sl_pct",
            "net_delta",
            "net_delta_avgvol",
            "sim_close_type",
            "bar_outcome",
        ]

        excursion_cols = [
            "position_id",
            "symbol",
            "side",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "volume",
            "avg_volume_used",
            "tp_pct",
            "sl_pct",
            "spread_pct",
            "actual_trade_won",
            "sim_trade_won",
            "actual_pre_reset_mae_r",
            "actual_pre_reset_tp_r",
            "sim_pre_reset_mae_r",
            "sim_pre_reset_tp_r",
            "sim_post_reset_mae_r",
            "sim_post_reset_tp_r",
            "sim_close_type",
            "bar_outcome",
        ]

        pnl_df = out[[c for c in pnl_cols if c in out.columns]].copy()
        excursion_df = out[[c for c in excursion_cols if c in out.columns]].copy()

        pnl_df = pnl_df.sort_values(["symbol", "entry_time", "position_id"]).reset_index(drop=True)
        excursion_df = excursion_df.sort_values(["symbol", "entry_time", "position_id"]).reset_index(drop=True)

        for col in [
            "volume",
            "avg_volume_used",
            "actual_net_pnl",
            "actual_tp_pct",
            "actual_sl_pct",
            "sim_net_pnl",
            "sim_avgvol_net_pnl",
            "sim_tp_pct",
            "sim_sl_pct",
            "net_delta",
            "net_delta_avgvol",
            "actual_pre_reset_mae_r",
            "actual_pre_reset_tp_r",
            "sim_pre_reset_mae_r",
            "sim_pre_reset_tp_r",
            "sim_post_reset_mae_r",
            "sim_post_reset_tp_r",
        ]:
            if col in pnl_df.columns:
                pnl_df[col] = pnl_df[col].apply(round_price)
            if col in excursion_df.columns:
                excursion_df[col] = excursion_df[col].apply(round_price)

        for col in ["tp_pct", "sl_pct", "spread_pct"]:
            if col in pnl_df.columns:
                pnl_df[col] = pnl_df[col].apply(round_pct)
            if col in excursion_df.columns:
                excursion_df[col] = excursion_df[col].apply(round_pct)

        pnl_df.to_csv(PNL_DETAIL_FILE, index=False)
        excursion_df.to_csv(EXCURSION_DETAIL_FILE, index=False)

        resolved = pnl_df["sim_close_type"].isin(["tp", "sl"])
        unresolved = pnl_df["sim_close_type"].eq("not_hit_in_data")

        overall = {
            "trades": int(len(pnl_df)),
            "actual_avg_tp_pct": round_pct(pnl_df["actual_tp_pct"].dropna().mean()),
            "actual_avg_sl_pct": round_pct(pnl_df["actual_sl_pct"].dropna().mean()),
            "sim_avg_tp_pct": round_pct(pnl_df["sim_tp_pct"].dropna().mean()),
            "sim_avg_sl_pct": round_pct(pnl_df["sim_sl_pct"].dropna().mean()),
            "resolved_sim_trades": int(resolved.sum()),
            "unresolved_sim_trades": int(unresolved.sum()),
            "actual_total_net_pnl": round_price(pnl_df["actual_net_pnl"].sum()),
            "sim_total_net_pnl": round_price(pnl_df.loc[resolved, "sim_net_pnl"].sum()),
            "sim_avgvol_total_net_pnl": round_price(pnl_df.loc[resolved, "sim_avgvol_net_pnl"].sum()),
            "total_net_delta": round_price(pnl_df.loc[resolved, "net_delta"].sum()),
            "total_net_delta_avgvol": round_price(pnl_df.loc[resolved, "net_delta_avgvol"].sum()),
            "avg_volume_used": round_price(pnl_df["avg_volume_used"].mean()),
            "actual_avg_net_pnl": round_price(pnl_df["actual_net_pnl"].mean()),
            "sim_avg_net_pnl": round_price(pnl_df.loc[resolved, "sim_net_pnl"].mean()),
            "sim_avgvol_avg_net_pnl": round_price(pnl_df.loc[resolved, "sim_avgvol_net_pnl"].mean()),
            "actual_avg_pre_reset_mae_r": round_price(excursion_df["actual_pre_reset_mae_r"].dropna().mean()),
            "actual_avg_pre_reset_tp_r": round_price(excursion_df["actual_pre_reset_tp_r"].dropna().mean()),
            "sim_avg_pre_reset_mae_r": round_price(excursion_df["sim_pre_reset_mae_r"].dropna().mean()),
            "sim_avg_pre_reset_tp_r": round_price(excursion_df["sim_pre_reset_tp_r"].dropna().mean()),
            "sim_avg_post_reset_mae_r": round_price(excursion_df["sim_post_reset_mae_r"].dropna().mean()),
            "sim_avg_post_reset_tp_r": round_price(excursion_df["sim_post_reset_tp_r"].dropna().mean()),
            "tp_hits": int((pnl_df["bar_outcome"].isin(["tp_first", "tp_same_bar"])).sum()),
            "sl_hits": int((pnl_df["bar_outcome"].isin(["sl_first", "sl_same_bar"])).sum()),
            "not_hit_in_data": int((pnl_df["sim_close_type"] == "not_hit_in_data").sum()),
        }

        overall_df = pd.DataFrame([overall])
        overall_df.to_csv(OVERVIEW_FILE, index=False)

        summary_rows = []
        for sym, g in pnl_df.groupby("symbol"):
            resolved_g = g["sim_close_type"].isin(["tp", "sl"])
            exc_g = excursion_df[excursion_df["symbol"] == sym].copy()

            summary_rows.append(
                {
                    "symbol": sym,
                    "trades": int(len(g)),
                    "resolved_sim_trades": int(resolved_g.sum()),
                    "unresolved_sim_trades": int((g["sim_close_type"] == "not_hit_in_data").sum()),
                    "avg_volume_used": round_price(g["avg_volume_used"].mean()),
                    "actual_total_net_pnl": round_price(g["actual_net_pnl"].sum()),
                    "sim_total_net_pnl": round_price(g.loc[resolved_g, "sim_net_pnl"].sum()),
                    "sim_avgvol_total_net_pnl": round_price(g.loc[resolved_g, "sim_avgvol_net_pnl"].sum()),
                    "total_net_delta": round_price(g.loc[resolved_g, "net_delta"].sum()),
                    "total_net_delta_avgvol": round_price(g.loc[resolved_g, "net_delta_avgvol"].sum()),
                    "actual_avg_net_pnl": round_price(g["actual_net_pnl"].mean()),
                    "sim_avg_net_pnl": round_price(g.loc[resolved_g, "sim_net_pnl"].mean()),
                    "sim_avgvol_avg_net_pnl": round_price(g.loc[resolved_g, "sim_avgvol_net_pnl"].mean()),
                    "actual_avg_pre_reset_mae_r": round_price(exc_g["actual_pre_reset_mae_r"].dropna().mean()) if not exc_g.empty else np.nan,
                    "actual_avg_pre_reset_tp_r": round_price(exc_g["actual_pre_reset_tp_r"].dropna().mean()) if not exc_g.empty else np.nan,
                    "sim_avg_pre_reset_mae_r": round_price(exc_g["sim_pre_reset_mae_r"].dropna().mean()) if not exc_g.empty else np.nan,
                    "sim_avg_pre_reset_tp_r": round_price(exc_g["sim_pre_reset_tp_r"].dropna().mean()) if not exc_g.empty else np.nan,
                    "sim_avg_post_reset_mae_r": round_price(exc_g["sim_post_reset_mae_r"].dropna().mean()) if not exc_g.empty else np.nan,
                    "sim_avg_post_reset_tp_r": round_price(exc_g["sim_post_reset_tp_r"].dropna().mean()) if not exc_g.empty else np.nan,
                    "tp_hits": int((g["bar_outcome"].isin(["tp_first", "tp_same_bar"])).sum()),
                    "sl_hits": int((g["bar_outcome"].isin(["sl_first", "sl_same_bar"])).sum()),
                    "not_hit_in_data": int((g["sim_close_type"] == "not_hit_in_data").sum()),
                    "tp_pct": round_pct(g["tp_pct"].iloc[0]),
                    "sl_pct": round_pct(g["sl_pct"].iloc[0]),
                    "spread_pct": round_pct(g["spread_pct"].iloc[0]),
                    "last_n_trades": int(symbol_last_n(sym)),
                }
            )

        summary = pd.DataFrame(summary_rows).sort_values("symbol").reset_index(drop=True)
        summary.to_csv(SUMMARY_FILE, index=False)

        print("\n=== OVERALL SUMMARY ===")
        print(overall_df.to_string(index=False))

        print("\n=== PER SYMBOL SUMMARY ===")
        print(summary.to_string(index=False))

        print("\n=== TRADE HISTORY SAVED ===")
        print(f"Raw deals file         : {RAW_DEALS_FILE}")
        print(f"Closed trades file     : {CLOSED_TRADES_FILE}")
        print(f"Overall summary        : {OVERVIEW_FILE}")
        print(f"Symbol summary         : {SUMMARY_FILE}")
        print(f"PnL detail file        : {PNL_DETAIL_FILE}")
        print(f"Excursion detail file  : {EXCURSION_DETAIL_FILE}")
        print(f"Skipped no rule        : {skipped_no_rule}")
        print(f"Skipped (No Bar Data)  : {skipped_no_data}")

        return out

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run()
