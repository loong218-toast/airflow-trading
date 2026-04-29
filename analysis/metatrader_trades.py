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
    "UK100": {"tp_pct": 0.07, "sl_pct": 0.20, "spread_pct": 0.01, "last_n_trades": 20},
    "AUDJPY": {"tp_pct": 0.06, "sl_pct": 0.14, "spread_pct": 0.01, "last_n_trades": 10},
    "USDCHF": {"tp_pct": 0.08, "sl_pct": 0.25, "spread_pct": 0.02, "last_n_trades": -1},
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
DETAIL_FILE = ANALYSIS_DIR / "trade_level_summary.csv"


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


def pct_to_ratio(pct_value: float) -> float:
    return float(pct_value) / 100.0


def round_price(x: Any) -> Any:
    if pd.isna(x):
        return x
    return round(float(x), 2)


def round_pct(x: Any) -> Any:
    if pd.isna(x):
        return x
    return round(float(x), 2)


def rule_for(symbol: str) -> Optional[Dict[str, float]]:
    return SYMBOL_RULES.get(symbol)


def symbol_last_n(symbol: str) -> int:
    rule = rule_for(symbol)
    if rule is None:
        return -1
    return int(rule.get("last_n_trades", -1))


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
                # Same-bar ambiguity or doji ambiguity: force SL.
                bar_outcome = "sl_same_bar"
                sim_exit_price_used = sl_price
                sim_close_type = "sl"
                break

            if tp_hit:
                bar_outcome = "tp_first"
                sim_exit_price_used = tp_price
                sim_close_type = "tp"
                break

            if sl_hit:
                bar_outcome = "sl_first"
                sim_exit_price_used = sl_price
                sim_close_type = "sl"
                break

            if same_open_close and (tp_hit or sl_hit):
                bar_outcome = "sl_same_bar"
                sim_exit_price_used = sl_price
                sim_close_type = "sl"
                break

    sim_gross_profit = np.nan
    sim_net_pnl = np.nan
    net_delta = np.nan
    sim_pnl_pct = np.nan
    sim_avgvol_net_pnl = np.nan
    net_delta_avgvol = np.nan

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
        "sim_close_type": sim_close_type,
        "bar_outcome": bar_outcome,
        "sim_pnl_pct": round_pct(sim_pnl_pct),
        "sim_gross_profit": round_price(sim_gross_profit),
        "sim_net_pnl": round_price(sim_net_pnl),
        "net_delta": round_price(net_delta),
        "sim_avgvol_net_pnl": round_price(sim_avgvol_net_pnl),
        "net_delta_avgvol": round_price(net_delta_avgvol),
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

    # Scale volume-linked costs to the average size
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
        for col in ["entry_price", "exit_price", "gross_profit", "commission", "swap", "fee", "net_actual_pnl"]:
            if col in closed.columns:
                closed[col] = closed[col].apply(round_price)
        append_csv_dedup(closed, CLOSED_TRADES_FILE, key_cols=["position_id"])


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

        results: List[Dict[str, Any]] = []
        skipped_no_rule = 0

        for _, tr in trades_df.iterrows():
            symbol = str(tr["symbol"])
            rule = rule_for(symbol)
            if rule is None:
                skipped_no_rule += 1
                continue

            entry_time = ensure_utc(tr["entry_time"])
            entry_price = float(tr["entry_price"])
            exit_price = float(tr["exit_price"])
            side = int(tr["side"])
            volume = float(tr["volume"]) if pd.notna(tr["volume"]) else np.nan
            avg_volume = float(avg_volume_by_symbol.get(symbol, volume))

            actual_net_pnl = float(tr["net_actual_pnl"])

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

            results.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "volume": round_price(volume),
                    "avg_volume_used": round_price(avg_volume),
                    "entry_time": entry_time,
                    "exit_time": ensure_utc(tr["exit_time"]),
                    "tp_pct": round_pct(rule["tp_pct"]),
                    "sl_pct": round_pct(rule["sl_pct"]),
                    "spread_pct": round_pct(rule["spread_pct"]),
                    "actual_net_pnl": round_price(actual_net_pnl),
                    "sim_net_pnl": sim["sim_net_pnl"],
                    "sim_avgvol_net_pnl": sim["sim_avgvol_net_pnl"],
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

        detail_cols = [
            "symbol",
            "side",
            "volume",
            "avg_volume_used",
            "entry_time",
            "exit_time",
            "tp_pct",
            "sl_pct",
            "spread_pct",
            "actual_net_pnl",
            "sim_net_pnl",
            "sim_avgvol_net_pnl",
            "net_delta",
            "net_delta_avgvol",
            "sim_close_type",
            "bar_outcome",
        ]
        detail_cols = [c for c in detail_cols if c in out.columns]
        out = out[detail_cols]

        for col in [
            "volume",
            "avg_volume_used",
            "actual_net_pnl",
            "sim_net_pnl",
            "sim_avgvol_net_pnl",
            "net_delta",
            "net_delta_avgvol",
        ]:
            if col in out.columns:
                out[col] = out[col].apply(round_price)

        for col in ["tp_pct", "sl_pct", "spread_pct"]:
            if col in out.columns:
                out[col] = out[col].apply(round_pct)

        out.to_csv(DETAIL_FILE, index=False)

        resolved = out["sim_close_type"].isin(["tp", "sl"])
        unresolved = out["sim_close_type"].eq("not_hit_in_data")

        overall = {
            "trades": int(len(out)),
            "resolved_sim_trades": int(resolved.sum()),
            "unresolved_sim_trades": int(unresolved.sum()),
            "actual_total_net_pnl": round_price(out["actual_net_pnl"].sum()),
            "sim_total_net_pnl": round_price(out.loc[resolved, "sim_net_pnl"].sum()),
            "sim_avgvol_total_net_pnl": round_price(out.loc[resolved, "sim_avgvol_net_pnl"].sum()),
            "total_net_delta": round_price(out.loc[resolved, "net_delta"].sum()),
            "total_net_delta_avgvol": round_price(out.loc[resolved, "net_delta_avgvol"].sum()),
            "avg_volume_used": round_price(out["avg_volume_used"].mean()),
            "actual_avg_net_pnl": round_price(out["actual_net_pnl"].mean()),
            "sim_avg_net_pnl": round_price(out.loc[resolved, "sim_net_pnl"].mean()),
            "sim_avgvol_avg_net_pnl": round_price(out.loc[resolved, "sim_avgvol_net_pnl"].mean()),
            "tp_hits": int((out["bar_outcome"].isin(["tp_first", "tp_same_bar"])).sum()),
            "sl_hits": int((out["bar_outcome"].isin(["sl_first", "sl_same_bar"])).sum()),
            "not_hit_in_data": int((out["sim_close_type"] == "not_hit_in_data").sum()),
        }

        overall_df = pd.DataFrame([overall])
        overall_df.to_csv(OVERVIEW_FILE, index=False)

        summary_rows = []
        for sym, g in out.groupby("symbol"):
            resolved_g = g["sim_close_type"].isin(["tp", "sl"])
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
        print(f"Raw deals file     : {RAW_DEALS_FILE}")
        print(f"Closed trades file : {CLOSED_TRADES_FILE}")
        print(f"Overall summary    : {OVERVIEW_FILE}")
        print(f"Symbol summary     : {SUMMARY_FILE}")
        print(f"Detail file        : {DETAIL_FILE}")
        print(f"Skipped no rule    : {skipped_no_rule}")

        return out

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run()