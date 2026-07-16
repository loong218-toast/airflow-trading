# fetch_mt5_cache.py

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# =============================================================================
# Paths
# =============================================================================

DAGS_ROOT = Path(__file__).resolve().parents[2]  # .../dags
REPO_ROOT = DAGS_ROOT.parent  # .../airflow-trading

os.environ.setdefault("AIRFLOW_TRADING_ROOT", str(REPO_ROOT))

if str(DAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(DAGS_ROOT))

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(DAGS_ROOT / ".env")

from research.expectancy_config import (  # noqa: E402
    CACHE_DIR,
    INSTRUMENT_CONFIG,
    MT5_CHUNK_DAYS,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_PATH,
    MT5_PORTABLE,
    MT5_SERVER,
    MT5_TIMEOUT,
)

# =============================================================================
# Settings
# =============================================================================

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "365"))
OVERLAP_DAYS = int(os.getenv("MT5_OVERLAP_DAYS", "3"))
TIMEFRAME = "M5"
FULL_REFRESH_DEFAULT = os.getenv("MT5_FULL_REFRESH", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
STARTUP_DELAY_SECONDS = int(os.getenv("MT5_STARTUP_DELAY_SECONDS", "20"))

# =============================================================================
# Time helpers
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def utc_days_ago(days: int) -> datetime:
    return utc_now() - timedelta(days=days)

def safe_dt_from_unix(ts: int | float | None) -> str:
    if ts is None:
        return ""
    try:
        ts_int = int(ts)
    except Exception:
        return ""
    if ts_int <= 0:
        return ""
    return datetime.fromtimestamp(ts_int, tz=timezone.utc).isoformat()

# =============================================================================
# MT5 helpers
# =============================================================================

def import_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MetaTrader5 is not installed in this Python environment. "
            "Run this updater on the Windows machine where MT5 and the package exist."
        ) from exc
    return mt5

def init_mt5():
    mt5 = import_mt5()

    init_kwargs: dict[str, Any] = {}
    if MT5_PATH:
        init_kwargs["path"] = MT5_PATH
    if MT5_LOGIN:
        try:
            init_kwargs["login"] = int(MT5_LOGIN)
        except Exception as exc:
            raise RuntimeError(f"MT5_LOGIN is not a valid integer: {MT5_LOGIN!r}") from exc
    if MT5_PASSWORD:
        init_kwargs["password"] = MT5_PASSWORD
    if MT5_SERVER:
        init_kwargs["server"] = MT5_SERVER
    if MT5_TIMEOUT:
        init_kwargs["timeout"] = int(MT5_TIMEOUT)
    if MT5_PORTABLE:
        init_kwargs["portable"] = True

    ok = mt5.initialize(**init_kwargs) if init_kwargs else mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    return mt5

def shutdown_mt5(mt5) -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass

def resolve_mt5_symbol(mt5, requested: str) -> str:
    requested = str(requested).strip()
    if not requested:
        raise RuntimeError("Empty MT5 symbol requested")

    if mt5.symbol_info(requested) is not None:
        return requested

    all_symbols = mt5.symbols_get() or []

    exact_ci = [s.name for s in all_symbols if s.name.lower() == requested.lower()]
    if len(exact_ci) == 1:
        return exact_ci[0]

    contains = [s.name for s in all_symbols if requested.lower() in s.name.lower()]
    if len(contains) == 1:
        return contains[0]

    if len(contains) > 1:
        raise RuntimeError(f"Multiple symbols match '{requested}': {contains}")

    raise RuntimeError(f"Symbol '{requested}' not found in MT5 terminal")

# =============================================================================
# Data fetching
# =============================================================================

def fetch_mt5_rates_chunk(
    mt5,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    rates = mt5.copy_rates_range(
        symbol,
        mt5.TIMEFRAME_M5,
        start_dt,
        end_dt,
    )

    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rates)

def normalize_mt5_rates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "time" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df = df.drop_duplicates(subset=["time"]).sort_values("time")

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["time"])

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    df["time_ns"] = df["time"].astype("int64")

    cols = ["time_ns", "open", "high", "low", "close"]
    df = df[cols].dropna().reset_index(drop=True)
    return df

def fetch_mt5_rates_range_chunked(
    mt5,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    chunk_days: int,
) -> pd.DataFrame:
    symbol = resolve_mt5_symbol(mt5, symbol)

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select failed for {symbol}: {mt5.last_error()}")

    chunks: list[pd.DataFrame] = []
    cur = start_dt

    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=chunk_days), end_dt)
        df = fetch_mt5_rates_chunk(mt5, symbol, cur, chunk_end)
        if not df.empty:
            chunks.append(df)
        cur = chunk_end

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    return normalize_mt5_rates(df)

# =============================================================================
# Persistence
# =============================================================================

def cache_paths_for_symbol(mt5_symbol: str) -> tuple[Path, Path]:
    cache_file = CACHE_DIR / f"{mt5_symbol}_m5_cache.csv"
    meta_file = cache_file.with_suffix(".meta.json")
    return cache_file, meta_file

def read_existing_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        old = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    required = {"time_ns", "open", "high", "low", "close"}
    if not required.issubset(old.columns):
        return pd.DataFrame()

    old = old.copy()
    old["time_ns"] = pd.to_numeric(old["time_ns"], errors="coerce")
    for col in ("open", "high", "low", "close"):
        old[col] = pd.to_numeric(old[col], errors="coerce")

    old = old.dropna(subset=["time_ns", "open", "high", "low", "close"])
    old["time_ns"] = old["time_ns"].astype("int64")
    old = old.sort_values("time_ns").drop_duplicates(subset=["time_ns"], keep="last")
    return old.reset_index(drop=True)

def merge_and_deduplicate(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = new.copy()
    elif new.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new], ignore_index=True)

    if merged.empty:
        return merged

    merged = merged.copy()
    merged["time_ns"] = pd.to_numeric(merged["time_ns"], errors="coerce")
    for col in ("open", "high", "low", "close"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = (
        merged.dropna(subset=["time_ns", "open", "high", "low", "close"])
        .sort_values("time_ns")
        .drop_duplicates(subset=["time_ns"], keep="last")
        .reset_index(drop=True)
    )
    merged["time_ns"] = merged["time_ns"].astype("int64")
    return merged

def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)

def save_metadata(
    path: Path,
    *,
    symbol: str,
    timeframe: str,
    bars_available: int,
    oldest_bar_utc: str,
    newest_bar_utc: str,
    lookback_days: float,
    lookback_hours: float,
    status: str,
    full_refresh: bool,
    overlap_days: int,
    account_info: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "bars_available": bars_available,
        "oldest_bar_utc": oldest_bar_utc,
        "newest_bar_utc": newest_bar_utc,
        "lookback_days": round(float(lookback_days), 6),
        "lookback_hours": round(float(lookback_hours), 6),
        "status": status,
        "full_refresh": bool(full_refresh),
        "overlap_days": int(overlap_days),
        "updated_utc": utc_now().isoformat(),
        "cache_file": str(path.with_suffix(".csv")),
        "python_executable": sys.executable,
    }
    if account_info:
        payload["account_info"] = account_info

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)

# =============================================================================
# Orchestration
# =============================================================================

def load_last_cached_dt(path: Path) -> Optional[datetime]:
    existing = read_existing_cache(path)
    if existing.empty:
        return None

    last_ns = int(existing["time_ns"].iloc[-1])
    ts = pd.to_datetime(last_ns, unit="ns", utc=True)
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()

def build_fetch_window(full_refresh: bool, path: Path) -> tuple[datetime, datetime]:
    end_dt = utc_now()
    default_start_dt = utc_days_ago(LOOKBACK_DAYS)

    if full_refresh or not path.exists():
        return default_start_dt, end_dt

    last_cached_dt = load_last_cached_dt(path)
    if last_cached_dt is None:
        return default_start_dt, end_dt

    start_dt = max(default_start_dt, last_cached_dt - timedelta(days=OVERLAP_DAYS))
    return start_dt, end_dt

def build_mt5_cache_for_symbol(
    mt5,
    instrument_key: str,
    cfg: dict[str, Any],
    full_refresh: bool = FULL_REFRESH_DEFAULT,
) -> tuple[str, int]:
    mt5_symbol = str(cfg["mt5_symbol"]).strip()
    cache_file, meta_file = cache_paths_for_symbol(mt5_symbol)

    start_dt, end_dt = build_fetch_window(full_refresh, cache_file)

    account_info: dict[str, Any] | None = None
    account = mt5.account_info()
    if account is not None:
        account_info = {
            "login": getattr(account, "login", None),
            "server": getattr(account, "server", None),
            "company": getattr(account, "company", None),
            "name": getattr(account, "name", None),
            "leverage": getattr(account, "leverage", None),
            "currency": getattr(account, "currency", None),
            "balance": getattr(account, "balance", None),
            "equity": getattr(account, "equity", None),
        }

    new_df = fetch_mt5_rates_range_chunked(
        mt5=mt5,
        symbol=mt5_symbol,
        start_dt=start_dt,
        end_dt=end_dt,
        chunk_days=MT5_CHUNK_DAYS,
    )

    if new_df.empty:
        raise RuntimeError(f"No MT5 data returned for {instrument_key} ({mt5_symbol})")

    existing_df = (
        read_existing_cache(cache_file)
        if cache_file.exists() and not full_refresh
        else pd.DataFrame()
    )
    merged_df = merge_and_deduplicate(existing_df, new_df)

    if merged_df.empty:
        raise RuntimeError(f"Merged cache is empty for {instrument_key} ({mt5_symbol})")

    save_cache(merged_df, cache_file)

    oldest_ns = int(merged_df["time_ns"].iloc[0])
    newest_ns = int(merged_df["time_ns"].iloc[-1])
    lookback_seconds = (newest_ns - oldest_ns) / 1_000_000_000.0

    save_metadata(
        meta_file,
        symbol=mt5_symbol,
        timeframe=TIMEFRAME,
        bars_available=len(merged_df),
        oldest_bar_utc=safe_dt_from_unix(oldest_ns / 1_000_000_000),
        newest_bar_utc=safe_dt_from_unix(newest_ns / 1_000_000_000),
        lookback_days=lookback_seconds / 86_400.0,
        lookback_hours=lookback_seconds / 3_600.0,
        status="ok",
        full_refresh=full_refresh,
        overlap_days=OVERLAP_DAYS,
        account_info=account_info,
    )

    print(f"[{instrument_key}] saved {len(merged_df):,} rows -> {cache_file}")
    print(f"[{instrument_key}] metadata -> {meta_file}")
    return instrument_key, len(merged_df)

def build_all_mt5_caches(full_refresh: bool = FULL_REFRESH_DEFAULT) -> None:
    if STARTUP_DELAY_SECONDS > 0:
        time.sleep(STARTUP_DELAY_SECONDS)

    mt5 = init_mt5()
    failures: list[tuple[str, str]] = []
    successes: list[str] = []

    try:
        for instrument_key, cfg in INSTRUMENT_CONFIG.items():
            try:
                build_mt5_cache_for_symbol(
                    mt5=mt5,
                    instrument_key=instrument_key,
                    cfg=cfg,
                    full_refresh=full_refresh,
                )
                successes.append(instrument_key)
            except Exception as exc:
                failures.append((instrument_key, str(exc)))
                print(f"[{instrument_key}] FAILED: {exc}")

    finally:
        shutdown_mt5(mt5)

    print(
        f"DONE: {len(successes)} succeeded, {len(failures)} failed out of {len(INSTRUMENT_CONFIG)} instruments"
    )

    if failures:
        failed_keys = ", ".join(k for k, _ in failures)
        raise RuntimeError(f"One or more instruments failed: {failed_keys}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MT5 OHLC cache and keep it updated.")
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore existing cache and rebuild from the full lookback window.",
    )
    args = parser.parse_args()

    build_all_mt5_caches(full_refresh=bool(args.full_refresh))

if __name__ == "__main__":
    main()
