# research/run_garch_scan.py
from __future__ import annotations

import json
import logging
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger("airflow.task")


# ============================================================
# DB HELPERS
# ============================================================


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


def get_engine_from_env(application_name: str = "garch_qlike_scan") -> Engine:
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


# ============================================================
# BASIC HELPERS
# ============================================================


def _parse_utc_dt(dt_in: Any) -> Optional[pd.Timestamp]:
    if dt_in is None:
        return None
    ts = pd.Timestamp(dt_in)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _dt_to_ns(dt_in: Any) -> Optional[int]:
    ts = _parse_utc_dt(dt_in)
    if ts is None:
        return None
    return int(ts.value)


def _as_int_list(v: Any, default: Sequence[int]) -> List[int]:
    if v is None:
        return list(default)
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out or list(default)
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                pass
        return out or list(default)
    try:
        return [int(v)]
    except Exception:
        return list(default)


def qlike_loss(realized_var: float, forecast_var: float) -> float:
    rv = float(max(realized_var, 1e-18))
    fv = float(max(forecast_var, 1e-18))
    x = rv / fv
    return float(x - math.log(x) - 1.0)


def compute_log_returns(close_arr: np.ndarray) -> np.ndarray:
    close_arr = np.asarray(close_arr, dtype=np.float64)
    if close_arr.size < 2:
        return np.empty(0, dtype=np.float64)

    close_arr = np.where(np.isfinite(close_arr), close_arr, np.nan)
    close_arr = close_arr[np.isfinite(close_arr)]
    close_arr = close_arr[close_arr > 0.0]

    if close_arr.size < 2:
        return np.empty(0, dtype=np.float64)

    rets = np.diff(np.log(close_arr))

    # Keep internal units stable for GARCH fit on BTC.
    # Everything in this module uses pct returns and pct^2 variance.
    rets = rets * 100.0
    rets = np.clip(rets, -25.0, 25.0)
    rets = rets[np.isfinite(rets)]

    return rets.astype(np.float64, copy=False)


def align_price_to_returns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert a close series into a return-aligned frame.

    Output columns:
      - time_ns: timestamp of the return observation, aligned to the right edge
      - close: close at that timestamp
      - ret: log return in percent from previous close to current close
    """
    if df.is_empty():
        return pl.DataFrame(schema={"time_ns": pl.Int64, "close": pl.Float64, "ret": pl.Float64})

    out = (
        df.select(
            pl.col("time_ns").cast(pl.Int64),
            pl.col("close").cast(pl.Float64),
        )
        .sort("time_ns")
        .with_columns(
            pl.col("close")
            .log()
            .diff()
            .mul(100.0)
            .clip(-25.0, 25.0)
            .alias("ret")
        )
        .drop_nulls(["ret"])
    )

    return out.select("time_ns", "close", "ret")


# ============================================================
# DATA LOAD / CACHE
# ============================================================


def load_close_df_for_grid(
    engine: Engine,
    pair: str,
    market_type: str,
    grid_start_date: str,
    grid_end_date: str,
) -> pd.DataFrame:
    start_ns = _dt_to_ns(grid_start_date)
    end_ns = _dt_to_ns(grid_end_date)

    if start_ns is None or end_ns is None:
        return pd.DataFrame(columns=["time_ns", "close"])

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
        df = pd.read_sql_query(
            query,
            conn,
            params={
                "pair": pair,
                "market_type": market_type,
                "start_ns": start_ns,
                "end_ns": end_ns,
            },
        )

    if df.empty:
        return pd.DataFrame(columns=["time_ns", "close"])

    df["time_ns"] = df["time_ns"].astype("int64")
    df["close"] = df["close"].astype("float64")
    return df.sort_values("time_ns").reset_index(drop=True)


def resample_close_to_timeframe(df_5m: pd.DataFrame, timeframe_min: int) -> pd.DataFrame:
    if df_5m.empty:
        return df_5m.copy()

    if timeframe_min == 5:
        return df_5m.copy()

    out = df_5m.copy()
    out["dt"] = pd.to_datetime(out["time_ns"], unit="ns", utc=True)
    out = out.set_index("dt")[ ["close"] ]

    resampled = out.resample(f"{int(timeframe_min)}min").last().dropna()

    if resampled.empty:
        return pd.DataFrame(columns=["time_ns", "close"])

    resampled = resampled.reset_index()
    resampled["time_ns"] = resampled["dt"].astype("int64")
    resampled = resampled.drop(columns=["dt"])
    resampled["close"] = resampled["close"].astype("float64")
    return resampled[["time_ns", "close"]].sort_values("time_ns").reset_index(drop=True)


def prepare_timeframe_cache(
    session_dir: str,
    run_cfg: Dict[str, Any],
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    session_dir = Path(session_dir)
    cache_dir = session_dir / "garch_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    grid_start_date = str(run_cfg["grid_start_date"])
    grid_end_date = str(run_cfg["grid_end_date"])
    pair = str(run_cfg.get("pair", "XXBTZUSD"))
    market_type = str(run_cfg.get("market_type", "spot"))
    timeframes = _as_int_list(run_cfg.get("garch_timeframes", [5, 15]), default=[5, 15])

    engine = get_engine_from_env()

    logger.info("📥 Loading base price data from DB for garch cache...")
    base_df = load_close_df_for_grid(engine, pair, market_type, grid_start_date, grid_end_date)
    logger.info("📦 Base rows loaded: %d", int(base_df.shape[0]))

    base_pl = pl.from_pandas(base_df) if not base_df.empty else pl.DataFrame(schema={"time_ns": pl.Int64, "close": pl.Float64})

    manifest = {
        "pair": pair,
        "market_type": market_type,
        "grid_start_date": grid_start_date,
        "grid_end_date": grid_end_date,
        "timeframes": [],
    }

    for tf_min in timeframes:
        cache_path = cache_dir / f"close_{tf_min}m.parquet"

        if cache_path.exists() and not force_rebuild:
            tf_rows = int(pl.read_parquet(cache_path).height)
            logger.info("✅ Cache exists: %s (%d rows)", cache_path.name, tf_rows)
            manifest["timeframes"].append(
                {"timeframe_min": int(tf_min), "cache_path": str(cache_path), "rows": tf_rows}
            )
            continue

        if tf_min == 5:
            tf_df = base_pl.clone()
        else:
            tf_df = (
                base_pl
                .with_columns(
                    pl.from_epoch(pl.col("time_ns"), time_unit="ns")
                    .dt.truncate(f"{int(tf_min)}m")
                    .alias("dt")
                )
                .group_by("dt", maintain_order=True)
                .agg(
                    pl.col("close").last().alias("close")
                )
                .sort("dt")
                .with_columns(pl.col("dt").cast(pl.Int64).alias("time_ns"))
                .select("time_ns", "close")
            )

        if tf_df.is_empty():
            logger.warning("⚠️ Resample returned no rows for %dm", tf_min)
            manifest["timeframes"].append(
                {"timeframe_min": int(tf_min), "cache_path": str(cache_path), "rows": 0}
            )
            continue

        tf_df.write_parquet(cache_path)
        logger.info("💾 Wrote cache: %s (%d rows)", cache_path.name, int(tf_df.height))
        manifest["timeframes"].append(
            {"timeframe_min": int(tf_min), "cache_path": str(cache_path), "rows": int(tf_df.height)}
        )

    manifest_path = session_dir / "garch_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")

    return {
        "manifest_path": str(manifest_path),
        "cache_dir": str(cache_dir),
        "timeframes": manifest["timeframes"],
    }


# ============================================================
# GARCH(1,1)
# ============================================================


def _garch11_neg_loglik(params: np.ndarray, resid: np.ndarray) -> float:
    omega, alpha, beta = float(params[0]), float(params[1]), float(params[2])

    if not np.isfinite(omega) or not np.isfinite(alpha) or not np.isfinite(beta):
        return 1e50
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0:
        return 1e50

    pers = alpha + beta
    if pers >= 0.999999:
        return 1e30 + (pers - 0.999999) * 1e33

    n = resid.shape[0]
    if n < 10:
        return 1e50

    var0 = float(np.var(resid, ddof=1))
    if not np.isfinite(var0) or var0 <= 0.0:
        return 1e50

    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = max(var0, 1e-12)

    for t in range(1, n):
        sigma2[t] = omega + alpha * (resid[t - 1] ** 2) + beta * sigma2[t - 1]
        if not np.isfinite(sigma2[t]) or sigma2[t] <= 1e-18:
            return 1e50

    ll = 0.0
    for t in range(n):
        s2 = max(sigma2[t], 1e-18)
        e2 = resid[t] * resid[t]
        ll += 0.5 * (math.log(2.0 * math.pi) + math.log(s2) + (e2 / s2))

    if not np.isfinite(ll):
        return 1e50

    return float(ll)


def fit_garch11(returns: np.ndarray) -> Optional[Dict[str, Any]]:
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]

    if returns.size < 100:
        return None

    mu = float(np.mean(returns))
    resid = returns - mu
    resid = resid[np.isfinite(resid)]

    if resid.size < 100:
        return None

    var = float(np.var(resid, ddof=1))
    if not np.isfinite(var) or var <= 0.0:
        return None

    starts = [
        np.array([max(var * 0.01, 1e-8), 0.05, 0.90], dtype=np.float64),
        np.array([max(var * 0.05, 1e-8), 0.10, 0.80], dtype=np.float64),
        np.array([max(var * 0.001, 1e-8), 0.15, 0.70], dtype=np.float64),
    ]

    bounds = [
        (1e-12, None),
        (1e-12, 0.999),
        (1e-12, 0.999),
    ]

    best_res = None
    best_val = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x0 in starts:
            res = minimize(
                _garch11_neg_loglik,
                x0=x0,
                args=(resid,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-10, "disp": False},
            )
            if res.success and np.isfinite(res.fun) and res.fun < best_val:
                best_res = res
                best_val = float(res.fun)

    if best_res is None:
        return None

    omega, alpha, beta = map(float, best_res.x)
    if not np.isfinite(omega) or not np.isfinite(alpha) or not np.isfinite(beta):
        return None
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0:
        return None
    if alpha + beta >= 0.999999:
        return None

    n = resid.shape[0]
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = max(var, 1e-12)

    for t in range(1, n):
        sigma2[t] = omega + alpha * (resid[t - 1] ** 2) + beta * sigma2[t - 1]
        if not np.isfinite(sigma2[t]) or sigma2[t] <= 1e-18:
            return None

    return {
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "resid": resid,
        "sigma2": sigma2,
    }


def garch_step_variance(
    omega: float,
    alpha: float,
    beta: float,
    last_sigma2: float,
    last_resid2: float,
) -> float:
    """One-step-ahead conditional variance."""
    val = float(omega) + float(alpha) * float(last_resid2) + float(beta) * float(last_sigma2)
    if not np.isfinite(val) or val <= 0.0:
        return np.nan
    return float(val)


def garch_cum_forecast_variance(
    omega: float,
    alpha: float,
    beta: float,
    last_sigma2: float,
    last_resid2: float,
    horizon: int,
) -> float:
    """
    Cumulative expected variance over a horizon of h bars.

    Uses the closed-form expectation for multi-step GARCH(1,1):
        E[sigma^2_{t+h}] = omega * (1 - a^h) / (1 - a) + a^h * sigma^2_{t+1}
    where a = alpha + beta and sigma^2_{t+1} is the one-step forecast.

    Returns sum_{h=1..H} E[sigma^2_{t+h}].
    """
    horizon = int(horizon)
    if horizon <= 0:
        return np.nan

    a = float(alpha) + float(beta)
    w = float(omega)
    s1 = garch_step_variance(omega, alpha, beta, last_sigma2, last_resid2)
    if not np.isfinite(s1) or s1 <= 0.0:
        return np.nan

    if a < 0.0 or a >= 0.999999:
        return np.nan

    if abs(1.0 - a) < 1e-12:
        # Near-integrated case; use recursion to avoid blowups.
        total = 0.0
        f = s1
        for _ in range(horizon):
            if not np.isfinite(f) or f <= 0.0:
                return np.nan
            total += f
            f = w + a * f
        return float(total)

    total = 0.0
    for h in range(1, horizon + 1):
        exp_sigma2 = w * (1.0 - a**h) / (1.0 - a) + (a**h) * s1
        if not np.isfinite(exp_sigma2) or exp_sigma2 <= 0.0:
            return np.nan
        total += exp_sigma2

    return float(total)


def ewma_last_variance(resid: np.ndarray, lam: float = 0.94) -> Tuple[float, float]:
    resid = np.asarray(resid, dtype=np.float64)
    resid = resid[np.isfinite(resid)]
    if resid.size < 2:
        return np.nan, np.nan

    sigma2 = float(np.var(resid[: min(100, resid.size)], ddof=1))
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        sigma2 = 1e-8

    for i in range(1, resid.size):
        sigma2 = lam * sigma2 + (1.0 - lam) * (resid[i - 1] ** 2)

    last_resid2 = float(resid[-1] ** 2)
    return float(sigma2), float(last_resid2)


# ============================================================
# JOB BUILDING
# ============================================================


def build_scan_jobs(
    session_dir: str,
) -> List[Dict[str, Any]]:
    session_dir = Path(session_dir)
    snapshot_path = session_dir / "run_config.json"
    cache_manifest_path = session_dir / "garch_cache_manifest.json"

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing run_config snapshot: {snapshot_path}")
    if not cache_manifest_path.exists():
        raise FileNotFoundError(f"Missing cache manifest: {cache_manifest_path}")

    run_cfg = json.loads(snapshot_path.read_text(encoding="utf8"))
    manifest = json.loads(cache_manifest_path.read_text(encoding="utf8"))

    train_window_days = int(run_cfg.get("garch_train_window_days", 30))
    refit_every_days = int(run_cfg.get("garch_refit_every_days", 2))
    horizons_hours = _as_int_list(run_cfg.get("garch_horizons_hours", [24]), default=[24])
    chunk_origins = int(run_cfg.get("garch_chunk_origins", 8000))
    if chunk_origins <= 0:
        chunk_origins = 8000

    jobs: List[Dict[str, Any]] = []

    for tf_info in manifest.get("timeframes", []):
        timeframe_min = int(tf_info["timeframe_min"])
        cache_path = Path(tf_info["cache_path"])
        if not cache_path.exists():
            logger.warning("⚠️ Missing cache file: %s", cache_path)
            continue

        tf_df = pl.read_parquet(cache_path)
        n_rows = int(tf_df.height)
        if n_rows < 200:
            logger.warning("⚠️ Not enough rows in cache for %dm (%d rows)", timeframe_min, n_rows)
            continue

        bars_per_day = int(round((24 * 60) / float(timeframe_min)))
        train_window_bars = int(train_window_days * bars_per_day)
        refit_every_bars = max(1, int(refit_every_days * bars_per_day))
        horizon_bars_list = [max(1, int(round(h * 60 / float(timeframe_min)))) for h in horizons_hours]
        max_horizon_bars = max(horizon_bars_list)

        # Need enough bars to train and enough bars after each origin to score every horizon.
        start_origin = train_window_bars
        end_origin = n_rows - max_horizon_bars - 1

        if end_origin <= start_origin:
            logger.warning("⚠️ Insufficient window for %dm: start=%d end=%d", timeframe_min, start_origin, end_origin)
            continue

        for chunk_start in range(start_origin, end_origin + 1, chunk_origins):
            chunk_end = min(end_origin, chunk_start + chunk_origins - 1)
            job_id = f"tf{timeframe_min}m_{chunk_start}_{chunk_end}"

            jobs.append(
                {
                    "job_id": job_id,
                    "timeframe_min": timeframe_min,
                    "cache_path": str(cache_path),
                    "origin_start": int(chunk_start),
                    "origin_end": int(chunk_end),
                    "train_window_days": train_window_days,
                    "refit_every_days": refit_every_days,
                    "horizons_hours": horizons_hours,
                    "progress_every": int(run_cfg.get("garch_progress_every", 500)),
                    "score_every_bar": bool(run_cfg.get("garch_score_every_bar", False)),
                }
            )

    jobs_path = session_dir / "garch_jobs.json"
    jobs_path.write_text(json.dumps(jobs, indent=2), encoding="utf8")

    logger.info("📋 Built %d GARCH jobs", len(jobs))
    return jobs


# ============================================================
# JOB RUNNER
# ============================================================


def _make_baseline_row(realized_var: float, forecast_var: float, prefix: str) -> Dict[str, float]:
    q = qlike_loss(realized_var, forecast_var)
    return {
        f"{prefix}_var": float(forecast_var),
        f"qlike_{prefix}": float(q),
    }


def run_scan_job(
    job: Dict[str, Any],
    session_dir: str,
) -> Dict[str, Any]:
    session_dir = Path(session_dir)
    parts_dir = session_dir / "garch_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    job_id = str(job["job_id"])
    timeframe_min = int(job["timeframe_min"])
    cache_path = Path(job["cache_path"])
    origin_start = int(job["origin_start"])
    origin_end = int(job["origin_end"])
    train_window_days = int(job["train_window_days"])
    refit_every_days = int(job["refit_every_days"])
    horizons_hours = [int(x) for x in job["horizons_hours"]]
    progress_every = max(1, int(job.get("progress_every", 500)))
    score_every_bar = bool(job.get("score_every_bar", False))

    logger.info(
        "🧵 START job=%s tf=%dm origin=[%d,%d] cache=%s",
        job_id,
        timeframe_min,
        origin_start,
        origin_end,
        cache_path.name,
    )

    tf_df = pl.read_parquet(cache_path).sort("time_ns")
    if tf_df.is_empty():
        logger.warning("⚠️ job=%s empty cache", job_id)
        return {"job_id": job_id, "rows": 0, "part_path": "", "timeframe_min": timeframe_min}

    aligned = align_price_to_returns(tf_df)
    if aligned.height < 100:
        logger.warning("⚠️ job=%s insufficient rows after return alignment", job_id)
        return {"job_id": job_id, "rows": 0, "part_path": "", "timeframe_min": timeframe_min}

    time_ns = aligned.get_column("time_ns").to_numpy()
    close = aligned.get_column("close").to_numpy()
    returns = aligned.get_column("ret").to_numpy()

    if returns.size < 100:
        logger.warning("⚠️ job=%s insufficient returns after preprocessing", job_id)
        return {"job_id": job_id, "rows": 0, "part_path": "", "timeframe_min": timeframe_min}

    bars_per_day = int(round((24 * 60) / float(timeframe_min)))
    train_window_bars = int(train_window_days * bars_per_day)
    refit_every_bars = max(1, int(refit_every_days * bars_per_day))
    horizon_bars_list = [max(1, int(round(h * 60 / float(timeframe_min)))) for h in horizons_hours]
    max_horizon_bars = max(horizon_bars_list)

    # Returns are aligned so returns[i] ends at time_ns[i + 1]
    max_origin = len(returns) - max_horizon_bars - 1
    origin_start = max(origin_start, train_window_bars)
    origin_end = min(origin_end, max_origin)

    if origin_end <= origin_start:
        logger.warning("⚠️ job=%s empty effective origin range", job_id)
        return {"job_id": job_id, "rows": 0, "part_path": "", "timeframe_min": timeframe_min}

    logger.info(
        "📈 job=%s tf=%dm train_bars=%d refit_bars=%d horizons=%s returns=%d aligned_rows=%d",
        job_id,
        timeframe_min,
        train_window_bars,
        refit_every_bars,
        horizons_hours,
        int(returns.size),
        int(aligned.height),
    )

    rows: List[Dict[str, Any]] = []
    fit_attempts = 0
    fit_success = 0
    fit_fail = 0
    skipped_no_future = 0

    total_origins = ((origin_end - origin_start) // refit_every_bars) + 1
    last_fit_log_ts = 0.0
    cached_fit: Optional[Dict[str, Any]] = None
    cached_fit_origin: Optional[int] = None

    for idx, origin in enumerate(range(origin_start, origin_end + 1, refit_every_bars), start=1):
        if idx == 1 or idx % progress_every == 0:
            logger.info(
                "🔄 job=%s progress %d/%d origin=%d rows=%d",
                job_id,
                idx,
                total_origins,
                origin,
                len(rows),
            )

        train_start = origin - train_window_bars
        train_end = origin
        train_ret = returns[train_start:train_end]
        train_ret = train_ret[np.isfinite(train_ret)]

        if train_ret.size < 100:
            fit_fail += 1
            if idx == 1 or idx % progress_every == 0:
                logger.warning(
                    "⚠️ job=%s skipped origin=%d because train_ret too small (%d)",
                    job_id,
                    origin,
                    int(train_ret.size),
                )
            continue

        fit_attempts += 1

        t0 = time.perf_counter()
        if idx == 1 or idx % progress_every == 0:
            logger.info(
                "⏳ job=%s fitting origin=%d train_n=%d mean=%.6f std=%.6f",
                job_id,
                origin,
                int(train_ret.size),
                float(np.mean(train_ret)),
                float(np.std(train_ret)),
            )

        fit = fit_garch11(train_ret)
        fit_sec = time.perf_counter() - t0

        if fit is None:
            fit_fail += 1
            cached_fit = None
            cached_fit_origin = None
            if idx == 1 or idx % progress_every == 0:
                logger.warning(
                    "⚠️ job=%s fit failed origin=%d took=%.2fs",
                    job_id,
                    origin,
                    fit_sec,
                )
            continue

        fit_success += 1
        cached_fit = fit
        cached_fit_origin = origin

        if idx == 1 or idx % progress_every == 0 or fit_sec >= 10.0:
            logger.info(
                "✅ job=%s fit ok origin=%d took=%.2fs",
                job_id,
                origin,
                fit_sec,
            )

        mu = float(fit["mu"])
        omega = float(fit["omega"])
        alpha = float(fit["alpha"])
        beta = float(fit["beta"])
        resid_train = np.asarray(fit["resid"], dtype=np.float64)
        sigma2_train = np.asarray(fit["sigma2"], dtype=np.float64)

        last_sigma2 = float(sigma2_train[-1])
        last_resid2 = float(resid_train[-1] ** 2)
        sample_var = float(np.var(resid_train, ddof=1))

        # Fair EWMA baseline: same training slice, same origin, same forecast horizon logic.
        ewma_sigma2, _ = ewma_last_variance(resid_train, lam=0.94)

        # This origin corresponds to returns[origin] ending at time_ns[origin + 1].
        # So use time_ns[origin + 1] as the timestamp of the latest observed return.
        origin_time_idx = min(origin + 1, len(time_ns) - 1)
        origin_time_ns = int(time_ns[origin_time_idx])

        # Future realized variance starts at the next return after the origin.
        future_slice = returns[origin + 1 : origin + 1 + max_horizon_bars]
        if future_slice.size < max_horizon_bars:
            skipped_no_future += 1
            continue

        # Optional score-every-bar mode: if true, use current fit for all bars until next refit.
        # Here the job itself advances by refit_every_bars, so the default is sparse scoring.
        # This hook is kept for later extension.
        _ = score_every_bar

        for h_bars, h_hours in zip(horizon_bars_list, horizons_hours):
            future_h = future_slice[:h_bars]
            realized_var = float(np.sum(np.square(future_h)))

            garch_var = garch_cum_forecast_variance(
                omega=omega,
                alpha=alpha,
                beta=beta,
                last_sigma2=last_sigma2,
                last_resid2=last_resid2,
                horizon=h_bars,
            )

            ewma_cum_var = float(h_bars * ewma_sigma2) if np.isfinite(ewma_sigma2) else np.nan
            sample_cum_var = float(h_bars * sample_var)

            if not np.isfinite(garch_var) or garch_var <= 0.0:
                continue

            q_garch = qlike_loss(realized_var, garch_var)
            q_ewma = qlike_loss(realized_var, ewma_cum_var)
            q_sample = qlike_loss(realized_var, sample_cum_var)

            rows.append(
                {
                    "job_id": job_id,
                    "timeframe_min": int(timeframe_min),
                    "time_ns": origin_time_ns,
                    "train_window_days": int(train_window_days),
                    "refit_every_days": int(refit_every_days),
                    "horizon_hours": int(h_hours),
                    "horizon_bars": int(h_bars),
                    "realized_var": realized_var,
                    "garch_var": float(garch_var),
                    "ewma_var": float(ewma_cum_var),
                    "sample_var": float(sample_cum_var),
                    "qlike_garch": q_garch,
                    "qlike_ewma": q_ewma,
                    "qlike_sample": q_sample,
                    "garch_better_than_ewma": int(q_garch < q_ewma),
                    "garch_better_than_sample": int(q_garch < q_sample),
                    "mu": mu,
                    "omega": omega,
                    "alpha": alpha,
                    "beta": beta,
                    "last_sigma2": last_sigma2,
                    "last_resid2": last_resid2,
                    "sample_var_train": sample_var,
                    "cached_fit_origin": int(cached_fit_origin if cached_fit_origin is not None else origin),
                }
            )

        now_ts = time.time()
        if idx == 1 or idx % progress_every == 0 or (now_ts - last_fit_log_ts) > 120.0:
            last_fit_log_ts = now_ts
            logger.info(
                "📌 job=%s status: origins_done=%d/%d rows=%d fits=%d ok=%d fail=%d skipped_future=%d",
                job_id,
                idx,
                total_origins,
                len(rows),
                fit_attempts,
                fit_success,
                fit_fail,
                skipped_no_future,
            )

    if rows:
        part_df = pl.DataFrame(rows)
        part_path = parts_dir / f"{job_id}.parquet"
        part_df.write_parquet(part_path)
        logger.info(
            "✅ DONE job=%s rows=%d fit_attempts=%d success=%d fail=%d skipped_future=%d -> %s",
            job_id,
            len(rows),
            fit_attempts,
            fit_success,
            fit_fail,
            skipped_no_future,
            part_path.name,
        )
        return {
            "job_id": job_id,
            "rows": len(rows),
            "part_path": str(part_path),
            "timeframe_min": timeframe_min,
        }

    logger.info(
        "ℹ️ DONE job=%s rows=0 fit_attempts=%d success=%d fail=%d skipped_future=%d",
        job_id,
        fit_attempts,
        fit_success,
        fit_fail,
        skipped_no_future,
    )
    return {
        "job_id": job_id,
        "rows": 0,
        "part_path": "",
        "timeframe_min": timeframe_min,
    }


# ============================================================
# COMBINE / SUMMARIZE
# ============================================================


def summarize_scan(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    grouped = (
        detail_df
        .groupby(["timeframe_min", "horizon_hours"], as_index=False)
        .agg(
            n=("qlike_garch", "size"),
            mean_qlike_garch=("qlike_garch", "mean"),
            mean_qlike_ewma=("qlike_ewma", "mean"),
            mean_qlike_sample=("qlike_sample", "mean"),
            median_qlike_garch=("qlike_garch", "median"),
            median_qlike_ewma=("qlike_ewma", "median"),
            median_qlike_sample=("qlike_sample", "median"),
            garch_beats_ewma=("garch_better_than_ewma", "mean"),
            garch_beats_sample=("garch_better_than_sample", "mean"),
        )
        .sort_values(["timeframe_min", "horizon_hours"])
        .reset_index(drop=True)
    )

    return grouped


def combine_scan_outputs(session_dir: str) -> Dict[str, Any]:
    session_dir = Path(session_dir)
    parts_dir = session_dir / "garch_parts"

    part_files = sorted(parts_dir.glob("*.parquet"))
    if not part_files:
        logger.warning("No part files found in %s", parts_dir)
        return {
            "rows": 0,
            "detail_path": "",
            "summary_path": "",
        }

    dfs = [pl.read_parquet(p) for p in part_files]
    detail = pl.concat(dfs, how="vertical_relaxed")
    detail_path = session_dir / "garch_qlike_detail_all.parquet"
    detail.write_parquet(detail_path)

    detail_pd = detail.to_pandas()
    summary_pd = summarize_scan(detail_pd)

    summary_path = session_dir / "garch_qlike_summary_all.csv"
    summary_pd.to_csv(summary_path, index=False)

    logger.info("✅ Combined %d part files into %d rows", len(part_files), int(detail.height))
    logger.info("✅ Detail: %s", detail_path)
    logger.info("✅ Summary: %s", summary_path)

    return {
        "rows": int(detail.height),
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
    }
