import sys
import os
# 1. Path to the root (airflow-trading)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

# 2. Path to dags folder so "from etl.db import..." works inside your modules
dags_path = os.path.join(root_path, 'dags')
sys.path.append(dags_path)

import pandas as pd
from dags.etl.kraken_api import fetch_ohlc
# etl/grid.py

import json
import time
import tempfile
import math
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pandas as pd

# local imports from your etl package (assumes these exist from earlier steps)
from dags.etl.transform import build_df_main_from_5m, load_candles_from_db
from dags.etl.feature_helpers import generate_filtered_signals
from dags.etl.db import get_engine, save_df_to_sql  # optional DB save if you want
from dags.etl.backtest import backtest_signals_sl_tp
# you may also import combine_and_save_results if you kept that earlier



DEFAULT_DATA_LAKE_ROOT = os.getenv("DATA_LAKE_ROOT", "/opt/airflow/airflow-trading/data_lake")
SESSION_PREFIX = "Opt_Session"

def _atomic_write_parquet(df: pd.DataFrame, path: "Path"):
    path.parent.mkdir(parents=True, exist_ok=True)
    # write to temp file then atomically replace
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    os.close(fd)
    try:
        df.to_parquet(tmp, compression='snappy')
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass

def _list_sessions(base: "Path") -> List["Path"]:
    return sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(SESSION_PREFIX)],
                  key=lambda p: p.stat().st_mtime)

def prepare_base_data(session_dir: str, db_uri: str, force: bool = False) -> Dict:
    """
    Load 5m candles from DB once, compute df_main (MA/stoch) and save to
    session_dir/base_data.parquet. Skips if file exists unless force=True.
    Returns summary.
    """
    session_dir = Path(session_dir)
    base_path = session_dir / "base_data.parquet"
    if base_path.exists() and not force:
        return {"status": "skipped_exists", "path": str(base_path)}

    engine = get_engine(db_uri)
    # NOTE: we assume single pair overall (or you can extend to write per-pair base files)
    # Here we look at all pairs present in candle_raw — but simplest is to choose the pair from run_config.
    # For generality, the transform step can write a base_data per pair, but we'll assume single pair for now.
    # Caller must ensure run_config/pair is known.
    # We'll load ALL 5min candles for the pair(s) present in the session config:
    # For simplicity we'll load the most common pair used by configs (read first cfg to get pair)
    cfg_dir = session_dir / "configs"
    first_cfg = next(cfg_dir.glob("cfg_*.json"), None)
    if first_cfg is None:
        raise RuntimeError("No config files found in session configs directory")

    with open(first_cfg, "r", encoding="utf8") as f:
        cfg0 = json.load(f)
    pair = cfg0.get("pair", cfg0.get("PAIR", "XXBTZUSD"))

    # load from DB (single heavy read)
    df_candles = load_candles_from_db(engine, pair=pair, interval_minutes=5)

    # build df_main (this computes MAs & stoch as in earlier code)
    df_main, nan_report = build_df_main_from_5m(df_candles, run_config=cfg0)

    # atomic write to base_data.parquet
    _atomic_write_parquet(df_main, base_path)

    return {"status": "written", "path": str(base_path), "rows": int(len(df_main))}

def resolve_or_create_session(base_root: str, resume_if_possible: bool = True) -> Path:
    """
    Choose an existing incomplete session to resume, otherwise create a new session folder.
    Criteria to resume: last session exists and has fewer result files than configs.
    """
    base = Path(base_root)
    base.mkdir(parents=True, exist_ok=True)

    sessions = _list_sessions(base)
    if resume_if_possible and sessions:
        last = sessions[-1]
        configs_dir = last / "configs"
        results_dir = last / "results"
        if configs_dir.exists():
            cfg_files = list(configs_dir.glob("cfg_*.json"))
            if cfg_files:
                done = len(list(results_dir.glob("cfg_*.parquet"))) if results_dir.exists() else 0
                if done < len(cfg_files):
                    # resume this one
                    return last

    # otherwise create a new session
    ts = time.strftime("%Y%m%d_%H%M%S")
    # pick a small seq to avoid dupes
    seq = 1
    while True:
        candidate = base / f"{SESSION_PREFIX}_{ts}_{seq:02d}"
        if not candidate.exists():
            break
        seq += 1
    candidate.mkdir(parents=True)
    (candidate / "configs").mkdir()
    (candidate / "results").mkdir()
    return candidate

def _expand_sl_tp(run_cfg: Dict) -> (List[float], List[float]):
    """
    Determine SL and TP candidate lists.
    Priority:
      1) run_cfg.get('sl_values') / 'tp_values' -> use as floats
      2) run_cfg.get('sl_range') / 'tp_range' -> expand using min,max,step
      3) fallback to BTC_SETTINGS values (sl_min, sl_max, point_interval)
    Returns (sl_values, tp_values)
    """
    def _from_values_or_range(cfg, key, fallback_min=None, fallback_max=None, fallback_step=None):
        if key + "_values" in cfg and isinstance(cfg[key + "_values"], list):
            return [float(x) for x in cfg[key + "_values"]]
        if key + "_range" in cfg and isinstance(cfg[key + "_range"], dict):
            r = cfg[key + "_range"]
            mn = float(r.get("min", fallback_min or 0))
            mx = float(r.get("max", fallback_max or mn))
            step = float(r.get("step", fallback_step or 1))
            vals = []
            v = mn
            while v <= mx + 1e-9:
                vals.append(round(float(v), 8))
                v += step
            return vals
        # fallback to BTC_SETTINGS
        b = cfg.get("BTC_SETTINGS", {})
        if key == "sl":
            mn = float(b.get("sl_min", fallback_min or 20))
            mx = float(b.get("sl_max", fallback_max or mn))
        else:
            mn = float(b.get("tp_min", fallback_min or 800))
            mx = float(b.get("tp_max", fallback_max or mn))
        step = float(b.get("point_interval", fallback_step or 40))
        vals = []
        v = mn
        while v <= mx + 1e-9:
            vals.append(round(float(v), 8))
            v += step
        return vals

    sl_vals = _from_values_or_range(run_cfg, "sl", fallback_min=20, fallback_max=800, fallback_step=run_cfg.get("BTC_SETTINGS", {}).get("point_interval", 40))
    tp_vals = _from_values_or_range(run_cfg, "tp", fallback_min=800, fallback_max=15000, fallback_step=run_cfg.get("BTC_SETTINGS", {}).get("point_interval", 40))
    return sl_vals, tp_vals

def _prune_by_min_rr(sl_vals: List[float], tp_vals: List[float], min_rr: float) -> List[tuple]:
    """
    Returns list of (sl,tp) tuples where tp >= min_rr * sl
    """
    combos = []
    for s in sl_vals:
        for t in tp_vals:
            if t >= (min_rr * s):
                combos.append((s, t))
    return combos

def _expand_value_list(cfg: dict, key: str):
    # Returns a list of numeric values for 'key' using either explicit list or range shorthand
    if key + "_values" in cfg and isinstance(cfg[key + "_values"], list):
        return [float(x) for x in cfg[key + "_values"]]
    if key + "_range" in cfg and isinstance(cfg[key + "_range"], dict):
        r = cfg[key + "_range"]
        mn = float(r.get("min", 0))
        mx = float(r.get("max", mn))
        step = float(r.get("step", 1))
        if step <= 0:
            raise ValueError(f"{key}_range step must be > 0")
        vals = []
        v = mn
        # inclusive of mx (use round to avoid fp drift)
        while v <= mx + 1e-9:
            vals.append(float(round(v, 6)))
            v += step
        return vals
    # default fallback
    return [float(cfg.get(key, 0.0))] if cfg.get(key) is not None else [0.0]

def generate_configs(run_config: Dict, session_dir: Path) -> List[Path]:
    """
    Generate configs similar to notebook but expand SL/TP combos from ranges or explicit lists
    and apply min_rr pruning. Each config file includes "sl" and "tp".
    """
    cfg_dir = session_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    exit_windows = run_config.get("exit_windows", run_config.get("BTC_SETTINGS", {}).get("exit_windows", [1,4,12,24,48,72,168,336,672]))
    lookbacks = run_config.get("entry_lookback_list_hours", [24])
    sl_vals, tp_vals = _expand_sl_tp(run_config)
    min_rr = float(run_config.get("min_rr", 3))

    combos = _prune_by_min_rr(sl_vals, tp_vals, min_rr)

    idx = 0
    saved = []
    for ma_rev in [False, True]:
        for ma_int in range(16):
            for use_stoch in [False, True]:
                for use_lb in [False, True]:
                    for lb_hours in (lookbacks if use_lb else [0]):
                        for (sl, tp) in combos:
                            for exit_h in exit_windows:
                                cfg = dict(run_config)
                                cfg.update({
                                    "config_id": f"{idx:04d}",
                                    "ma_int": ma_int,
                                    "ma_reversion": ma_rev,
                                    "use_stochastic": use_stoch,
                                    "use_entry_lookback": use_lb,
                                    "entry_lookback_h": int(lb_hours),
                                    "sl": float(sl),
                                    "tp": float(tp),
                                    "exit_window_h": int(exit_h),
                                    "sl_tp_in_pct": bool(run_config.get("sl_tp_in_pct", True))
                                })
                                path = cfg_dir / f"cfg_{idx:04d}.json"
                                with open(path, "w", encoding="utf8") as f:
                                    json.dump(cfg, f, indent=2)
                                saved.append(path)
                                idx += 1
    return saved

def list_pending_config_paths(session_dir: Path) -> List[str]:
    """
    Return the list of config file paths that do NOT yet have a result file.
    We return strings (so they map cleanly in Airflow expand).
    """
    cfg_dir = session_dir / "configs"
    results_dir = session_dir / "results"
    cfg_paths = sorted(cfg_dir.glob("cfg_*.json"))
    pending = []
    for p in cfg_paths:
        cfg_id = p.stem.split("_")[1]  # cfg_XXXX
        result_path = results_dir / f"{p.stem}.parquet"
        if not result_path.exists():
            pending.append(str(p))
    return pending

def _split_period_windows(df: pd.DataFrame, months: int) -> List[tuple]:
    """
    Return list of (start_time, end_time) windows covering the df time range in non-overlap months.
    months is window length in months (int).
    """
    if df.empty:
        return []
    first = df['time'].min().to_pydatetime()
    last = df['time'].max().to_pydatetime()
    import datetime
    windows = []
    # normalize first to month start
    cur_year, cur_month = first.year, first.month
    start = pd.Timestamp(year=cur_year, month=cur_month, day=1, tz='UTC')
    while start.to_pydatetime() <= last:
        # compute end by adding months
        m = start.month - 1 + months
        y = start.year + m // 12
        mm = (m % 12) + 1
        # end is last instant of previous month of next window -> we will use exclusive end
        end = pd.Timestamp(year=y, month=mm, day=1, tz='UTC')
        windows.append((start, end))
        start = end
    return windows

def compute_config_and_save(cfg_path: str, session_dir: str, db_uri: Optional[str] = None, compute_backtest: bool = True) -> Dict:
    """
    Worker: read base_data.parquet and apply a single config, then optionally backtest SL/TP across  sl_tp_interval_months.
    Writes:
      - results/cfg_XXXX.parquet (signals)
      - results/cfg_XXXX_summary.json (metrics per sl-tp window)
    """

    cfg_path = Path(cfg_path)
    session_dir = Path(session_dir)
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{cfg_path.stem}.parquet"
    summary_path = results_dir / f"{cfg_path.stem}_summary.json"
    if out_path.exists() and summary_path.exists():
        return {"config": cfg_path.name, "status": "skipped_exists"}

    # load config
    with open(cfg_path, "r", encoding="utf8") as f:
        cfg = json.load(f)

    # read base data (precomputed df_main)
    base_file = session_dir / "base_data.parquet"
    if not base_file.exists():
        raise RuntimeError("Missing base_data.parquet; run prepare_base_data first")

    df_main = pd.read_parquet(base_file)  # consider columns selection to save memory

    # local worker caches
    cache_stoch = {}
    cache_lb = {}

    # generate signals (fast)
    df_signals = generate_filtered_signals(df_main, cfg,
                                           base_minutes=cfg.get("BASE_MINUTES", 5),
                                           cache_stoch=cache_stoch,
                                           cache_lb=cache_lb)

    # write signals (always write, even if empty)
    df_signals.to_parquet(out_path, compression='snappy')

    summary = {"config": cfg_path.name, "rows": int(len(df_signals)), "metrics": []}

    if compute_backtest and not df_signals.empty:
        # split into windows and run backtest per window
        months = int(cfg.get("sl_tp_interval_months", cfg.get("sl_tp_interval_months", 6)))
        windows = _split_period_windows(df_main, months)

        for (start, end) in windows:
            # slice df_main & df_signals for this window (time >= start and time < end)
            mask_main = (df_main['time'] >= start) & (df_main['time'] < end)
            if not mask_main.any():
                continue
            df_m_slice = df_main.loc[mask_main].reset_index(drop=True)

            mask_sig = (df_signals['time'] >= start) & (df_signals['time'] < end)
            df_s_slice = df_signals.loc[mask_sig].reset_index(drop=True)
            
            if df_s_slice.empty:
                summary["metrics"].append({
                    "start": str(start), "end": str(end),
                    "total_trades": 0
                })
                continue

            res = backtest_signals_sl_tp(
                df_main=df_m_slice,
                df_signals=df_s_slice,
                sl=float(cfg.get("sl", 0.0)),
                tp=float(cfg.get("tp", 0.0)),
                sl_tp_in_pct=bool(cfg.get("sl_tp_in_pct", True)),
                exit_window_h=int(cfg.get("exit_window_h", 24)),
                base_minutes=int(cfg.get("BASE_MINUTES", 5)),
                spread=float(cfg.get("BTC_SETTINGS", {}).get("spread", 0.0)),
                conservative_sl_first=True,
                treat_no_hit_as_loss=True
            )

            res.update({"start": str(start), "end": str(end)})
            summary["metrics"].append(res)

    # atomic write summary json
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    return {"config": cfg_path.name, "status": "done", "rows": int(len(df_signals)), "summary_path": str(summary_path)}

def combine_results_to_master(session_dir: str, output_name: str = "master_results.parquet") -> Dict:
    """
    Combine all per-config result parquet files into a single master parquet inside session_dir.
    Returns summary (num_files, total_rows).
    """
    session_dir = Path(session_dir)
    results_dir = session_dir / "results"
    files = sorted(results_dir.glob("cfg_*.parquet"))
    parts = []
    total_rows = 0
    for p in files:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        parts.append(df)
        total_rows += len(df)
    if not parts:
        # write an empty master to indicate job completed
        pd.DataFrame().to_parquet(session_dir / output_name, compression="snappy")
        return {"files": 0, "rows": 0}
    master = pd.concat(parts, ignore_index=True, sort=False)
    master.to_parquet(session_dir / output_name, compression="snappy")
    return {"files": len(parts), "rows": total_rows}