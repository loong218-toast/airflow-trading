# %%
# === Section 0. Imports ===

# --- System & Path ---
import os
import sys
import gc
import time
import uuid
import glob
import shutil
import hashlib
import tempfile
from pathlib import Path

# --- Data Structures & Typing ---
import json
import jsonschema
import copy
from collections import OrderedDict
from typing import Tuple, Optional, List, Dict
from itertools import product

# --- Data Science & Technical Analysis ---
import numpy as np
import pandas as pd
import talib as ta

# --- Performance & Parallelism ---
from numba import njit, prange

# --- Storage (Parquet/Arrow) ---
import pyarrow as pa
import pyarrow.parquet as pq

# --- Visualization ---
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display

# --- Utilities ---
from datetime import datetime, timedelta
from tqdm import tqdm
import psutil  # Highly recommended for monitoring your 5.85GB RAM limit

# --- Global Settings ---
pd.options.mode.chained_assignment = None

# %%
# Section 1. Config

DEFAULT_RUN_CONFIG = {
    "sl_tp_interval_months": 6,
  "version_prefix": "v",
  "BASE_MINUTES": 5,
  "feature_columns": ["ma_gap_a","ma_gap_b","ma_gap_c","price_ma_gap"],
  "feature_definitions": {
    "price_ma_gap": {"calc": "pct_div", "args": ["close", "ma_hour"]},
    "ma_gap_a": {"calc": "pct_div", "args": ["ma_5min", "ma_hour"]},
    "ma_gap_b": {"calc": "pct_div", "args": ["ma_hour", "ma_day"]},
    "ma_gap_c": {"calc": "pct_div", "args": ["ma_day", "ma_week"]}
  },
  "feature_aggregations": {"default":"mean"},
  "ma_periods": {"ma_5min":50, "ma_hour":50, "ma_day":1200, "ma_week":8400},
  "stochastic": {"use_stochastic": "false", "stoch_k":12, "stoch_d":12, "stoch_slow":8},
  "BTC_SETTINGS": {
    "sl_min": 20, "sl_max": 800,
    "tp_min": 800, "tp_max": 15000,
    "point_interval": 40, "round_points": 2,
    "spread": 1.0, "threshold": -100000
  },
  "exit_windows": [1,4,12,24,48,72,168,336,672],
  "entry_lookback_list_hours": [1,4,8,12,16,20,24,28,32,36,40,44,48,72,168],
  "lookahead": {
    "use_cache": "true",
    "max_cached_windows": 2,
    "sentinel_time_ns": -9223372036854775808
  },
  "cache": {
    "lookahead_cache_max": 4,
    "lookback_cache_max": 4,
    "stoch_cache_max": 2
  },
  "min_rr": 3,
  "session": {"interactive": "true", "cleanup_session_threshold": 100, "save_compression": "snappy"}
}

# minimal schema (optional)
RUN_CONFIG_SCHEMA = {
    "type":"object",
    "properties":{
        "BASE_MINUTES":{"type":"integer", "minimum":1},
        "BTC_SETTINGS":{"type":"object"},
        "feature_columns":{"type":"array", "items":{"type":"string"}}
    },
    "required":["BTC_SETTINGS","feature_columns"]
}

def atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=str(path.parent), encoding='utf8') as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True)
        tmp.flush()
    os.replace(tmp.name, str(path))

def load_master_config_safe(config_path: Path, defaults: dict = DEFAULT_RUN_CONFIG):
    """
    Robust loader:
      - if file missing -> write defaults and return defaults
      - if file empty or invalid JSON -> back it up to .invalid.TIMESTAMP.json, write defaults, return defaults
      - merges missing keys from defaults non-destructively
    """
    config_path = Path(config_path)
    if not config_path.exists():
        atomic_write_json(config_path, defaults)
        cfg = defaults.copy()
    else:
        try:
            text = config_path.read_text(encoding='utf8').strip()
            if not text:
                # empty file -> backup & write defaults
                backup = config_path.with_suffix(f".empty_backup.{int(time.time())}.json")
                config_path.rename(backup)
                atomic_write_json(config_path, defaults)
                cfg = defaults.copy()
            else:
                cfg = json.loads(text)
        except json.JSONDecodeError:
            # make a backup of the invalid file and replace with defaults
            backup = config_path.with_suffix(f".invalid_{int(time.time())}.json")
            shutil.move(str(config_path), str(backup))
            atomic_write_json(config_path, defaults)
            cfg = defaults.copy()

    # merge missing defaults (non-destructive)
    merged = defaults.copy()
    merged.update(cfg)
    cfg = merged

    # optional validation (fail fast if schema mismatch)
    try:
        jsonschema.validate(cfg, RUN_CONFIG_SCHEMA)
    except Exception:
        # if jsonschema isn't critical for you, just pass
        pass

    return cfg

def _evict_generic_cache(cache: OrderedDict, max_items: int):
    while len(cache) > max_items:
        cache.popitem(last=False)
    gc.collect()

def _evict_window_lookaheads_if_needed():
    while len(_WINDOW_LOOKAHEADS) > _WINDOW_LOOKAHEADS_MAX:
        old_hours, old_entry = _WINDOW_LOOKAHEADS.popitem(last=False)
        # explicitly drop nested arrays for safety
        for k in ('next_min','next_max','next_min_time_ns','next_max_time_ns','index_map','time'):
            old_entry.pop(k, None)
        gc.collect()

def _maybe_trim_all_caches():
    _evict_generic_cache(_LOOKAHEAD_CACHE, _LOOKAHEAD_CACHE_MAX)
    _evict_generic_cache(_LOOKBACK_CACHE, _LOOKBACK_CACHE_MAX)
    _evict_generic_cache(_STOCH_CACHE, _STOCH_CACHE_MAX)
    _evict_window_lookaheads_if_needed()

# --- Dictionary Helper ---
def save_config_dictionary(session_dir: Path):
    """
    Scans the 'configs' folder and creates a human-readable lookup JSON.
    Maps config_id -> Human Friendly Logic.
    """
    configs_dir = session_dir / "configs"
    dict_path = session_dir / "config_dictionary.json"
    
    # Only run if there are actual configs to map
    config_files = list(configs_dir.glob("*.json"))
    if not config_files:
        return

    mapping = {}
    for cfg_file in sorted(config_files):
        with open(cfg_file, 'r', encoding='utf8') as f:
            c = json.load(f)
            
            ma_val = c.get("ma_int", 0)

            cid = str(c.get('config_id', cfg_file.stem.replace('cfg_', '')))
            mapping[cid] = {
                "logic": {
                    "ma_5m": bool(ma_val & (1 << 0)),
                    "ma_1h": bool(ma_val & (1 << 1)),
                    "ma_1d": bool(ma_val & (1 << 2)),
                    "ma_1w": bool(ma_val & (1 << 3)),
                    "ma_reversion": bool(c.get("ma_reversion")),
                    "stoch": bool(c.get("use_stochastic")),
                    "lookback_h": int(c.get("entry_lookback_h", 0))
                },
                "ma_int": ma_val,
                "ma_reversion": 1 if c.get("ma_reversion") else 0,
                "version": c.get("version_prefix", "v")
            }
            
    with open(dict_path, "w", encoding="utf8") as f:
        json.dump(mapping, f, indent=4)
    print(f"📖 Updated lookup dictionary: {dict_path.name}")
    
def initialize_session(base_path: Path, config: dict,
                       new_needed: Optional[bool] = None,
                       interactive: bool = True) -> Path:
    """
    Create or resume an optimization session folder.

    - base_path: parent folder where Opt_Session_* dirs live
    - config: dict to freeze into session_config.json when creating a new session
    - new_needed:
        - True  -> force CREATE new session
        - False -> force RESUME last session (if none exists, create new)
        - None  -> ask the user (only if interactive=True), otherwise default to resume if exists
    - interactive: allow input() prompts when new_needed is None

    Returns the resolved session_dir (Path).
    """
    base_path = Path(base_path).expanduser()
    base_path.mkdir(parents=True, exist_ok=True)

    # helper: find existing "Opt_Session_*" dirs sorted by modification time
    def _list_sessions():
        sessions = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("Opt_Session_")]
        # sort by mtime ascending (oldest -> newest)
        sessions = sorted(sessions, key=lambda p: p.stat().st_mtime)
        return sessions

    sessions = _list_sessions()
    last_session = sessions[-1] if sessions else None

    # decide create vs resume
    if new_needed is None:
        if interactive:
            if last_session:
                prompt = f"Found {len(sessions)} session(s). Resume last ({last_session.name}) or create NEW? [R/n]: "
                choice = input(prompt).strip().lower()
                do_create = (choice == 'n')
            else:
                choice = input("No existing sessions found. Create NEW session? [Y/n]: ").strip().lower()
                do_create = (choice != 'n')
        else:
            # non-interactive default: resume if available, otherwise create
            do_create = False if last_session else True
    else:
        do_create = bool(new_needed)

    # If resuming but no sessions exist -> fallback to create
    if (not do_create) and (last_session is None):
        do_create = True

    if do_create:
        # create a unique name using timestamp plus a small sequence to avoid dupes
        ts = time.strftime("%Y%m%d_%H%M%S")
        # count sessions created this second to add uniqueness
        same_ts = [d for d in sessions if d.name.startswith(f"Opt_Session_{ts}")]
        seq = len(same_ts) + 1
        session_name = f"Opt_Session_{ts}_{seq:02d}"
        session_dir = base_path / session_name
        session_dir.mkdir(parents=True, exist_ok=False)

        # freeze config into session_config.json
        with open(session_dir / "session_config.json", "w", encoding="utf8") as f:
            json.dump(config, f, indent=4, sort_keys=True)

        # create results/configs dirs
        (session_dir / "results").mkdir(parents=True, exist_ok=True)
        (session_dir / "configs").mkdir(parents=True, exist_ok=True)

        # write a small metadata file
        meta = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_uuid": uuid.uuid4().hex,
            "session_name": session_name
        }
        with open(session_dir / "session_meta.json", "w", encoding="utf8") as f:
            json.dump(meta, f, indent=2)

        print(f"✨ Created NEW session: {session_dir.name}")
    else:
        # resume last session
        session_dir = Path(last_session)
        print(f"🔄 Resuming session: {session_dir.name}")

        # if there is no frozen config file, optionally write the provided config for traceability
        cfg_path = session_dir / "session_config.json"
        if not cfg_path.exists():
            print(f"⚠️ session_config.json missing in {session_dir.name} — writing provided config for trace.")
            with open(cfg_path, "w", encoding="utf8") as f:
                json.dump(config, f, indent=4, sort_keys=True)

    # simple lock mechanism (non-blocking): create a lock file with pid/timestamp
    lock_path = session_dir / "session.lock"
    current_pid = os.getpid()
    now = time.time()
    if lock_path.exists():
        try:
            with open(lock_path, "r", encoding="utf8") as f:
                txt = f.read().strip()
            print(f"⚠️ Warning: lock file already exists: {lock_path.name} -> {txt}")
            # do not error out; allow resume, but inform user
        except Exception:
            pass
    else:
        with open(lock_path, "w", encoding="utf8") as f:
            f.write(json.dumps({"pid": current_pid, "started_at": now, "note": "in-progress"}))

    save_config_dictionary(session_dir)

    return session_dir

# Initialize

# --- DIRECTORY SETUP ---
# This finds the directory where the script ITSELF is located
SCRIPT_DIR = Path(__file__).resolve()

# If Backtest_results is in the project root, go up to the root then down
BASE_DIR = SCRIPT_DIR.parent / "Backtest_results" / "BTC"
DATA_DIR = BASE_DIR / "@Main"
CONFIG_PATH = DATA_DIR / "run_config.json"

RUN_CONFIG = load_master_config_safe(CONFIG_PATH)
BASE_MINUTES = RUN_CONFIG.get("BASE_MINUTES", 5)
SENTINEL_TIME_NS = int(RUN_CONFIG.get("lookahead", {}).get("sentinel_time_ns", -9223372036854775808))
NEW_SESSION_NEEDED = None  # Toggle this to False to resume

# Global Cache Initializations
_WINDOW_LOOKAHEADS_MAX = int(RUN_CONFIG.get("lookahead", {}).get("max_cached_windows", 2))
_LOOKAHEAD_CACHE_MAX = int(RUN_CONFIG.get("cache", {}).get("lookahead_cache_max", 4))
_LOOKBACK_CACHE_MAX = int(RUN_CONFIG.get("cache", {}).get("lookback_cache_max", 4))
_STOCH_CACHE_MAX = int(RUN_CONFIG.get("cache", {}).get("stoch_cache_max", 2))


_WINDOW_LOOKAHEADS = OrderedDict()   # hours -> entry dict (numpy arrays only)
_LOOKAHEAD_CACHE = OrderedDict()     # key -> small ndarray dict
_LOOKBACK_CACHE = OrderedDict()
_STOCH_CACHE = OrderedDict()
RUN_BASE_DIR = initialize_session(BASE_DIR, RUN_CONFIG, new_needed=NEW_SESSION_NEEDED)
combined_output_path = RUN_BASE_DIR / "master_results.parquet"

# 1. First, define a flag to see if we are in "New Session" mode.
# We do this by checking if there are any results in the current session yet.
# If it's brand new, results_dir will be empty.
is_new_session_action = len(list((RUN_BASE_DIR / "results").glob("*.parquet"))) == 0

# 2. Only run the cleanup if it's a NEW session action
if is_new_session_action:
    print(f"🧹 New session detected. Cleaning up failed/incomplete old sessions...")

    # decide whether to prompt for cleanup based on run config's interactive flag
    interactive_cleanup = RUN_CONFIG.get("session", {}).get("interactive", "true") == "true"
    cleanup_threshold = int(RUN_CONFIG.get("session", {}).get("cleanup_session_threshold", 100))

    # Ask the user (second prompt) about deleting uncompleted sessions with fewer than cleanup_threshold cfg files
    do_cleanup = True
    if interactive_cleanup:
        ans = input(f"Delete uncompleted last sessions with fewer than {cleanup_threshold} cfg files? [Y/n]: ").strip().lower()
        do_cleanup = (ans != 'n')

    if do_cleanup:
        for session_path in BASE_DIR.iterdir():
            if session_path.is_dir() and session_path.name.startswith("Opt_Session_"):
                # NEVER delete the session we just created/opened
                if session_path.resolve() == RUN_BASE_DIR.resolve():
                    continue

                configs_dir = session_path / "configs"
                # count config JSON files (cfg files)
                cfg_files = list(configs_dir.glob("*.json")) if configs_dir.exists() else []

                # If the session has fewer than cleanup_threshold cfg files, it's considered "failed/incomplete"
                if len(cfg_files) < cleanup_threshold:
                    try:
                        gc.collect()  # Release file handles
                        time.sleep(0.2)
                        shutil.rmtree(session_path)
                        print(f"🗑 Deleted incomplete session ({len(cfg_files)} cfg files): {session_path.name}")
                    except Exception as e:
                        print(f"⚠️ Could not delete {session_path.name}: {e}")
    else:
        print("🛑 Cleanup skipped by user.")
else:
    print(f"🛡 Resume mode: Protection active. No old sessions will be deleted.")

# %%
# Section 2. Helper
# -------------------------

def _write_chunk_part(df_chunk: pd.DataFrame, results_dir: Path, cfg_id: str, part_idx: int):
    """
    Write a single chunk as a parquet 'part' file to results_dir.
    Caller ensures df_chunk is small-ish and memory-trimmed.
    """
    part_path = results_dir / f"cfg_{cfg_id}_part{part_idx:04d}.parquet"
    # write with pyarrow for speed & minimal memory spike
    table = pa.Table.from_pandas(df_chunk, preserve_index=False)
    pq.write_table(table, str(part_path), compression=RUN_CONFIG.get("session", {}).get("save_compression","snappy"))
    return part_path

# feature_helpers.py
def compute_features_from_config(df: pd.DataFrame, cfg: dict):
    """
    Generate features defined in config. Assumes required MA columns already exist.
    Minimal checks only — fail fast if a required column is missing.
    """
    feat_defs = cfg.get("feature_definitions", {})

    for feat_name, definition in feat_defs.items():
        if feat_name in df.columns:
            continue

        calc_type = definition.get("calc")
        args = definition.get("args", [])

        if calc_type == "pct_div":
            col_a, col_b = args[0], args[1]
            # Fail-fast: we expect the base columns to exist
            assert col_a in df.columns and col_b in df.columns, f"Missing cols for {feat_name}: {col_a}, {col_b}"
            df[feat_name] = ((df[col_a] / df[col_b]) - 1).astype('float32')

    # Stochastic only if requested and not already present
    if cfg.get("use_stochastic", False) and '%DSlow' not in df.columns:
        s_cfg = cfg.get("stochastic", {})
        K, D, DSlow = calculate_stochastic(
            df,
            k=int(s_cfg.get("stoch_k", 12)),
            d=int(s_cfg.get("stoch_d", 12)),
            slow=int(s_cfg.get("stoch_slow", 8))
        )
        df['%K'], df['%D'], df['%DSlow'] = K.astype('float32'), D.astype('float32'), DSlow.astype('float32')

    return df

def downcast_numeric_inplace(df):
    """
    Aggressively downcasts all columns. 
    If it's a float, it becomes float32. If it's an int, it finds the smallest fit.
    """
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def safe_free(*objs):
    """Explicitly deletes objects and triggers garbage collection."""
    for o in objs:
        try: del o
        except: pass
    gc.collect()

def _ts_for_key(ts):
    """Stable integer keys for caching."""
    if pd.isna(ts): return None
    try: return int(ts.value)
    except:
        try: return int(pd.to_datetime(ts).value)
        except: return None

# ---- I/O helpers ----
def find_first_csv(folder, pattern="*.csv"):
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder} matching {pattern}")
    return files[0]

# ---- Datetime helpers ----
def to_datetime_series(s):
    """
    Convert a column/series to timezone-aware datetimes.
    Handles numeric epochs in ms or s.
    """
    if pd.api.types.is_numeric_dtype(s):
        maxv = int(np.nanmax(s.fillna(0)))
        # heuristics for epoch unit
        if maxv > 1_000_000_000_000:
            return pd.to_datetime(s, unit='ms', utc=True, errors='coerce')
        if maxv > 1_000_000_000:
            return pd.to_datetime(s, unit='s', utc=True, errors='coerce')
        return pd.to_datetime(s, unit='ms', utc=True, errors='coerce')
    return pd.to_datetime(s, utc=True, errors='coerce')

def _fmt_elapsed(seconds):
    seconds = int(round(seconds))
    days, rem = divmod(seconds, 86400)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


# ---- Simple indicators ----
def sma(series, length, ta=None):
    """
    Simple moving average. If `ta` (ta-lib) is provided it will be used.
    """
    length = int(max(1, length))
    if ta:
        arr = ta.SMA(series.values.astype(float), timeperiod=length)
        return pd.Series(arr, index=series.index)
    return series.rolling(length, min_periods=1).mean()



# ---- Stochastic oscillator ----
def calculate_stochastic(df, k=12, d=12, slow=3):
    """
    Returns (%K, %D, %DSlow) as float32 Series.
    Uses time_ns for high-speed caching.
    """
    # 1. Quick Cache Check
    n = len(df)
    # Use time_ns directly—it's much faster than parsing 'time' objects
    t0 = t1 = 0 

    if n > 0:
        # We cast to int because sometimes .iat can return a numpy.int64 
        # and we want a standard Python int for the most reliable hashing
        t0 = int(df['time_ns'].iat[0])
        t1 = int(df['time_ns'].iat[-1])

    key = (n, t0, t1, k, d, slow)
    
    if key in _STOCH_CACHE:
        return _STOCH_CACHE[key]

    # 2. Compute Indicators
    # Rolling min/max on High/Low
    low_k = df['low'].rolling(k, min_periods=1).min()
    high_k = df['high'].rolling(k, min_periods=1).max()
    
    # %K calculation (handle div by zero by filling with 50 or 0)
    denom = (high_k - low_k)
    K = 100 * (df['close'] - low_k) / denom.replace(0, np.nan)
    K = K.fillna(50).astype('float32') # Default to neutral 50 if no range
    
    # Smoothing for %D and %DSlow
    D = K.rolling(d, min_periods=1).mean().astype('float32')
    DSlow = D.rolling(slow, min_periods=1).mean().astype('float32')

    # 3. Cache & Return
    results = (K, D, DSlow)
    _STOCH_CACHE[key] = results
    return results

def _compute_rolling_breakout(df, lb_hours, base_minutes=5):
    """
    Robust rolling breakout_strength (0..1), shifted to avoid lookahead.
    Returns a Series aligned to df.index. Uses a small in-memory cache.
    """
    # quick validation
    if 'time' not in df.columns:
        raise KeyError("df must contain 'time' column for lookback breakout")

    # normalize & copy
    d = df.reset_index(drop=True).copy()
    d = d.sort_values('time').reset_index(drop=True)

    # coerce numeric - this guards against object/string dtypes
    for c in ('high','low','close'):
        if c not in d.columns:
            raise KeyError(f"Missing required column for breakout: {c}")
        d[c] = pd.to_numeric(d[c], errors='coerce')

    # cache key using integer ns timestamps
    first_ts = _ts_for_key(d['time'].iat[0]) if len(d) else None
    last_ts  = _ts_for_key(d['time'].iat[-1]) if len(d) else None
    key = (len(d), lb_hours, base_minutes, first_ts, last_ts)
    if key in _LOOKBACK_CACHE:
        return _LOOKBACK_CACHE[key].reindex(df.index)

    periods = int((lb_hours * 60) / base_minutes)
    if periods <= 0:
        s = pd.Series(np.nan, index=df.index)
        _LOOKBACK_CACHE[key] = s
        return s

    # rolling over fixed number of rows (since df is regular 5-min resample)
    rolling_high = d['high'].rolling(periods, min_periods=1).max().shift(1)
    rolling_low  = d['low'].rolling(periods, min_periods=1).min().shift(1)

    denom = rolling_high - rolling_low
    # protect divide-by-zero or ill-defined windows
    denom_zero = denom == 0
    denom = denom.replace(0, np.nan)

    breakout = ((d['close'] - rolling_low) / denom)
    # set explicit NaN where denom was zero or rolling_high/low are NaN
    breakout[denom_zero | rolling_high.isna() | rolling_low.isna()] = np.nan

    # align back to original index (preserve original index)
    breakout = pd.Series(breakout.values, index=d.index, name='breakout_strength')
    # map it back to original df index (use location alignment)
    out = pd.Series(np.nan, index=df.index)
    out.iloc[:len(breakout)] = breakout.values

    _LOOKBACK_CACHE[key] = out
    return out

def generate_filtered_signals(df: pd.DataFrame, cfg: dict, base_minutes: int = 5) -> pd.DataFrame:
    """
    Minimal, fast signal filter.
    Returns columns: time, time_ns, era_int [, era_day_int], open, high, low, close, features...
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    buy_mask = np.ones(n, dtype=bool)
    sell_mask = np.ones(n, dtype=bool)

    # Stochastic (only if requested)
    stoch_cfg = cfg.get("stochastic", {})
    if stoch_cfg.get("use_stochastic", cfg.get("use_stochastic", False)):
        if '%DSlow' in df.columns:
            dslow = df['%DSlow']
        else:
            _, _, dslow = calculate_stochastic(df,
                                               k=cfg.get("stoch_k", 12),
                                               d=cfg.get("stoch_d", 12),
                                               slow=cfg.get("stoch_slow", 8))
        buy_mask &= (dslow < 80)
        sell_mask &= (dslow > 20)

    # Entry lookback breakout
    lb_h = int(cfg.get("entry_lookback_h", cfg.get("lookback_breakout_hours", 0)))
    if cfg.get("use_entry_lookback") and lb_h > 0:
        breakout = _compute_rolling_breakout(df, lb_h, base_minutes)
        buy_mask &= (breakout >= 1.0)
        sell_mask &= (breakout <= 0.0)

    # MA trend filters (bit-packed ma_int)
    ma_int = int(cfg.get("ma_int", 0))
    ma_cols = ["ma_5min", "ma_hour", "ma_day", "ma_week"]
    is_rev = bool(cfg.get("ma_reversion", False))
    for i, col_name in enumerate(ma_cols):
        if ((ma_int >> i) & 1) and (col_name in df.columns):
            if not is_rev:
                buy_mask &= (df['close'] > df[col_name])
                sell_mask &= (df['close'] < df[col_name])
            else:
                buy_mask &= (df['close'] < df[col_name])
                sell_mask &= (df['close'] > df[col_name])

    mask = buy_mask | sell_mask
    if not mask.any():
        return pd.DataFrame()

    features = cfg.get("feature_columns", [])
    core_cols = ['time', 'time_ns', 'era_int', 'open', 'high', 'low', 'close'] + features
    # Include era_day if the dataset has it — this avoids recomputation later
    if 'era_day_int' in df.columns:
        core_cols.insert(3, 'era_day_int')  # keep near era_int

    core_cols = [c for c in core_cols if c in df.columns]

    out = df.loc[mask, core_cols].copy()
    out['side'] = np.int8(0)
    out.loc[buy_mask & mask, 'side'] = np.int8(1)
    out.loc[sell_mask & mask, 'side'] = np.int8(-1)

    # minimal casting
    out['position_type'] = out['side'].astype(np.int8)
    out['entry_lookback_h'] = np.uint16(cfg.get("entry_lookback_h", 0))
    out['ma_reversion'] = np.int8(1 if is_rev else 0)
    out['ma_int'] = np.int8(ma_int)
    out['time_ns'] = out['time_ns'].astype(np.int64)
    out['era_int'] = out['era_int'].astype(np.int64)
    if 'era_day_int' in out.columns:
        out['era_day_int'] = out['era_day_int'].astype(np.int64)

    # cast features to float32 if present
    for col in features:
        if col in out.columns:
            out[col] = out[col].astype(np.float32)

    out.sort_values('time', inplace=True)
    return out.reset_index(drop=True)

# ---- Config generation / persistence ----
def generate_configs(base_config, run_base_dir: Path):
    cfg_dir = run_base_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (run_base_dir / "results").mkdir(parents=True, exist_ok=True) # Ensure results dir exists

    lookbacks = base_config.get("entry_lookback_list_hours", [24])
    
    idx = 0
    saved_paths = [] # <--- 1. Initialize list
    
    for ma_rev in [False, True]:
        for ma_int in range(16): 
            for use_stoch in [False, True]:
                for use_lb in [False, True]:
                    for lb_hours in (lookbacks if use_lb else [0]):
                        
                        cfg = copy.deepcopy(base_config)
                        # Clean up template keys
                        for k in ['use_ma_5min', 'use_ma_hour', 'use_ma_day', 'use_ma_week']:
                            cfg.pop(k, None)
                        
                        cfg.update({
                            "config_id": f"{idx:04d}",
                            "ma_int": ma_int,
                            "ma_reversion": ma_rev,
                            "use_stochastic": use_stoch,
                            "use_entry_lookback": use_lb,
                            "entry_lookback_h": lb_hours
                        })

                        file_path = cfg_dir / f"cfg_{idx:04d}.json"
                        with open(file_path, "w") as f:
                            json.dump(cfg, f, indent=2)
                        
                        saved_paths.append(file_path) # <--- 2. Store path
                        idx += 1
                        
    print(f"✅ Generated {idx} unique strategy configurations.")
    return saved_paths # <--- 3. Return the list

def combine_and_save_results(session_dir: Path, output_filename="master_results.parquet", verbose=True, preview_only=10):
    """
    Concatenate all per-config parquet outputs into a single master parquet.
    Assumes each result file already contains era_int and config_id.
    """
    results_dir = session_dir / "results"
    output_file = session_dir / output_filename

    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        if verbose:
            print("⚠️ No result parquets found.")
        return pd.DataFrame()

    parts = []
    for p in files:
        try:
            df = pd.read_parquet(p)
            if df.empty:
                continue

            # ensure config_id exists (derive from filename if not present)
            if 'config_id' not in df.columns:
                stem = p.stem
                # expecting pattern cfg_XXXX_results
                m = re.search(r'cfg_(\d{4})', stem)
                df['config_id'] = m.group(1) if m else stem

            # ensure era_int type
            if 'era_int' in df.columns:
                df['era_int'] = df['era_int'].astype(np.int64)

            parts.append(df)
        except Exception as e:
            if verbose:
                print(f"❌ Failed to read {p.name}: {e}")

    if not parts:
        if verbose:
            print("⚠️ No valid data read.")
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True, sort=False)

    # compact common numeric columns
    for col in ('score', 'win_pos', 'total_pos', 'exit_window_h'):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], downcast='integer')

    # sort by era -> config -> score desc
    sort_cols = ['era_int', 'config_id']
    if 'score' in combined.columns:
        combined = combined.sort_values(sort_cols + ['score'], ascending=[True, True, False])
    else:
        combined = combined.sort_values(sort_cols, ascending=[True, True])

    # atomic write
    _atomic_parquet_save(combined, session_dir / output_filename)

    if verbose:
        print(f"✅ Master saved: {output_filename} ({len(combined)} rows from {len(parts)} files)")

    return combined.head(preview_only)

def safe_save_all_to_disk(session_dir: Path, out_filename="master_database.parquet" , run_config=RUN_CONFIG):
    """
    Stream all per-config parquet files into one large parquet on disk.
    Fixes Schema Mismatch by forcing consistent dtypes.
    """

    features = run_config.get("feature_columns", [])
    results_dir = session_dir / "results"
    out_path = session_dir / out_filename
    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        print("No files to stream.")
        return

    side_map = {'buy': 1, 'sell': -1}

    # 1. Define Core Dtypes
    dtype_map = {
        'score': np.int32, 'SL': np.int32, 'TP': np.int32,
        'win_pos': np.int32, 'total_pos': np.int32,
        'exit_window_h': np.uint16, 'era_int': np.int64,
        'ma_int': np.int8, 'ma_reversion': np.int8,
        'config_id': np.int32, 'side': np.int8
    }

    # 2. Add Dynamic Features from Config
    # If config says 'rsi', we look for 'avg_rsi' in the result files
    features = run_config.get("feature_columns", [])
    for feat in features:
        dtype_map[f'avg_{feat}'] = np.float32

    writer = None
    try:
        for p in tqdm(files, desc="Streaming to master"):
            df_tmp = pd.read_parquet(p)
            if df_tmp.empty: continue

            if 'config_id' in df_tmp.columns:
                # If it's a string like "cfg_0132", extract just the numbers
                if df_tmp['config_id'].dtype == object:
                    df_tmp['config_id'] = df_tmp['config_id'].astype(str).str.extract(r'(\d+)').fillna(0).astype(np.int32)

            if 'side' in df_tmp.columns and df_tmp['side'].dtype == object:
                # Map strings to numbers if they haven't been converted yet
                df_tmp['side'] = df_tmp['side'].astype(np.int8)
        
            for col, dtype in dtype_map.items():
                if col in df_tmp.columns:
                    df_tmp[col] = df_tmp[col].astype(dtype)
                elif col.startswith('avg_'):
                    # If a feature is missing in a specific file, fill with NaN
                    df_tmp[col] = np.float32(np.nan)
                    df_tmp[col] = df_tmp[col].astype(np.float32)
        
            # 3. Stream to Disk
            table = pa.Table.from_pandas(df_tmp, preserve_index=False)
            if writer is None:
                # The first file defines the "Law of the Land" for the rest of the file
                writer = pq.ParquetWriter(str(out_path), table.schema, compression='snappy')
            
            writer.write_table(table)

            del df_tmp, table; gc.collect()
    finally:
        if writer:
            writer.close()

    print(f"✅ Stream-saved master file to: {out_path}")



# %%
# Section 3. Agg Calc & Search for Dataframe

PARQUET_PATH = DATA_DIR / "BTC_5m_processed_main.parquet"
resample_rule = f"{BASE_MINUTES}min"

if PARQUET_PATH.exists():
    print(f"✅ Loading processed data: {PARQUET_PATH}")
    df_main = pd.read_parquet(PARQUET_PATH)
else:
    # --- START CSV PROCESSING ---
    csv_path = find_first_csv(DATA_DIR, "*.csv")
    print(f"📂 Processing CSV: {csv_path}")
    
    df_raw = pd.read_csv(csv_path, low_memory=False)
    
    # 1) Column Mapping & Canonicalization
    mapping = {'open time': 'time', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
    lower_cols = {c.lower(): c for c in df_raw.columns}
    df_raw.rename(columns={lower_cols[k]: v for k, v in mapping.items() if k in lower_cols}, inplace=True)
    
    for c in ('open', 'high', 'low', 'close', 'volume'):
        df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce').astype('float32')
    
    df_raw['time'] = to_datetime_series(df_raw['time'])
    df_raw.dropna(subset=['time', 'close'], inplace=True)

    # 2) Resample & Basic Time Metadata
    agg_logic = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    df_main = df_raw.set_index('time').resample(resample_rule).agg(agg_logic).dropna(subset=['close']).reset_index()
    df_main['time'] = pd.to_datetime(df_main['time'], utc=True)
    df_main['time_ns'] = df_main['time'].values.astype('int64')

    # 3) Compute Moving Averages (Based on Config ma_periods)
    mp = RUN_CONFIG.get("ma_periods", {})
    df_hour = df_main.set_index('time').resample('1h').agg({'close': 'last'}).dropna().reset_index()
    
    # Compute MAs on hourly basis for higher order trends
    df_hour['ma_hour'] = sma(df_hour['close'], int(mp.get("ma_hour", 50))).shift(1)
    df_hour['ma_day']  = sma(df_hour['close'], int(mp.get("ma_day", 1200))).shift(1)
    df_hour['ma_week'] = sma(df_hour['close'], int(mp.get("ma_week", 8400))).shift(1)
    
    # Merge hourly MAs back to 5m DF
    df_main['hour_time'] = df_main['time'].dt.floor('h')
    df_main = df_main.merge(
        df_hour[['time', 'ma_hour', 'ma_day', 'ma_week']].rename(columns={'time': 'hour_time'}), 
        on='hour_time', how='left'
    )
    
    # Local 5min MA
    df_main['ma_5min'] = sma(df_main['close'], int(mp.get("ma_5min", 50))).shift(1)
    df_main.drop(columns=['hour_time'], inplace=True)

    # 4) DYNAMIC FEATURE CALCULATION (Zero Hardcoding)
    # Uses 'feature_definitions' from Config
    feat_defs = RUN_CONFIG.get("feature_definitions", {})
    for feat_name, definition in feat_defs.items():
        calc_type = definition.get("calc")
        args = definition.get("args", [])
        
        try:
            if calc_type == "pct_div":
                # Formula: (A / B) - 1
                col_a, col_b = args[0], args[1]
                df_main[feat_name] = ((df_main[col_a] / df_main[col_b]) - 1).astype('float32')
            
            # You can easily add more calc types here (e.g., 'diff', 'rsi')
        except Exception as e:
            print(f"⚠️ Failed to compute dynamic feature {feat_name}: {e}")

    # 5) Stochastic (Check config)
    stoch_cfg = RUN_CONFIG.get("stochastic", {})
    K, D, DSlow = calculate_stochastic(
        df_main, 
        k=int(stoch_cfg.get("stoch_k", 12)), 
        d=int(stoch_cfg.get("stoch_d", 12)), 
        slow=int(stoch_cfg.get("stoch_slow", 8))
    )
    df_main['%K'], df_main['%D'], df_main['%DSlow'] = K.astype('float32'), D.astype('float32'), DSlow.astype('float32')

    # era_int encodes year/month/day/hour as YYYYMMDDHH (so it is unique for each hour)
    df_main['era_int'] = (
        df_main['time'].dt.year * 1000000 + 
        df_main['time'].dt.month * 10000 + 
        df_main['time'].dt.day * 100 + 
        df_main['time'].dt.hour
        ).astype(np.int64)

    df_main['era_day_int'] = (
        df_main['time'].dt.year * 10000 +
        df_main['time'].dt.month * 100 +
        df_main['time'].dt.day
        ).astype(np.int32)

    # 7) Final Cleanup & Save
    df_main = downcast_numeric_inplace(df_main)
    df_main.to_parquet(PARQUET_PATH, compression='snappy')
    print(f"✅ Data processed and saved to {PARQUET_PATH}")

# %%
# --- Addon: Column Names and Datatypes ---
print("\n--- Column Data Types ---")
print(df_main.dtypes)
print("-" * 25)

print(f"Total rows in df_main: {len(df_main)}")
print(f"Data Range: {df_main['time'].min()} to {df_main['time'].max()}")

# Check how many signals are generated before optimization
test_cfg = RUN_CONFIG.copy()
test_cfg['ma_int'] = 15 # Test with all MAs active
signals = generate_filtered_signals(df_main, test_cfg)

print(f"Total signals generated: {len(signals)}")
if not signals.empty:
    print(f"Signal Date Range: {signals['time'].min()} to {signals['time'].max()}")
    # Optional: Check signal dtypes if they differ from df_main
    # print(signals.dtypes)

# %%
# Section 4. Optimizer core (lookahead + grid scan)

@njit
def _forward_lookahead_numba(low, high, window):
    n = low.shape[0]
    # Match your df_main float32 precision for RAM efficiency
    next_min = np.full(n, np.nan, dtype=np.float32)
    next_max = np.full(n, np.nan, dtype=np.float32)
    next_min_idx = np.full(n, -1, dtype=np.int64)
    next_max_idx = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        start = i + 1
        if start >= n: continue
        end = min(i + window, n - 1)
        
        minv = low[start]; minpos = start
        maxv = high[start]; maxpos = start
        for j in range(start + 1, end + 1):
            if low[j] < minv:
                minv = low[j]; minpos = j
            if high[j] > maxv:
                maxv = high[j]; maxpos = j
        next_min[i] = minv; next_min_idx[i] = minpos
        next_max[i] = maxv; next_max_idx[i] = maxpos
    return next_min, next_min_idx, next_max, next_max_idx
        
@njit
def _grid_kernel_numba(closes_f32, n_min_f32, n_max_f32,
                       t_min_ns_i64, t_max_ns_i64,
                       s_vals_f32, t_vals_f32,
                       breakeven_after_f32,
                       min_rr_f32,      # NEW param
                       tie_mode_i32, position_type_i32):
    n_rows = closes_f32.shape[0]
    ns = s_vals_f32.shape[0]
    nt = t_vals_f32.shape[0]
    good = np.zeros(ns * nt, dtype=np.int32)
    be = np.zeros(ns * nt, dtype=np.int32)
    
    for i in range(ns):
        sl_val = s_vals_f32[i]
        for j in range(nt):
            tp_val = t_vals_f32[j]

            # MIN-RR pruning: skip combos where TP < SL * min_rr
            if tp_val < (sl_val * min_rr_f32):
                # leave good/be as zero for this combo (it's effectively skipped)
                continue

            count_good = 0
            count_be = 0
            
            for r in range(n_rows):
                # Sentinel check
                if t_min_ns_i64[r] < 0: continue

                # Corrected Tie Logic
                if position_type_i32 == 1: # BUY
                    t_tp, t_sl = t_max_ns_i64[r], t_min_ns_i64[r]
                    dist_tp = n_max_f32[r] - closes_f32[r]  # Profit move
                    dist_sl = closes_f32[r] - n_min_f32[r]  # Adverse move
                else: # SELL
                    t_tp, t_sl = t_min_ns_i64[r], t_max_ns_i64[r]
                    dist_tp = closes_f32[r] - n_min_f32[r]  # Profit move
                    dist_sl = n_max_f32[r] - closes_f32[r]  # Adverse move

                hit_tp = dist_tp >= tp_val
                hit_sl = dist_sl >= sl_val

                tp_wins = False
                if hit_tp and hit_sl:
                    if tie_mode_i32 == 1: # TP wins tie
                        tp_wins = (t_tp <= t_sl)
                    else: # SL wins tie
                        tp_wins = (t_tp < t_sl)
                elif hit_tp:
                    tp_wins = True

                if tp_wins:
                    count_good += 1
                elif dist_tp >= (tp_val - breakeven_after_f32):
                    count_be += 1
            
            good[i * nt + j] = count_good
            be[i * nt + j] = count_be
    return good, be
    
def _forward_lookahead_py(low, high, window):
    n = low.shape[0]
    next_min = np.full(n, np.nan, dtype=np.float32)
    next_max = np.full(n, np.nan, dtype=np.float32)
    next_min_idx = np.full(n, -1, dtype=np.int32)
    next_max_idx = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        start = i + 1
        if start >= n:
            continue
        end = i + window
        if end >= n:
            end = n - 1
        minv = low[start]; minpos = start
        maxv = high[start]; maxpos = start
        for j in range(start + 1, end + 1):
            lv = low[j]
            if lv < minv:
                minv = lv; minpos = j
            hv = high[j]
            if hv > maxv:
                maxv = hv; maxpos = j
        next_min[i] = minv; next_min_idx[i] = minpos
        next_max[i] = maxv; next_max_idx[i] = maxpos
    return next_min, next_min_idx, next_max, next_max_idx

def add_lookahead_columns(df, window_input, base_minutes=5, input_is_hours=True, use_cache=True):
    """
    Produces minimal lookahead arrays and returns a filtered DataFrame.
    Fixed: Scope issues with local variables when using cache.
    """
    if input_is_hours:
        periods_per_hour = int(60 // base_minutes)
        window_periods = int(window_input * periods_per_hour)
    else:
        window_periods = int(window_input)
    
    if window_periods <= 0:
        raise ValueError("window_periods must be > 0")

    n = len(df)
    if n == 0:
        return df

    downcast_numeric_inplace(df)

    if 'time_ns' not in df.columns:
        if 'time' in df.columns:
            df['time_ns'] = pd.to_datetime(df['time'], utc=True).values.astype(np.int64)
        else:
            raise KeyError("DataFrame must contain 'time' or 'time_ns' to calculate lookahead.")

    # Use .values or .to_numpy(copy=False) for speed
    low = df['low'].values.astype(np.float32)
    high = df['high'].values.astype(np.float32)
    times_ns = df['time_ns'].values.astype(np.int64)

    key = (n, window_periods, int(times_ns[0]) if n else 0, int(times_ns[-1]) if n else 0)

    # --- 1. TRY CACHE ---
    if use_cache and key in _LOOKAHEAD_CACHE:
        cached = _LOOKAHEAD_CACHE[key]
        _LOOKAHEAD_CACHE.move_to_end(key, last=True)
        df['next_min'] = cached['next_min'].copy()
        df['next_max'] = cached['next_max'].copy()
        df['next_min_time_ns'] = cached['next_min_time_ns'].copy()
        df['next_max_time_ns'] = cached['next_max_time_ns'].copy()
    
    # --- 2. CALCULATE IF NOT IN CACHE ---
    else:
        nmin_vals, nmin_idx, nmax_vals, nmax_idx = _forward_lookahead_numba(low, high, window_periods)

        # Map indices to actual nanosecond timestamps
        next_min_time_ns = np.full(n, SENTINEL_TIME_NS, dtype=np.int64)
        next_max_time_ns = np.full(n, SENTINEL_TIME_NS, dtype=np.int64)
        
        # Vectorized mapping is faster than a loop if n is large, 
        # but keeping your loop logic for index safety:
        for i in range(n):
            mi, xi = int(nmin_idx[i]), int(nmax_idx[i])
            if mi >= 0: next_min_time_ns[i] = times_ns[mi]
            if xi >= 0: next_max_time_ns[i] = times_ns[xi]

        # Assign to DF
        df['next_min'] = nmin_vals.astype(np.float32)
        df['next_max'] = nmax_vals.astype(np.float32)
        df['next_min_time_ns'] = next_min_time_ns
        df['next_max_time_ns'] = next_max_time_ns

        # WRITE TO CACHE (Only here because we have the local variables available)
        if use_cache:
            _LOOKAHEAD_CACHE[key] = {
                'next_min': df['next_min'].values.copy(),
                'next_max': df['next_max'].values.copy(),
                'next_min_time_ns': df['next_min_time_ns'].values.copy(),
                'next_max_time_ns': df['next_max_time_ns'].values.copy(),
            }
            # Add these two lines here:
            _LOOKAHEAD_CACHE.move_to_end(key, last=True)
            _evict_generic_cache(_LOOKAHEAD_CACHE, _LOOKAHEAD_CACHE_MAX)

    # --- 3. FILTER AND RETURN ---
    # 1. Create the mask to identify rows that actually have a full future window
    valid_mask = (~np.isnan(df['next_min'])) & (~np.isnan(df['next_max']))
    
    # 2. Filter the dataframe using the mask. 
    # IMPORTANT: We keep the original index to allow alignment later!
    out = df.loc[valid_mask].copy() 
    
    # 3. Add human-friendly times for the subset (useful for debugging)
    out['next_min_time'] = pd.to_datetime(out['next_min_time_ns'], unit='ns', errors='coerce')
    out['next_max_time'] = pd.to_datetime(out['next_max_time_ns'], unit='ns', errors='coerce')
    
    return out

def compute_window_lookahead_once(hours, base_minutes=5, use_cache=True):
    """
    Returns a dict with numpy arrays for the lookahead of the *full* dataset.
    Caches only arrays (not the entire unneeded DataFrame) and evicts old windows LRU-style.
    """
    if hours in _WINDOW_LOOKAHEADS:
        _WINDOW_LOOKAHEADS.move_to_end(hours, last=True)
        return _WINDOW_LOOKAHEADS[hours]

    # Build a working DataFrame from the preloaded df (we let add_lookahead_columns fill new cols)
    needed_cols = ['low', 'high', 'time_ns', 'time']
    df_work = df_main[needed_cols].copy()
    full = add_lookahead_columns(df_work, window_input=hours, base_minutes=base_minutes, use_cache=True)
    if full is None or full.empty:
        return None

    entry = {
        'next_min': full['next_min'].to_numpy(dtype=np.float32, copy=True),
        'next_max': full['next_max'].to_numpy(dtype=np.float32, copy=True),
        'next_min_time_ns': full['next_min_time_ns'].to_numpy(dtype=np.int64, copy=True),
        'next_max_time_ns': full['next_max_time_ns'].to_numpy(dtype=np.int64, copy=True),
        'time': full['time'].to_numpy(dtype='datetime64[ns]', copy=True),
        'index_map': full.index.to_numpy(copy=True)
    }

    _WINDOW_LOOKAHEADS[hours] = entry
    _WINDOW_LOOKAHEADS.move_to_end(hours, last=True)
    #_evict_old_window_if_needed()
    
    del full, df_work
    gc.collect()
    return entry

# optimize single window (both sides)
def optimize_window(df_signals, window_hours, sl_min, sl_max, tp_min, tp_max, point_interval,
                    tie_buy='sl', tie_sell='tp', show_progress=False, min_rr=1.0):
    start = time.time()

    df = df_signals
    for c in ['time', 'open', 'high', 'low', 'close']:
        if c not in df.columns:
            raise KeyError(f"optimize_window requires column '{c}'")

    if 'side' not in df.columns and 'position_type' in df.columns:
        df['side'] = df['position_type']
    if 'position_type' not in df.columns:
        df['position_type'] = df.get('side', np.nan)
    if 'cs_color' not in df.columns:
        df['cs_color'] = df.get('side', 'buy').map({'buy': 'green', 'sell': 'red'}).fillna('green')

    required_look = ('next_min', 'next_max')
    if not all(c in df.columns for c in required_look):
        raise KeyError("optimize_window expects lookahead numeric columns present")

    buy_df = df[df['side'] == 1]
    sell_df = df[df['side'] == -1]

    s_count, t_count, combos_per_side, _, _ = count_combinations(sl_min, sl_max, tp_min, tp_max, point_interval)

    total_window_combos = combos_per_side * ((0 if buy_df.empty else 1) + (0 if sell_df.empty else 1))
    pbar = None
    if show_progress and total_window_combos > 0:
        pbar = tqdm(total=total_window_combos, desc=f"{window_hours}h window", unit="combo")

    parts = []
    if not buy_df.empty:
        rbuy = sl_tp_with_progress(buy_df, 1,
                           sl_min=sl_min, sl_max=sl_max, tp_min=tp_min, tp_max=tp_max,
                           point_interval=point_interval, tie_breaker=tie_buy,
                           show_progress=False, pbar=pbar, verbose=False, min_rr=min_rr)
        if rbuy is not None and not rbuy.empty:
            rbuy['side'] = 'buy'
            parts.append(rbuy)

    if not sell_df.empty:
        rsell = sl_tp_with_progress(sell_df, -1,
                            sl_min=sl_min, sl_max=sl_max, tp_min=tp_min, tp_max=tp_max,
                            point_interval=point_interval, tie_breaker=tie_sell,
                            show_progress=False, pbar=pbar, verbose=False, min_rr=min_rr)
        if rsell is not None and not rsell.empty:
            rsell['side'] = 'sell'
            parts.append(rsell)

    if pbar is not None:
        pbar.close()

    # if no grid-combos produced, return empty frame with expected columns
    out = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame(columns=['score','SL','TP','win_pos','total_pos','side'])
    out['exit_window_h'] = int(window_hours)

    # Do NOT attach start_time/end_time here. The driver will assign era_int for the slice.
    return out

# grid helpers + sl/tp scan (explicit ranges)
def count_combinations(sl_min, sl_max, tp_min, tp_max, point_interval):
    s_vals = np.arange(sl_min, sl_max + 0.0001, point_interval)
    t_vals = np.arange(tp_min, tp_max + 0.0001, point_interval)
    return len(s_vals), len(t_vals), len(s_vals) * len(t_vals), s_vals, t_vals

def sl_tp_with_progress(input_df, position_type,
                        sl_min=20, sl_max=20, tp_min=20, tp_max=20,
                        point_interval=40, breakeven_after=0.0, tie_breaker='sl',
                        show_progress=False, pbar=None, verbose=False, min_rr=1.0):
    """
    Scan SL/TP grid for a given side.

    This version removes the Python fallback and requires the numba kernel.
    """

    s_count, t_count, total_combos, s_vals, t_vals = count_combinations(
        sl_min, sl_max, tp_min, tp_max, point_interval
    )

    closes = input_df['close'].values.astype(np.float32)
    n_min = input_df['next_min'].values.astype(np.float32)
    n_max = input_df['next_max'].values.astype(np.float32)
    t_min = input_df['next_min_time_ns'].values.astype(np.int64)
    t_max = input_df['next_max_time_ns'].values.astype(np.int64)
    entry_ns = input_df['time_ns'].values.astype(np.int64)

    tie_mode = 1 if tie_breaker == 'tp' else 0 if tie_breaker == 'sl' else 2
    pos_flag = 1 if position_type == 1 else -1

    # sentinel protection: drop rows where either next_min or next_max equals sentinel
    sentinel = SENTINEL_TIME_NS
    valid_time_mask = (t_min != sentinel) & (t_max != sentinel)
    total_pos = int(np.count_nonzero(valid_time_mask))

    # For time-range reporting, decide exit times per side but use valid_time_mask (not filtered by lead)
    if pos_flag == 1:  # buy: exit if TP/time_max occurs
        exit_times_ns = t_max
    else:  # sell: exit if TP/time_min occurs for sells
        exit_times_ns = t_min

    if total_pos > 0:
        start_ns = int(entry_ns[valid_time_mask].min())
        end_ns = int(exit_times_ns[valid_time_mask].max())
        start_ts = pd.to_datetime(start_ns, unit='ns', utc=True)
        end_ts = pd.to_datetime(end_ns, unit='ns', utc=True)
    else:
        start_ts = pd.NaT
        end_ts = pd.NaT

    num_rows = len(closes)

    good_arr, be_arr = _grid_kernel_numba(
        closes.astype(np.float32),
        n_min.astype(np.float32),
        n_max.astype(np.float32),
        t_min.astype(np.int64),
        t_max.astype(np.int64),
        s_vals.astype(np.float32),
        t_vals.astype(np.float32),
        np.float32(breakeven_after),
        np.float32(min_rr),
        np.int32(tie_mode),
        np.int32(pos_flag),
    )

    # mesh to flat arrays consistent with earlier code
    sl_grid, tp_grid = np.meshgrid(s_vals, t_vals, indexing='ij')
    sl_flat = sl_grid.ravel()
    tp_flat = tp_grid.ravel()

    bad_arr = total_pos - (good_arr + be_arr)
    scores = (good_arr * tp_flat) + (be_arr * breakeven_after) - (bad_arr * sl_flat)

    df_out = pd.DataFrame({
        'score': scores.astype(np.int64),
        'SL': sl_flat.astype(np.float32),
        'TP': tp_flat.astype(np.float32),
        'win_pos': good_arr.astype(np.int32),
        'total_pos': np.int32(total_pos)
    })

    # update progress bar for all combos if provided (numba block produced all combos at once)
    if pbar is not None:
        try:
            pbar.update(int(total_combos))
        except Exception:
            pass

    return df_out

# %%
# Section 5. Optimizer File Save & Minor Calc

def _sanitize_for_json(obj):
    if isinstance(obj, set): return sorted(list(obj))
    if isinstance(obj, (list, tuple)): return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict): return {str(k): _sanitize_for_json(v) for k,v in obj.items()}
    try:
        import numpy as _np
        if isinstance(obj, _np.generic): return obj.item()
    except Exception:
        pass
    return obj
    
def _atomic_parquet_save(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(str(tmp), str(path))
    except Exception as e:
        tmp_csv = tmp.with_suffix(".csv.tmp")
        df.to_csv(tmp_csv, index=False)
        final_csv = path.with_suffix(".csv")
        os.replace(str(tmp_csv), str(final_csv))
        print(f"[warning] parquet write failed; saved CSV: {final_csv} (error: {e})")

def _make_comp_cfg_and_tag(run_config):
    def _san(x):
        if isinstance(x, set): return sorted(list(x))
        if isinstance(x, (list, tuple)): return list(x)
        if isinstance(x, dict): return {str(k): _san(v) for k,v in x.items()}
        try:
            import numpy as _np
            if isinstance(x, _np.generic): return x.item()
        except Exception:
            pass
        return x
    comp = {
        "BTC_SETTINGS": _san(run_config.get("BTC_SETTINGS", {})),
        "exit_windows": _san(run_config.get("exit_windows", [])),
        "use_ma_5min": bool(run_config.get("use_ma_5min", False)),
        "use_ma_hour": bool(run_config.get("use_ma_hour", False)),
        "use_ma_day": bool(run_config.get("use_ma_day", False)),
        "use_ma_week": bool(run_config.get("use_ma_week", False)),
        "use_stochastic": bool(run_config.get("use_stochastic", False)),
        "use_entry_lookback": bool(run_config.get("use_entry_lookback", True)),
        "stoch_k": int(run_config.get("stoch_k", 12)),
        "stoch_d": int(run_config.get("stoch_d", 12)),
        "stoch_slow": int(run_config.get("stoch_slow", 8)),
        "entry_lookback_h": int(run_config.get("entry_lookback_h", run_config.get("lookback_breakout_hours", 0))),
    }
    js = json.dumps(comp, sort_keys=True, separators=(',', ':'))
    tag = hashlib.sha1(js.encode('utf-8')).hexdigest()[:10]
    return comp, tag

def run_and_autosave(session_dir: Path, config_path: Path, df_main_with_mas: pd.DataFrame, base_minutes=BASE_MINUTES):
    """
    Updated:
      - hourly feature averages (uses era_int as hour key)
      - SL/TP optimized per `sl_tp_interval_months` (default 6 months)
      - honors avoid_selection_bias and per_config_single_file flags
      - removes era_day_int from final saved outputs and renames avg_price_ma_gap -> avg_price_ma_gap_a
    """
    with open(config_path, 'r') as f:
        run_config = json.load(f)

    cfg_id = str(run_config.get("config_id", "unknown"))
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_marker = results_dir / f"cfg_{cfg_id}_results_manifest.json"
    # skip if already ran (either parts or single cfg file depending on config)
    per_config_single = bool(run_config.get("per_config_single_file", True))
    single_path = results_dir / f"cfg_{cfg_id}.parquet" if per_config_single else None
    if list(results_dir.glob(f"cfg_{cfg_id}_part*.parquet")) or (per_config_single and single_path.exists()):
        return

    # 1) signals (same as before)
    df_filtered = generate_filtered_signals(df_main_with_mas, run_config, base_minutes=base_minutes)
    if df_filtered is None or df_filtered.empty:
        return

    BTC = run_config["BTC_SETTINGS"]
    sl_min, sl_max = BTC.get("sl_min", 20), BTC.get("sl_max", 20)
    tp_min, tp_max = BTC.get("tp_min", 800), BTC.get("tp_max", 15000)
    point_interval = int(BTC.get("point_interval", 50))
    exit_windows = run_config.get("exit_windows", [1,4,12])
    features_to_agg = run_config.get("feature_columns", [])

    # --- Build hourly feature map (era_int is already hourly in your pipeline) ---
    # This provides avg_{feature} per hour (fast: grouping by hour reduces cardinality drastically)
    hour_feature_map = {}
    if features_to_agg:
        if 'era_int' not in df_main_with_mas.columns:
            raise RuntimeError("df_main_with_mas must contain 'era_int' (hour identifier) for hourly feature aggregation.")
        # groupby era_int (hour) and compute mean for features
        g = df_main_with_mas.groupby('era_int')[features_to_agg].mean()
        hour_feature_map = {int(idx): row.to_dict() for idx, row in g.iterrows()}
        del g
        gc.collect()

    # metadata / small constants
    ma_int = np.int8(run_config.get("ma_int", 0))
    is_rev = np.int8(1 if run_config.get("ma_reversion", False) else 0)
    lb_val = np.uint16(run_config.get("entry_lookback_h", 0))
    min_rr = float(run_config.get("min_rr", 1.0))
    avoid_bias = bool(run_config.get("avoid_selection_bias", True))
    sl_tp_interval_months = int(run_config.get("sl_tp_interval_months", 6))

    part_idx = 0
    rows_written = 0

    # writer when using single-file per config
    writer = None
    try:
        for hours in exit_windows:
            _maybe_trim_all_caches()

            entry = compute_window_lookahead_once(hours, base_minutes=base_minutes, use_cache=run_config.get("lookahead", {}).get("use_cache", True))
            if not entry:
                continue

            lookahead_indices = entry['index_map']
            mask = df_filtered.index.isin(lookahead_indices)
            df_full_window = df_filtered.loc[mask].copy()
            if df_full_window.empty:
                continue

            # align vectorized lookahead arrays
            sorter = np.argsort(lookahead_indices)
            pos = np.searchsorted(lookahead_indices, df_full_window.index.values, sorter=sorter)
            actual_locs = sorter[pos]

            df_full_window['next_min'] = entry['next_min'][actual_locs]
            df_full_window['next_max'] = entry['next_max'][actual_locs]
            df_full_window['next_min_time_ns'] = entry['next_min_time_ns'][actual_locs]
            df_full_window['next_max_time_ns'] = entry['next_max_time_ns'][actual_locs]

            # Build sl/tp period id per-row (integer bucket for N month intervals)
            # Example: months=6 => two buckets per year: bucket_id = year * (12//months) + ((month-1)//months)
            months = max(1, sl_tp_interval_months)
            buckets_per_year = 12 // months
            yr = df_full_window['time'].dt.year.astype(int)
            mo = df_full_window['time'].dt.month.astype(int)
            bucket_id = (yr * buckets_per_year) + ((mo - 1) // months)
            df_full_window['sltp_period'] = bucket_id.values

            # iterate over sltp periods (this reduces optimization frequency to one per N-month bucket)
            for period, df_period_slice in df_full_window.groupby('sltp_period'):
                if df_period_slice.empty:
                    continue

                # EARLY-EXIT PRUNE: config-driven to avoid selection bias
                if avoid_bias:
                    keeps_idx = np.arange(len(df_period_slice))
                else:
                    s = df_period_slice['side'].values
                    closes = df_period_slice['close'].values.astype(np.float32)
                    nmin = df_period_slice['next_min'].values.astype(np.float32)
                    nmax = df_period_slice['next_max'].values.astype(np.float32)
                    adverse = np.where(s == 1, closes - nmin, nmax - closes)
                    keep_mask = adverse <= float(sl_max)
                    keeps_idx = np.nonzero(keep_mask)[0]

                if keeps_idx.size == 0:
                    continue

                df_slice = df_period_slice.iloc[keeps_idx].copy()
                if df_slice.empty:
                    continue

                # Single heavy numba call per period (instead of per-day)
                outdf = optimize_window(df_slice, hours,
                                       sl_min, sl_max,
                                       tp_min, tp_max,
                                       point_interval,
                                       min_rr=min_rr)

                if outdf is None or outdf.empty:
                    del outdf
                    gc.collect()
                    continue

                # attach minimal metadata (no era_day_int)
                # we pick representative era_int (start hour of the slice)
                rep_era_int = int(df_slice['era_int'].min()) if 'era_int' in df_slice.columns else None
                outdf['era_int'] = np.int64(rep_era_int) if rep_era_int is not None else np.int64(df_slice['time'].dt.year.iloc[0] * 1000000)
                outdf['exit_window_h'] = np.uint16(hours)
                outdf['config_id'] = np.int32(cfg_id)
                outdf['ma_int'] = ma_int
                outdf['ma_reversion'] = is_rev
                outdf['entry_lookback_h'] = lb_val
                # optional: include sltp_period for traceability
                outdf['sltp_period'] = int(period)

                # attach aggregated features using hourly map (avg per era_int)
                for col in features_to_agg:
                    if rep_era_int is None:
                        avg_val = np.nan
                    else:
                        avg_val = hour_feature_map.get(rep_era_int, {}).get(col, np.nan)
                    if col == "price_ma_gap":
                        outdf["avg_price_ma_gap_a"] = np.float32(avg_val)
                    else:
                        outdf[f"avg_{col}"] = np.float32(avg_val)

                # minimize columns before writing
                base_cols = ['score','SL','TP','win_pos','total_pos','era_int','exit_window_h',
                             'config_id','ma_int','ma_reversion','entry_lookback_h','sltp_period']
                agg_cols = []
                for c in features_to_agg:
                    agg_cols.append("avg_price_ma_gap_a" if c == "price_ma_gap" else f"avg_{c}")
                cols_needed = [c for c in base_cols + agg_cols if c in outdf.columns]
                df_to_write = outdf.loc[:, cols_needed].copy()

                # write: append to single file or write legacy part file
                if per_config_single:
                    table = pa.Table.from_pandas(df_to_write, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(str(single_path), table.schema, compression=RUN_CONFIG.get("session", {}).get("save_compression","snappy"))
                    writer.write_table(table)
                else:
                    _write_chunk_part(df_to_write, results_dir, cfg_id, part_idx)
                    part_idx += 1

                rows_written += len(df_to_write)

                # cleanup
                del df_to_write
                del outdf
                del df_slice
                gc.collect()

            # per-window cleanup
            _maybe_trim_all_caches()

    finally:
        if writer:
            try:
                writer.close()
            except Exception:
                pass

    manifest = {"cfg_id": cfg_id,
                "parts_written": part_idx if not per_config_single else (1 if rows_written>0 else 0),
                "rows_written": int(rows_written),
                "sl_tp_interval_months": sl_tp_interval_months,
                "finished_at": time.time()}
    output_marker.write_text(json.dumps(manifest), encoding='utf8')

    gc.collect()
    return

# %%
# Section 6. Driver

# --- A. Config Generation ---
cfg_files = generate_configs(RUN_CONFIG, RUN_BASE_DIR)
if cfg_files is None:
    cfg_files = list((RUN_BASE_DIR / "configs").glob("*.json"))
total_cfgs = len(cfg_files)

print("="*50)
print(f"📂 SESSION: {RUN_BASE_DIR.name}")
print(f"⚙️  Generated {total_cfgs} config(s).")
print(f"📊 Main Data Shape: {df_main.shape}")
print("="*50)

# --- B. Optimization Loop ---
if cfg_files:
    pbar = tqdm(
        cfg_files, 
        total=total_cfgs,
        desc="🚀 Optimizing", 
        unit="cfg",
        bar_format='{l_bar}{bar:30}{r_bar}' 
    )
    
    for cfg_path in pbar:

        pbar.set_description(f"🚀 {cfg_path.stem}")
        
        # Runs the strategy, filters signals, and saves result to disk
        run_and_autosave(
            session_dir=RUN_BASE_DIR, 
            config_path=cfg_path, 
            df_main_with_mas=df_main
        )
        
        # --- Periodic RAM Maintenance ---
        # Extract ID from filename (e.g., cfg_0042_... -> 42)
        try:
            cfg_id = int(cfg_path.stem.split('_')[1])
        except (IndexError, ValueError):
            cfg_id = 1 # fallback
            
        if cfg_id % 20 == 0:
            _LOOKAHEAD_CACHE.clear()  # Clear Numba lookahead cache
            _STOCH_CACHE.clear()     # Clear Stochastic cache
            gc.collect()             # Force garbage collection


    # --- C. Final Persistence ---
    print("\n" + "="*50)
    print("📦 Streaming results to Master Parquet (RAM Safe)...")
    
    # Stream individual cfg parquets into one master file without loading all into RAM
    safe_save_all_to_disk(RUN_BASE_DIR, out_filename="master_results.parquet")
    
    print(f"✅ SESSION COMPLETE: {RUN_BASE_DIR.name}")
    print("="*50)
else:
    print("No configs found to run. Check your RUN_CONFIG settings.")

# %%
# 1) Confirm eras in df_main
print("unique eras in df_main:", df_main['era_int'].nunique())

# 2) Run a single config (or run the script) then inspect produced file
# After run_and_autosave finishes, load one cfg output and run:
df_out = pd.read_parquet(RUN_BASE_DIR / "results" / f"cfg_{cfg_id}_results.parquet")
print("rows:", len(df_out))
print("unique eras in results:", df_out['era_int'].nunique())
print("sample eras:", sorted(df_out['era_int'].unique())[:10])


