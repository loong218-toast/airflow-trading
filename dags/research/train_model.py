from __future__ import annotations

import gc
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =============================================================================
# Paths
# =============================================================================

ROOT = Path(os.environ.get("AIRFLOW_TRADING_ROOT", "/opt/airflow/airflow-trading"))
SESSION_DIR = Path(
    os.environ.get(
        "MODEL_SESSION_DIR",
        str(ROOT / "data_lake" / "Opt_Session_20260324_114114_01"),
    )
)

CURATED_SRC_DIR = SESSION_DIR / "equity_partitioned"
RUNS_DIR = SESSION_DIR / "model_runs"

EXCLUDE_ERAS = {20250901}

# =============================================================================
# Tunables
# =============================================================================

SPLIT_SAMPLE_FRACTION = float(os.environ.get("SPLIT_SAMPLE_FRACTION", "0.25"))
HASH_SCALE = 100_000
HASH_SEED = 42

TARGET_COL = "pnl_pct"

EQUITY_KEEP_COLS = [
    "regime_id",
    "era_int",
    "side",
    "time_ns",
    "entry_idx",
    "exit_idx",
    "SL",
    "TP",
    "pnl_pct",
    "equity",
    "ma_p_gap_a_entry",
    "ma_p_gap_b_entry",
    "ma_p_gap_a_exit",
    "ma_p_gap_b_exit",
    "rng_24h_entry",
    "rng_72h_entry",
    "rng_1w_entry",
    "rng_1m_entry",
]

MASTER_KEEP_COLS = [
    "regime_id",
    "era_int",
    "side",
    "ma_int",
    "ma_reversion",
    "entry_lookback_units",
    "exit_window_h",
    "use_stochastic",
    "stoch_key",
    "use_bbw",
    "bbw_periods",
    "bbw_std",
    "bbw_thresholds",
    "SL",
    "TP",
    "total_pos",
    "win_pos",
    "balance",
    "max_drawdown",
]

MODEL_FEATURES = [
    "side",
    "SL",
    "TP",
    "rr_ratio",
    "ma_int",
    "ma_reversion",
    "entry_lookback_units",
    "exit_window_h",
    "use_stochastic",
    "stoch_k",
    "stoch_d",
    "stoch_s",
    "stoch_range",
    "stoch_center",
    "use_bbw",
    "bbw_periods",
    "bbw_std",
    "bbw_thresholds",
    "ma_p_gap_a_entry",
    "ma_p_gap_b_entry",
    "rng_24h_entry",
    "rng_72h_entry",
    "rng_1w_entry",
    "rng_1m_entry",
    "has_rng_24h_entry",
    "has_rng_72h_entry",
    "has_rng_1w_entry",
    "has_rng_1m_entry",
]

CURATED_KEEP_COLS = [
    "regime_id",
    "era_int",
    "side",
    "time_ns",
    "entry_idx",
    "exit_idx",
    "SL",
    "TP",
    "pnl_pct",
    "ma_int",
    "ma_reversion",
    "entry_lookback_units",
    "exit_window_h",
    "use_stochastic",
    "stoch_k",
    "stoch_d",
    "stoch_s",
    "stoch_l",
    "stoch_u",
    "use_bbw",
    "bbw_periods",
    "bbw_std",
    "bbw_thresholds",
    "ma_p_gap_a_entry",
    "ma_p_gap_b_entry",
    "rng_24h_entry",
    "rng_72h_entry",
    "rng_1w_entry",
    "rng_1m_entry",
]

LGB_PARAMS = {
    "objective": "regression_l1",
    "metric": ["l1", "rmse"],
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 3.0,
    "max_depth": -1,
    "verbosity": -1,
    "seed": HASH_SEED,
    "feature_pre_filter": False,
}

NUM_BOOST_ROUND = 8000
EARLY_STOPPING_ROUNDS = 200
TRAIN_FRAC = 0.70
VALID_FRAC = 0.15


# =============================================================================
# Paths / run dirs
# =============================================================================

def sanitize_session_id(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(session_id))


def run_dir(session_id: str) -> Path:
    return RUNS_DIR / sanitize_session_id(session_id)


def split_dir(session_id: str, split: str) -> Path:
    return run_dir(session_id) / "splits" / split


def merged_dir(session_id: str) -> Path:
    return run_dir(session_id) / "merged"


def model_dir(session_id: str) -> Path:
    return run_dir(session_id) / "model"


def pred_dir(session_id: str) -> Path:
    return run_dir(session_id) / "predictions"


# =============================================================================
# Small helpers
# =============================================================================

def scale_key_expr(col: str, scale: int = 100_000) -> pl.Expr:
    """Creates a stable integer key for joining floats."""
    return (pl.col(col) * scale).round(0).cast(pl.Int64).alias(f"{col}_key")

def compact_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for c in df.columns:
        if c == "regime_id":
            exprs.append(pl.col(c).cast(pl.Int32))
        elif c in {"era_int", "time_ns", "entry_idx", "exit_idx"}:
            exprs.append(pl.col(c).cast(pl.Int64))
        elif c == "side":
            exprs.append(pl.col(c).cast(pl.Int8))
        elif c in {
            "ma_reversion",
            "use_stochastic",
            "use_bbw",
            "has_rng_24h_entry",
            "has_rng_72h_entry",
            "has_rng_1w_entry",
            "has_rng_1m_entry",
        }:
            exprs.append(pl.col(c).cast(pl.Int8))
        elif c in {
            "ma_int",
            "entry_lookback_units",
            "exit_window_h",
            "stoch_k",
            "stoch_d",
            "stoch_s",
            "bbw_periods",
            "bbw_thresholds",
        }:
            exprs.append(pl.col(c).cast(pl.Int16))
        elif c in {
            "SL",
            "TP",
            "pnl_pct",
            "rr_ratio",
            "bbw_std",
            "ma_p_gap_a_entry",
            "ma_p_gap_b_entry",
            "rng_24h_entry",
            "rng_72h_entry",
            "rng_1w_entry",
            "rng_1m_entry",
            "stoch_range",
            "stoch_center",
        }:
            exprs.append(pl.col(c).cast(pl.Float32))
        else:
            exprs.append(pl.col(c))
    return df.with_columns(exprs)


def sample_hash_expr(cols: List[str], fraction: float, seed: int = HASH_SEED) -> pl.Expr:
    cutoff = max(1, int(round(HASH_SCALE * float(fraction))))
    key = pl.struct([pl.col(c) for c in cols]).hash(seed=seed).cast(pl.UInt64)
    return (key % HASH_SCALE) < cutoff


def parse_era_int_from_equity_path(path: Path) -> int:
    """
    Expect equity path like:
    .../equity_partitioned/era_int=20230901/equity_era_int=20230901.parquet
    """
    m = re.search(r"equity_era_int=(\d{8})\.parquet$", path.name)
    if m:
        return int(m.group(1))

    m = re.search(r"era_int=(\d{8})", path.parent.name)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not parse era_int from equity path: {path}")


def source_era_files() -> List[Dict]:
    files = sorted(CURATED_SRC_DIR.rglob("*.parquet"))
    out = []
    for p in files:
        try:
            era_int = parse_era_int_from_equity_path(p)
        except Exception:
            continue
        if era_int in EXCLUDE_ERAS:
            continue
        out.append({"era_int": era_int, "path": str(p)})
    out.sort(key=lambda x: x["era_int"])
    return out


def assign_splits(era_specs: List[Dict]) -> List[Dict]:
    n = len(era_specs)
    if n < 3:
        raise RuntimeError(f"Found only {n} eras. Need at least 3.")

    # Ensure at least 1 file for valid and 1 for test
    test_count = max(1, int(n * (1 - TRAIN_FRAC - VALID_FRAC)))
    valid_count = max(1, int(n * VALID_FRAC))
    train_count = n - test_count - valid_count

    out = []
    for i, spec in enumerate(era_specs):
        if i < train_count:
            split = "train"
        elif i < (train_count + valid_count):
            split = "valid"
        else:
            split = "test"
        out.append({**spec, "split": split})
    return out


def ensure_dirs(session_id: str) -> None:
    for p in [
        split_dir(session_id, "train"),
        split_dir(session_id, "valid"),
        split_dir(session_id, "test"),
        merged_dir(session_id),
        model_dir(session_id),
        pred_dir(session_id),
    ]:
        p.mkdir(parents=True, exist_ok=True)


def cleanup_run_dir(session_id: str) -> None:
    shutil.rmtree(run_dir(session_id), ignore_errors=True)
    ensure_dirs(session_id)


# =============================================================================
# Lazy transforms
# =============================================================================

def add_feature_flags(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns([
        pl.when(pl.col("SL") > 0)
        .then(pl.col("TP") / pl.col("SL"))
        .otherwise(None)
        .cast(pl.Float32)
        .alias("rr_ratio"),
        pl.col("ma_reversion").fill_null(False).cast(pl.Int8),
        pl.col("use_stochastic").fill_null(False).cast(pl.Int8),
        pl.col("use_bbw").fill_null(False).cast(pl.Int8),
    ])

    ranges = {
        "rng_24h_entry": 288,
        "rng_72h_entry": 864,
        "rng_1w_entry": 2016,
        "rng_1m_entry": 8640,
    }

    for col, bars in ranges.items():
        lf = lf.with_columns([
            (pl.col("entry_idx") >= pl.lit(bars)).cast(pl.Int8).alias(f"has_{col}"),
            pl.when(pl.col("entry_idx") < pl.lit(bars))
            .then(None)
            .otherwise(pl.col(col))
            .cast(pl.Float32)
            .alias(col),
        ])

    lf = lf.with_columns([
        pl.when(pl.col("stoch_u").is_not_null() & pl.col("stoch_l").is_not_null())
        .then(pl.col("stoch_u") - pl.col("stoch_l"))
        .otherwise(None)
        .cast(pl.Float32)
        .alias("stoch_range"),
        pl.when(pl.col("stoch_u").is_not_null() & pl.col("stoch_l").is_not_null())
        .then((pl.col("stoch_u") + pl.col("stoch_l")) / 2.0)
        .otherwise(None)
        .cast(pl.Float32)
        .alias("stoch_center"),
    ])

    return lf


def curated_lazy_for_era(src_path: str, sample_fraction: float) -> pl.LazyFrame:
    lf = pl.scan_parquet(src_path).select(EQUITY_KEEP_COLS)

    lf = lf.filter(~pl.col("era_int").is_in(list(EXCLUDE_ERAS)))
    lf = lf.filter(
        sample_hash_expr(
            ["regime_id", "era_int", "time_ns", "entry_idx", "exit_idx"],
            sample_fraction,
            HASH_SEED,
        )
    )

    lf = lf.with_columns([
        pl.col("regime_id").cast(pl.Int32),
        pl.col("era_int").cast(pl.Int64),
        pl.col("side").cast(pl.Int8),
        pl.col("SL").cast(pl.Float32),
        pl.col("TP").cast(pl.Float32),
        pl.col("pnl_pct").cast(pl.Float32),
        pl.col("time_ns").cast(pl.Int64),
        pl.col("entry_idx").cast(pl.Int64),
        pl.col("exit_idx").cast(pl.Int64),
        pl.col("ma_p_gap_a_entry").cast(pl.Float32),
        pl.col("ma_p_gap_b_entry").cast(pl.Float32),
        pl.col("ma_p_gap_a_exit").cast(pl.Float32),
        pl.col("ma_p_gap_b_exit").cast(pl.Float32),
        pl.col("rng_24h_entry").cast(pl.Float32),
        pl.col("rng_72h_entry").cast(pl.Float32),
        pl.col("rng_1w_entry").cast(pl.Float32),
        pl.col("rng_1m_entry").cast(pl.Float32),
    ])

    return lf


# =============================================================================
# Airflow-callable functions
# =============================================================================

def discover_and_assign_era_specs() -> List[Dict]:
    specs = source_era_files()
    logger.info("Discovered %d era files", len(specs))
    for s in specs:
        logger.info("Era file: era_int=%s path=%s", s["era_int"], s["path"])
    assigned = assign_splits(specs)
    for s in assigned:
        logger.info("Assigned era_int=%s to split=%s", s["era_int"], s["split"])
    return assigned


def curate_one_era(era_spec: Dict, session_id: str, sample_fraction: float = SPLIT_SAMPLE_FRACTION) -> str:
    ensure_dirs(session_id)

    era_int = int(era_spec["era_int"])
    src_path = str(era_spec["path"])
    split = str(era_spec["split"])

    out_path = split_dir(session_id, split) / f"era_int={era_int}.parquet"
    logger.info("Curating era_int=%s split=%s src=%s out=%s", era_int, split, src_path, out_path)

    lf = curated_lazy_for_era(src_path, sample_fraction)

    # Streaming write: no large in-memory collect
    lf.sink_parquet(str(out_path), compression="zstd")
    logger.info("Finished era_int=%s split=%s", era_int, split)

    return str(out_path)


def merge_split_files(session_id: str, split: str) -> str:
    ensure_dirs(session_id)
    pattern = str(split_dir(session_id, split) / "*.parquet")
    out_path = merged_dir(session_id) / f"{split}.parquet"

    logger.info("Merging split=%s from %s -> %s", split, pattern, out_path)

    lf = pl.scan_parquet(pattern).select(EQUITY_KEEP_COLS)
    lf.sink_parquet(str(out_path), compression="zstd")

    logger.info("Merged split=%s done", split)
    return str(out_path)

def enrich_split_with_master(session_id: str, split: str) -> str:
    ensure_dirs(session_id)

    split_path = merged_dir(session_id) / f"{split}.parquet"
    out_path = merged_dir(session_id) / f"{split}_enriched.parquet"

    logger.info("Enriching split=%s from %s -> %s", split, split_path, out_path)

    split_lf = (
        pl.scan_parquet(str(split_path))
        .with_columns([
            scale_key_expr("SL"),
            scale_key_expr("TP"),
        ])
    )

    master_lf = (
        pl.scan_parquet(str(SESSION_DIR / "master_metrics.parquet"))
        .select(MASTER_KEEP_COLS)
        .filter(~pl.col("era_int").is_in(list(EXCLUDE_ERAS)))
        .with_columns([
            # Regex to extract numbers after k, d, s, l, and u
            pl.col("stoch_key").str.extract(r"k(\d+)", 1).cast(pl.Int16).alias("stoch_k"),
            pl.col("stoch_key").str.extract(r"d(\d+)", 1).cast(pl.Int16).alias("stoch_d"),
            pl.col("stoch_key").str.extract(r"s(\d+)", 1).cast(pl.Int16).alias("stoch_s"),
            pl.col("stoch_key").str.extract(r"l(\d+)", 1).cast(pl.Float32).alias("stoch_l"),
            pl.col("stoch_key").str.extract(r"u(\d+)", 1).cast(pl.Float32).alias("stoch_u"),
        ])
        .with_columns([
            pl.col("regime_id").cast(pl.Int32),
            pl.col("era_int").cast(pl.Int64),
            pl.col("side").cast(pl.Int8),
            pl.col("ma_int").cast(pl.Int32),
            scale_key_expr("SL"),
            scale_key_expr("TP"),
        ])
        .drop(["SL", "TP"])
        .unique(subset=["regime_id", "era_int", "side", "SL_key", "TP_key"])
    )

    enriched = (
        split_lf.join(
            master_lf,
            on=["regime_id", "era_int", "side", "SL_key", "TP_key"],
            how="left",
        )
        .drop(["SL_key", "TP_key"])
        .with_columns([
            pl.col("ma_reversion").fill_null(False).cast(pl.Int8),
            pl.col("use_stochastic").fill_null(False).cast(pl.Int8),
            pl.col("use_bbw").fill_null(False).cast(pl.Int8),
        ])
    )

    ranges = {
        "rng_24h_entry": 288,
        "rng_72h_entry": 864,
        "rng_1w_entry": 2016,
        "rng_1m_entry": 8640,
    }
    for col, bars in ranges.items():
        enriched = enriched.with_columns([
            (pl.col("entry_idx") >= pl.lit(bars)).cast(pl.Int8).alias(f"has_{col}"),
            pl.when(pl.col("entry_idx") < pl.lit(bars))
            .then(None)
            .otherwise(pl.col(col))
            .cast(pl.Float32)
            .alias(col),
        ])

    enriched = enriched.with_columns([
        pl.when(pl.col("stoch_u").is_not_null() & pl.col("stoch_l").is_not_null())
        .then(pl.col("stoch_u") - pl.col("stoch_l"))
        .otherwise(None)
        .cast(pl.Float32)
        .alias("stoch_range"),
        pl.when(pl.col("stoch_u").is_not_null() & pl.col("stoch_l").is_not_null())
        .then((pl.col("stoch_u") + pl.col("stoch_l")) / 2.0)
        .otherwise(None)
        .cast(pl.Float32)
        .alias("stoch_center"),
    ])

    enriched = enriched.with_columns([
        pl.when(pl.col("SL") > 0)
        .then(pl.col("TP") / pl.col("SL"))
        .otherwise(None)
        .cast(pl.Float32)
        .alias("rr_ratio"),
    ])

    enriched = enriched.select([
        "regime_id",
        "era_int",
        "side",
        "time_ns",
        "entry_idx",
        "exit_idx",
        "SL",
        "TP",
        "pnl_pct",
        "ma_int",
        "ma_reversion",
        "entry_lookback_units",
        "exit_window_h",
        "use_stochastic",
        "stoch_k",
        "stoch_d",
        "stoch_s",
        "stoch_l",
        "stoch_u",
        "stoch_range",
        "stoch_center",
        "use_bbw",
        "bbw_periods",
        "bbw_std",
        "bbw_thresholds",
        "ma_p_gap_a_entry",
        "ma_p_gap_b_entry",
        "rng_24h_entry",
        "rng_72h_entry",
        "rng_1w_entry",
        "rng_1m_entry",
        "rr_ratio",
        "has_rng_24h_entry",
        "has_rng_72h_entry",
        "has_rng_1w_entry",
        "has_rng_1m_entry",
    ])

    null_count = enriched.select(pl.col("ma_int").is_null().sum()).collect().item()
    total_count = enriched.select(pl.len()).collect().item()
    null_pct = (null_count / total_count) * 100
    logger.info("Enrichment Stats: Join missed %d rows (%.2f%%)", null_count, null_pct)

    enriched.sink_parquet(str(out_path), compression="zstd")
    logger.info("Enriched split=%s done", split)
    return str(out_path)


def to_lgb_arrays(df: pl.DataFrame):
    feat = (
        df.select(MODEL_FEATURES)
        .with_columns(pl.all().cast(pl.Float32, strict=False))
        .fill_null(np.nan)
    )
    X = feat.to_numpy().astype(np.float32, copy=False)

    y = (
        df.select(TARGET_COL)
        .with_columns(pl.col(TARGET_COL).cast(pl.Float32, strict=False))
        .fill_null(np.nan)
        .to_numpy()
        .reshape(-1)
        .astype(np.float32, copy=False)
    )
    return X, y


def load_split_df(path: str) -> pl.DataFrame:
    logger.info("Loading split file %s", path)
    df = pl.read_parquet(path)
    df = compact_dtypes(df)
    logger.info("Loaded rows=%d cols=%d", df.height, len(df.columns))
    return df


def train_model(train_path: str, valid_path: str, session_id: str) -> str:
    import lightgbm as lgb

    ensure_dirs(session_id)
    out_model = model_dir(session_id) / "lgbm_model.txt"

    logger.info("Loading train split")
    train_df = load_split_df(train_path)
    logger.info("Loading valid split")
    valid_df = load_split_df(valid_path)

    X_train, y_train = to_lgb_arrays(train_df)
    X_valid, y_valid = to_lgb_arrays(valid_df)

    logger.info("Building LightGBM datasets")
    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=MODEL_FEATURES,
        free_raw_data=True,
    )
    dvalid = lgb.Dataset(
        X_valid,
        label=y_valid,
        reference=dtrain,
        feature_name=MODEL_FEATURES,
        free_raw_data=True,
    )

    del train_df, valid_df, X_train, y_train, X_valid, y_valid
    gc.collect()

    logger.info("Training LightGBM")
    booster = lgb.train(
        params=LGB_PARAMS,
        train_set=dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    booster.save_model(str(out_model))
    logger.info("Saved model to %s", out_model)

    del dtrain, dvalid, booster
    gc.collect()

    return str(out_model)


def predict_and_score(model_path: str, test_path: str, session_id: str) -> str:
    import lightgbm as lgb

    ensure_dirs(session_id)
    out_pred = pred_dir(session_id) / "test_predictions.parquet"

    logger.info("Loading test split")
    test_df = load_split_df(test_path)

    logger.info("Loading model %s", model_path)
    booster = lgb.Booster(model_file=str(model_path))

    X_test, y_test = to_lgb_arrays(test_df)
    logger.info("Scoring test split")
    pred = booster.predict(X_test, num_iteration=booster.best_iteration)

    out_df = test_df.select([
        "regime_id",
        "era_int",
        "side",
        "SL",
        "TP",
        "time_ns",
        "entry_idx",
        "exit_idx",
        TARGET_COL,
    ]).with_columns([
        pl.Series("pred_pnl_pct", pred.astype(np.float32)),
        pl.Series("pred_error", (pred.astype(np.float32) - y_test).astype(np.float32)),
        pl.Series("pred_hit", (pred > 0).astype(np.int8)),
    ])

    out_df.write_parquet(str(out_pred), compression="zstd")
    logger.info("Saved predictions to %s", out_pred)

    y = out_df.get_column(TARGET_COL).to_numpy().astype(np.float64)
    p = out_df.get_column("pred_pnl_pct").to_numpy().astype(np.float64)
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    hit_rate = float(np.mean(p > 0))

    logger.info("TEST MAE=%.8f RMSE=%.8f R2=%.8f pred_gt_zero=%.4f", mae, rmse, r2, hit_rate)

    regime_report = (
        out_df
        .group_by(["regime_id", "era_int", "side", "SL", "TP"])
        .agg([
            pl.len().alias("n_trades"),
            pl.col(TARGET_COL).mean().alias("actual_mean_pnl"),
            pl.col("pred_pnl_pct").mean().alias("pred_mean_pnl"),
            pl.col(TARGET_COL).sum().alias("actual_sum_pnl"),
            pl.col("pred_pnl_pct").sum().alias("pred_sum_pnl"),
            (pl.col(TARGET_COL) > 0).mean().alias("actual_win_rate"),
        ])
        .sort("pred_mean_pnl", descending=True)
    )

    regime_path = pred_dir(session_id) / "test_regime_report.parquet"
    regime_report.write_parquet(str(regime_path), compression="zstd")
    logger.info("Saved regime report to %s", regime_path)

    del test_df, booster, X_test, y_test, out_df, regime_report
    gc.collect()

    return str(out_pred)