from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Sequence

import pandas as pd
import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# =============================
# CONFIG
# =============================

EQUITY_DIR = Path(
    os.getenv(
        "EQUITY_DIR",
        r"C:\Users\Owner\airflow-trading\data_lake\Saved_results\CCD_3\equity_search_partitioned\_tmp",
    )
)

PAIR = os.getenv("PAIR", "XXBTZUSD")
MARKET_TYPE = os.getenv("MARKET_TYPE", "spot")

# Human-readable timezone for bucket mapping
LOCAL_TZ = os.getenv("LOCAL_TZ", "Asia/Kuala_Lumpur")

# Optional: if you want to map times from df_main too
USE_DF_MAIN_TIME_MAP = os.getenv("USE_DF_MAIN_TIME_MAP", "0") == "1"


# =============================
# DB HELPERS
# =============================

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


def get_engine_from_env(application_name: str = "time_bucket_scan") -> Engine:
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


# =============================
# LOAD EQUITY PARQUETS
# =============================

def load_equity_parquet_folder(directory: Path) -> pl.DataFrame:
    if not directory.exists():
        raise FileNotFoundError(f"Equity directory not found: {directory}")

    files = sorted(directory.glob("*.parquet"))
    if not files:
        return pl.DataFrame()

    frames: List[pl.DataFrame] = []
    for fp in files:
        df = pl.read_parquet(fp)
        if df.height > 0:
            frames.append(df)

    if not frames:
        return pl.DataFrame()

    df = pl.concat(frames, how="vertical_relaxed")

    needed = ["regime_id", "era_int", "side", "SL", "TP", "time_ns", "entry_idx", "exit_idx", "pnl_pct", "equity"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column in equity data: {col}")

    return (
        df.with_columns(
            pl.col("regime_id").cast(pl.Int32),
            pl.col("era_int").cast(pl.Int64),
            pl.col("side").cast(pl.Int8),
            pl.col("SL").cast(pl.Float32),
            pl.col("TP").cast(pl.Float32),
            pl.col("time_ns").cast(pl.Int64),
            pl.col("entry_idx").cast(pl.Int64),
            pl.col("exit_idx").cast(pl.Int64),
            pl.col("pnl_pct").cast(pl.Float32),
            pl.col("equity").cast(pl.Float32),
        )
        .sort("time_ns")
    )


# =============================
# OPTIONAL df_main TIME MAP
# =============================

def load_df_main_time_map(
    engine: Engine,
    pair: str,
    market_type: str,
    min_ns: Optional[int] = None,
    max_ns: Optional[int] = None,
) -> pl.DataFrame:
    where_extra = ""
    params: Dict[str, Any] = {"pair": pair, "market_type": market_type}

    if min_ns is not None:
        where_extra += " AND time_ns >= :min_ns"
        params["min_ns"] = int(min_ns)

    if max_ns is not None:
        where_extra += " AND time_ns <= :max_ns"
        params["max_ns"] = int(max_ns)

    query = text(f"""
        SELECT
            time_ns,
            time
        FROM df_main
        WHERE pair = :pair
          AND market_type = :market_type
          {where_extra}
        ORDER BY time_ns ASC
    """)

    with engine.connect() as conn:
        df_pd = pd.read_sql_query(query, conn, params=params)

    if df_pd.empty:
        return pl.DataFrame()

    return (
        pl.from_pandas(df_pd)
        .with_columns(
            pl.col("time_ns").cast(pl.Int64),
            pl.col("time").cast(pl.Datetime(time_zone="UTC")),
        )
        .sort("time_ns")
    )


# =============================
# TIME CONVERSION + 4H BUCKETS
# =============================

def add_time_columns(
    df: pl.DataFrame,
    local_tz: str = "Asia/Kuala_Lumpur",
) -> pl.DataFrame:
    if df.is_empty():
        return df

    return (
        df.with_columns(
            pl.from_epoch("time_ns", time_unit="ns")
            .dt.replace_time_zone("UTC")
            .alias("time_utc")
        )
        .with_columns(
            pl.col("time_utc").dt.convert_time_zone(local_tz).alias("time_local"),
            pl.col("time_utc").dt.convert_time_zone(local_tz).dt.date().alias("date_local"),
            pl.col("time_utc").dt.convert_time_zone(local_tz).dt.hour().alias("hour_local"),
        )
        .with_columns(
            ((pl.col("hour_local") // 4) * 4).cast(pl.Int8).alias("bucket_4h_start"),
        )
        .with_columns(
            (
                pl.col("bucket_4h_start").cast(pl.Utf8)
                + ":00-"
                + (pl.col("bucket_4h_start") + 4).cast(pl.Utf8)
                + ":00"
            ).alias("bucket_4h_label")
        )
    )


# =============================
# METRICS
# =============================

def summarize_time_buckets(df: pl.DataFrame, master_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    if df.is_empty():
        return df

    join_keys = ["regime_id", "era_int", "side", "SL", "TP"]

    joined = df
    if master_df is not None and not master_df.is_empty():
        master_cols = [
            c for c in join_keys + ["balance", "total_pos", "win_pos", "max_drawdown"]
            if c in master_df.columns
        ]
        md = (
            master_df
            .select(master_cols)
            .unique(subset=join_keys, keep="last")
        )
        joined = df.join(md, on=join_keys, how="left")

    # STEP 1: PER ERA
    per_era_bucket = (
        joined
        .group_by(["era_int", "bucket_4h_start", "bucket_4h_label"])
        .agg([
            pl.len().alias("trades"),
            pl.col("pnl_pct").mean().alias("avg_pnl_pct"),
            pl.col("pnl_pct").sum().alias("sum_pnl_pct"),
            # Use the actual Master Data columns if they exist, otherwise fallback to pnl_pct logic
            (pl.col("win_pos").first() / (pl.col("total_pos").first() + 1e-9)).alias("trade_positive_rate"),
            pl.col("balance").median().alias("era_balance"),
            pl.col("total_pos").median().alias("era_total_pos"),
            pl.col("max_drawdown").median().alias("era_drawdown"),
        ])
        .with_columns([
            # --- EXPONENTIAL ALPHA FORMULA ---
            # Using the official era_total_pos from Master Data
            (
                (
                    (((pl.col("era_balance") - 100) / 100) / (pl.col("era_total_pos") + 1e-9))
                ).exp() 
                / (pl.col("era_drawdown") + 1e-9)
            ).alias("alpha_per_trade"),
            
            (pl.col("sum_pnl_pct") > 0).cast(pl.Float64).alias("era_positive_pnl"),
        ])
    )

    total_eras = max(1, per_era_bucket.select(pl.col("era_int").n_unique()).item())

    # STEP 2: GLOBAL SUMMARY
    out = (
        per_era_bucket
        .group_by(["bucket_4h_start", "bucket_4h_label"])
        .agg([
            pl.col("era_int").n_unique().alias("eras_covered"),
            pl.col("trades").sum().alias("trades_total"),

            pl.col("sum_pnl_pct").median().alias("median_era_pnl_pct"),
            pl.col("sum_pnl_pct").mean().alias("mean_era_pnl_pct"),
            pl.col("sum_pnl_pct").quantile(0.25).alias("q25_era_pnl_pct"),
            pl.col("sum_pnl_pct").quantile(0.75).alias("q75_era_pnl_pct"),

            pl.col("trade_positive_rate").median().alias("median_trade_positive_rate"),
            pl.col("trade_positive_rate").mean().alias("mean_trade_positive_rate"),

            pl.col("era_positive_pnl").mean().alias("prob_positive_era_pnl"),
            (pl.col("sum_pnl_pct") > 0).mean().alias("prob_positive_bucket_era"),

            pl.col("era_balance").median().alias("median_era_balance"),
            pl.col("era_drawdown").median().alias("median_era_drawdown"),
            pl.col("alpha_per_trade").median().alias("median_alpha_per_trade"),
        ])
        .with_columns([
            (pl.col("eras_covered") / float(total_eras)).alias("era_coverage"),
            (pl.col("median_era_pnl_pct") / (pl.col("median_era_drawdown") + 1e-9)).alias("robust_score"),
        ])
        .sort(["robust_score", "prob_positive_era_pnl", "median_era_pnl_pct"], descending=True)
    )

    return out


def summarize_by_strategy_and_bucket(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    out = (
        df.group_by(["regime_id", "side", "SL", "TP", "bucket_4h_start", "bucket_4h_label"])
        .agg(
            pl.len().alias("trades"),
            pl.col("pnl_pct").mean().alias("avg_pnl_pct"),
            pl.col("pnl_pct").median().alias("median_pnl_pct"),
            pl.col("pnl_pct").sum().alias("sum_pnl_pct"),
            (pl.col("pnl_pct") > 0).mean().alias("win_rate"),
            pl.col("time_local").min().alias("first_time_local"),
            pl.col("time_local").max().alias("last_time_local"),
        )
        .sort(["regime_id", "side", "SL", "TP", "sum_pnl_pct"], descending=[False, False, False, False, True])
    )
    return out


# =============================
# MAIN
# =============================

def main():
    # 1. Define Paths relative to EQUITY_DIR
    # This points to .../CCD_3/master_metrics.parquet
    master_path = EQUITY_DIR.parent.parent / "master_metrics.parquet"
    
    # 2. Load Data
    print(f"Loading Master Metrics from: {master_path}")
    # --- THIS WAS THE MISSING LINE ---
    master_df = pl.read_parquet(master_path) 
    
    print("Loading Equity Parquets...")
    df = load_equity_parquet_folder(EQUITY_DIR)
    
    if df.is_empty():
        print("No equity data found.")
        return

    # 3. Process Time
    df = add_time_columns(df, local_tz=LOCAL_TZ)

    # 4. Generate Summaries
    # Now master_df exists and can be passed here:
    bucket_summary = summarize_time_buckets(df, master_df)
    strategy_bucket_summary = summarize_by_strategy_and_bucket(df)

    # 5. Output & Save
    print(f"\nAnalysis for {PAIR} ({MARKET_TYPE})")
    print("-" * 50)
    print(bucket_summary.select([
        "bucket_4h_label",
        "trades_total",
        "prob_positive_era_pnl",
        "median_era_pnl_pct",
        "median_alpha_per_trade",
        "median_era_drawdown",
    ]).head(10))

    # Use EQUITY_DIR's grandparent to find the base path for output
    out_dir = EQUITY_DIR.parent.parent / "_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket_summary.write_csv(out_dir / "bucket_summary.csv")
    print(f"\nFull report saved to: {out_dir}")

if __name__ == "__main__":
    main()