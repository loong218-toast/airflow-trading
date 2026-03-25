from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
import os
from typing import Optional

app = FastAPI()

cors_origins = [
    x.strip()
    for x in os.getenv("CORS_ORIGINS", "").split(",")
    if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def db_uri() -> str:
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "postgres")
    db = os.getenv("POSTGRES_DB", "airflow")

    if not user or not password:
        raise RuntimeError("POSTGRES_USER / POSTGRES_PASSWORD are required")

    return f"postgresql+psycopg2://{user}:{password}@{host}/{db}"

engine = create_engine(db_uri(), pool_pre_ping=True)

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/volatility-index")
def volatility_index(
    pair: str = Query(default="XXBTZUSD"),
    market_type: str = Query(default="spot"),
    months: int = Query(default=6, ge=1, le=120),
):
    sql = text("""
        WITH stats AS (
            SELECT
                MIN(close) AS min_close,
                MAX(close) AS max_close,
                COUNT(*) AS candle_count,
                MIN(time) AS first_seen,
                MAX(time) AS last_seen
            FROM df_main
            WHERE pair = :pair
              AND market_type = :market_type
              AND time >= NOW() - (:months || ' months')::interval
        )
        SELECT
            min_close,
            max_close,
            candle_count,
            first_seen,
            last_seen,
            CASE
                WHEN min_close IS NULL OR min_close = 0 THEN NULL
                ELSE (max_close - min_close) / min_close
            END AS volatility_index
        FROM stats
    """)

    with engine.connect() as conn:
        row = conn.execute(
            sql,
            {"pair": pair, "market_type": market_type, "months": months},
        ).mappings().first()

    if not row or row["candle_count"] == 0:
        raise HTTPException(status_code=404, detail="No data found for this pair and market_type")

    return {
        "pair": pair,
        "market_type": market_type,
        "months": months,
        "data": dict(row),
    }

@app.get("/api/freshness")
def freshness(
    pair: str = Query(default="XXBTZUSD"),
    market_type: str = Query(default="spot"),
):
    sql = text("""
        SELECT
            MAX(time) AS last_seen,
            COUNT(*) AS candle_count
        FROM df_main
        WHERE pair = :pair
          AND market_type = :market_type
    """)

    with engine.connect() as conn:
        row = conn.execute(
            sql,
            {"pair": pair, "market_type": market_type},
        ).mappings().first()

    if not row or row["last_seen"] is None:
        raise HTTPException(status_code=404, detail="No data found for this pair and market_type")

    return {
        "pair": pair,
        "market_type": market_type,
        "last_seen": row["last_seen"],
        "candle_count": row["candle_count"],
    }