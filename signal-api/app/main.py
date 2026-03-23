from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
import os

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
    return f"postgresql+psycopg2://{user}:{password}@{host}/{db}"

engine = create_engine(db_uri(), pool_pre_ping=True)

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/volatility-index")
def volatility_index(
    pair: str = Query(default="XXBTZUSD"),
    market_type: str = Query(default="spot"),
    months: int = Query(default=6),
):
    sql = text("""
        WITH bounds AS (
            SELECT
                MIN(time) AS start_time,
                MAX(time) AS end_time
            FROM df_main
            WHERE pair = :pair
              AND market_type = :market_type
              AND time >= NOW() - (:months || ' months')::interval
        ),
        stats AS (
            SELECT
                MIN(close) AS min_close,
                MAX(close) AS max_close,
                COUNT(*) AS candle_count
            FROM df_main
            WHERE pair = :pair
              AND market_type = :market_type
              AND time >= NOW() - (:months || ' months')::interval
        )
        SELECT
            stats.min_close,
            stats.max_close,
            stats.candle_count,
            CASE
                WHEN stats.min_close IS NULL OR stats.min_close = 0 THEN NULL
                ELSE (stats.max_close - stats.min_close) / stats.min_close
            END AS volatility_index
        FROM stats
    """)

    with engine.connect() as conn:
        row = conn.execute(
            sql,
            {"pair": pair, "market_type": market_type, "months": months},
        ).mappings().first()

    return {
        "pair": pair,
        "market_type": market_type,
        "months": months,
        "data": dict(row) if row else None,
    }