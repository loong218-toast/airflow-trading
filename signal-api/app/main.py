from fastapi import FastAPI
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

@app.get("/api/latest-signal")
def latest_signal():
    query = text("""
        SELECT *
        FROM signal_state_latest
        ORDER BY updated_at DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(query).mappings().first()
    return {"data": dict(row) if row else None}