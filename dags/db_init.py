# dags/db_init.py
from airflow import DAG
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from pendulum import datetime

with DAG(
    dag_id="db_setup_kraken_tables",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["setup", "infrastructure"]
) as dag:

    initialize_database = SQLExecuteQueryOperator(
        task_id="initialize_database",
        conn_id="my_postgres_conn",
        sql="""
        -- 1. Safely rename candle_raw to ohlc_spot_raw if it exists
        DO $$ 
        BEGIN 
            IF EXISTS (SELECT FROM pg_tables WHERE tablename = 'candle_raw') AND 
               NOT EXISTS (SELECT FROM pg_tables WHERE tablename = 'ohlc_spot_raw') THEN
                ALTER TABLE candle_raw RENAME TO ohlc_spot_raw;
                ALTER INDEX IF EXISTS ix_candle_pair_interval_time_ns RENAME TO ix_ohlc_spot_pair_time_ns;
            END IF;
        END $$;

        -- 2. Ensure ohlc_spot_raw exists
        CREATE TABLE IF NOT EXISTS ohlc_spot_raw (
            pair TEXT NOT NULL,
            interval_minutes INT NOT NULL,
            time TIMESTAMP WITH TIME ZONE NOT NULL,
            time_ns BIGINT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (pair, interval_minutes, time)
        );

        -- 3. Create ohlc_future_raw (with funding_rate support)
        CREATE TABLE IF NOT EXISTS ohlc_future_raw (
            pair TEXT NOT NULL,
            interval_minutes INT NOT NULL,
            time TIMESTAMP WITH TIME ZONE NOT NULL,
            time_ns BIGINT NOT NULL,
            tick_type TEXT DEFAULT 'trade', -- trade, mark, or spot
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            funding_rate NUMERIC(12, 10),
            PRIMARY KEY (pair, interval_minutes, time)
        );

        -- 4. Indexes for fast time-series queries
        CREATE INDEX IF NOT EXISTS ix_ohlc_spot_pair_time_ns ON ohlc_spot_raw (pair, time_ns);
        CREATE INDEX IF NOT EXISTS ix_ohlc_future_pair_time_ns ON ohlc_future_raw (pair, time_ns);

        -- 5. df_main (processed features / indicators)
        -- Added market_type to differentiate spot/future features
        CREATE TABLE IF NOT EXISTS df_main (
        pair TEXT NOT NULL,
        market_type TEXT NOT NULL, 
        time TIMESTAMP WITH TIME ZONE NOT NULL,
        time_ns BIGINT NOT NULL,
        "open" REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        funding_rate NUMERIC(12, 10), 
        spread REAL,         
        change_24h REAL,     
        era_int BIGINT,
        is_outlier BOOLEAN DEFAULT FALSE,
        PRIMARY KEY (pair, market_type, time)
    );

    CREATE INDEX IF NOT EXISTS ix_dfmain_pair_time_ns ON df_main (pair, time_ns);
    CREATE INDEX IF NOT EXISTS ix_dfmain_era ON df_main (era_int);

        -- 6. Moving Averages Table
        CREATE TABLE IF NOT EXISTS indicator_ma (
        pair TEXT NOT NULL,
        market_type TEXT NOT NULL, 
        time TIMESTAMP WITH TIME ZONE NOT NULL,
        time_ns BIGINT NOT NULL,
        ma_a DOUBLE PRECISION, -- Generic Slot 1
        ma_b DOUBLE PRECISION, -- Generic Slot 2
        ma_c DOUBLE PRECISION, -- Generic Slot 3
        ma_d DOUBLE PRECISION, -- Generic Slot 4
        price_ma_gap_h REAL,
        ma_gap_a REAL,
        ma_gap_b REAL,
        ma_gap_c REAL,
        PRIMARY KEY (pair, market_type, time)
    );

        CREATE INDEX IF NOT EXISTS ix_ma_pair_time_ns ON indicator_ma (pair, time_ns);

        -- 7. Stochastic Oscillator Table
        CREATE TABLE IF NOT EXISTS indicator_stochastic (
            pair TEXT NOT NULL,
            market_type TEXT NOT NULL,
            time TIMESTAMP WITH TIME ZONE NOT NULL,
            time_ns BIGINT NOT NULL,
            pct_k REAL,            -- Fast line
            pct_d REAL,            -- Slow line (SMA of K)
            pct_d_slow REAL,       -- Further smoothed line
            PRIMARY KEY (pair, market_type, time)
        );

        CREATE INDEX IF NOT EXISTS ix_stoch_pair_time_ns ON indicator_stochastic (pair, time_ns);

        -- 8. metadata table
        CREATE TABLE IF NOT EXISTS transform_metadata (
            pair TEXT NOT NULL,
            market_type TEXT NOT NULL,
            last_time_ns BIGINT,
            last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            PRIMARY KEY (pair, market_type)
        );

        CREATE TABLE IF NOT EXISTS funding_history_raw (
        pair TEXT NOT NULL,
        time TIMESTAMP WITH TIME ZONE NOT NULL,
        time_ns BIGINT NOT NULL,
        funding_rate_abs NUMERIC(12, 10),
        funding_rate_rel NUMERIC(12, 10),
        PRIMARY KEY (pair, time)
    );

    CREATE INDEX IF NOT EXISTS ix_funding_hist_pair_time_ns ON funding_history_raw (pair, time_ns);

        """
    )