# exploration_config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

def detect_project_root() -> Path:
    here = Path(__file__).resolve()

    if Path("/opt/airflow/airflow-trading").exists():
        return Path("/opt/airflow/airflow-trading")
    if Path("/opt/airflow").exists():
        return Path("/opt/airflow")

    return next(
        (p for p in here.parents if (p / "dags").exists() or (p / "data_lake").exists()),
        here.parents[1],
    )

PROJECT_ROOT = detect_project_root()
DAGS_ROOT = PROJECT_ROOT / "dags" if (PROJECT_ROOT / "dags").exists() else Path("/opt/airflow/dags")
DATA_LAKE_ROOT = PROJECT_ROOT / "data_lake"
CACHE_DIR = DATA_LAKE_ROOT / "cache"
RESULTS_DIR = DATA_LAKE_ROOT / "Saved_results" / "expectancy_checks"
TIMEZONE = "Asia/Kuala_Lumpur"


@dataclass(frozen=True)
class ExplorationConfig:
    instrument: str = "USDCHF"
    pair: str = "USDCHF"
    mt5_symbol: str = "USDCHF"

    #instrument: str = "UK100"
    #pair: str = "UK100"
    #mt5_symbol: str = "UK100"

    target_pct: float = 0.09
    sl_pct: float = 0.25
    horizon_hours: int = 24
    spread_pct: float = 0.005
    risk_pct: float = 0.005

    use_trade_timer: tuple[bool, ...] = (False, True)
    trade_timer_minutes: tuple[int, ...] = (150, 300,600)

    simulation_mode: str = "overlapping_random"  # overlapping | overlapping_random | sequential_random
    side_mode: str = "mixed"  # mixed | buy | sell

    use_ma_filter: bool = False
    ma_type: str = "ema"
    ma_period_bars: int = 32

    randomize_entry_price: bool = True
    entry_nudge_max_fraction: float = 0.12
    entry_nudge_clip_to_candle: bool = True

    use_entry_time_window: bool = False
    entry_window_start_hour_myt: int = 15
    entry_window_end_hour_myt: int = 23

    entries_per_hour: int = 4

    use_daily_side_bias: bool = False
    use_utc_for_bias: bool = True

    bootstrap_rounds: int = 2000
    bootstrap_block_size: int = 8
    bootstrap_seed: int = 14121212

    save_trade_universe_csv: bool = False
    save_bootstrap_samples_csv: bool = False

    chart_lookback_days: int = 180
    chart_max_candles: int = 6000
    chart_marker_size: int = 3

    # Trailing exit sweep.
    # Activation is expressed as a fraction of the SL distance:
    #   0.6 == activate after price has moved % of the way to SL.
    use_trailing_tp: bool = True
    trailing_tp_activation_r: float = 0.6
    trailing_tp_distance_r: tuple[float, ...] = (0.05,0.1, 0.2)  # trailing distance in R units
    trailing_tp_interval: tuple[int, ...] = (24)  # update every N candles

    timezone: str = TIMEZONE

    @property
    def cache_file(self) -> Path:
        return CACHE_DIR / f"{self.mt5_symbol}_m5_cache.csv"

    @property
    def output_dir(self) -> Path:
        return RESULTS_DIR / self.instrument / "bootstrap"

    @property
    def chart_output_dir(self) -> Path:
        return RESULTS_DIR / self.instrument / "charts"

    @property
    def prefix(self) -> str:
        return (
            f"{self.instrument}_"
            f"tp_{str(self.target_pct).replace('.', 'p')}_"
            f"sl_{str(self.sl_pct).replace('.', 'p')}_"
            f"h_{int(self.horizon_hours)}_"
            f"{self.simulation_mode}_"
            f"{'ma' if self.use_ma_filter else 'no_ma'}"
        )

CONFIG = ExplorationConfig()
