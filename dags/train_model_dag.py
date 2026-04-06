from __future__ import annotations
import logging
import pendulum
from airflow.sdk import dag, task, get_current_context
# Import the logic, but we will wrap them in @task here
from research.train_model import *

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "retries": 0,
}

@dag(
    dag_id="train_lgbm_regime_model",
    start_date=pendulum.datetime(2023, 9, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["lightgbm", "polars", "regime", "training"],
)
def train_lgbm_regime_model():

    @task
    def enrich_task(session_id: str, split: str, trigger: any):
        # The 'trigger' argument is just to force Airflow to wait for the merge
        return enrich_split_with_master(session_id, split)

    # --- NATIVE TASK DEFINITIONS ---
    # Defining them here allows Airflow 3.0 to handle partial/expand correctly
    @task
    def curate_task(session_id, era_spec, sample_fraction):
        return curate_one_era(era_spec, session_id, sample_fraction)

    @task
    def merge_task(session_id, split):
        return merge_split_files(session_id, split)

    @task
    def train_task(train_path, valid_path, session_id):
        return train_model(train_path, valid_path, session_id)

    @task
    def score_task(model_path, test_path, session_id):
        return predict_and_score(model_path, test_path, session_id)

    @task
    def build_run_id() -> str:
        ctx = get_current_context()
        return ctx["dag_run"].run_id

    @task
    def prepare_workspace(session_id: str) -> str:
        cleanup_run_dir(session_id)
        return session_id

    @task
    def get_era_specs() -> list[dict]:
        return discover_and_assign_era_specs()

    # --- EXECUTION FLOW ---
    session_id_val = build_run_id()
    session_id_val = prepare_workspace(session_id_val)
    era_specs = get_era_specs()

    # This will now work because curate_task is a native @task
    curation_tasks = curate_task.partial(
        session_id=session_id_val,
        sample_fraction=SPLIT_SAMPLE_FRACTION,
    ).expand(
        era_spec=era_specs
    )

    train_merged = merge_task(session_id=session_id_val, split="train")
    valid_merged = merge_task(session_id=session_id_val, split="valid")
    test_merged  = merge_task(session_id=session_id_val, split="test")

    train_enriched = enrich_task(session_id=session_id_val, split="train", trigger=train_merged)
    valid_enriched = enrich_task(session_id=session_id_val, split="valid", trigger=valid_merged)
    test_enriched  = enrich_task(session_id=session_id_val, split="test", trigger=test_merged)

    model_path = train_task(
        train_path=train_enriched,
        valid_path=valid_enriched,
        session_id=session_id_val,
    )

    score_task(
        model_path=model_path,
        test_path=test_enriched,
        session_id=session_id_val,
    )

    curation_tasks >> [train_merged, valid_merged, test_merged]

train_lgbm_regime_model()