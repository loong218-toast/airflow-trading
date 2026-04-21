from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import polars as pl

TOP_PCT = 0.1
TRADE_FLOOR = 40
MIN_RETURN = 0.05
RECENCY_DECAY_DAYS = 90.0

PATH_DIR = Path(r"C:\Users\Owner\airflow-trading\data_lake\Opt_Session_20260417_080040_01")
MASTER_PATH = PATH_DIR / "master_metrics.parquet"

ANALYSIS_ROOT_NAME = "analysis"
SIGNAL_ROOT_NAME = "signal"
TRADE_MGMT_ROOT_NAME = "trade_management"

TRADE_MGMT_CORE_COLS: tuple[str, ...] = (
    "exit_window_h",
    "SL",
    "TP",
    "use_trailing_sl",
    "trailing_sl_pct",
    "trailing_sl_interval",
    "trailing_sl_stop_at_pos",
    "use_limit_entry",
    "limit_order_expiry_bars",
    "trade_window_interval",
)

TRADE_MGMT_OPTIONAL_COLS: tuple[str, ...] = (
    "trade_overlap",
    "trade_flip_on_entry",
)

TRADE_MGMT_FEATURE_COLS: tuple[str, ...] = TRADE_MGMT_CORE_COLS + TRADE_MGMT_OPTIONAL_COLS

ELITE_JOINT_WEIGHTS = {
    "era_consistency_score": 0.40,
    "recency_weighted_era_consistency_score": 0.30,
    "elite_dominance_score": 0.20,
    "elite_median_alpha": 0.10,
}

GLOBAL_JOINT_WEIGHTS = {
    "era_consistency_score": 0.40,
    "recency_weighted_era_consistency_score": 0.30,
    "global_dominance_score": 0.20,
    "global_median_alpha": 0.10,
}


def _write_csv(df: Optional[pl.DataFrame], path: Path) -> None:
    if df is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(str(path))


def _safe_slug(value: Any, max_len: int = 120) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "all")[:max_len]


def _analysis_root(session_dir: str | Path) -> Path:
    return Path(session_dir) / ANALYSIS_ROOT_NAME


def _signal_root(session_dir: str | Path) -> Path:
    return _analysis_root(session_dir) / SIGNAL_ROOT_NAME


def _trade_mgmt_root(session_dir: str | Path) -> Path:
    return _analysis_root(session_dir) / TRADE_MGMT_ROOT_NAME


def load_master_df(session_dir: str | Path | None = None) -> pl.DataFrame:
    candidates: list[Path] = []

    if session_dir is not None:
        session_dir = Path(session_dir)
        candidates.extend(
            [
                session_dir / "master_metrics.parquet",
                session_dir / "results" / "master_metrics.parquet",
                session_dir / "results" / "batch_master_metrics.parquet",
            ]
        )

    candidates.append(MASTER_PATH)

    for p in candidates:
        if p.exists():
            return pl.read_parquet(str(p))

    raise FileNotFoundError("Master parquet not found. Looked in session_dir and MASTER_PATH.")


def _scope_tokens(scope_text: Any) -> list[str]:
    text = str(scope_text or "")
    out: list[str] = []
    seen: set[str] = set()
    for token in text.split("|"):
        token = token.strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _scope_meta(scope_text: Any) -> dict[str, Any]:
    tokens = _scope_tokens(scope_text)
    families: list[str] = []
    timeframes: list[str] = []
    seen_fam: set[str] = set()
    seen_tf: set[str] = set()

    for token in tokens:
        parts = token.split("__")
        fam = parts[0].strip().lower() if parts else ""
        if fam == "stoch":
            fam = "stochastic"

        if fam and fam not in seen_fam:
            seen_fam.add(fam)
            families.append(fam)

        if len(parts) >= 2:
            tf = str(parts[1]).strip()
            if tf and tf not in seen_tf:
                seen_tf.add(tf)
                timeframes.append(tf)

    return {
        "scope_size": int(len(tokens)),
        "families": families,
        "timeframes": timeframes,
        "families_text": ",".join(families),
        "timeframes_text": ",".join(timeframes),
    }


def _num(x: Any) -> str:
    if x is None:
        return "na"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return str(int(x)) if x.is_integer() else str(x)
    return str(x)


def _first(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return _first(v[0]) if v else None
    return v


def _signal_obj(x: Any) -> dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _signal_blocks(signal_json: Any) -> dict[str, Any]:
    return _signal_obj(signal_json).get("signals", {}) or {}


def _atomic_signal_tokens(scope_text: Any) -> list[str]:
    text = str(scope_text or "")
    out: list[str] = []
    seen: set[str] = set()
    for token in text.split("|"):
        token = token.strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _scope_contains_token_expr(token: str) -> pl.Expr:
    return pl.col("signal_scope").cast(pl.Utf8, strict=False).str.contains(rf"(^|\|){re.escape(token)}(\||$)")


def _scope_contains_family_expr(family: str) -> pl.Expr:
    fam = str(family or "").strip().lower()
    if fam == "stochastic":
        return pl.col("signal_scope").cast(pl.Utf8, strict=False).str.contains(r"(^|\|)(?:stoch|stochastic)__")
    return pl.col("signal_scope").cast(pl.Utf8, strict=False).str.contains(rf"(^|\|){re.escape(fam)}__")


def _stoch_key(signal_json: Any, tf: str = "15m") -> str:
    signals = _signal_blocks(signal_json)
    st = signals.get("stochastic", {}).get(tf, {}) if isinstance(signals, dict) else {}
    k = _first(st.get("k"))
    d = _first(st.get("d"))
    s = _first(st.get("s"))
    th = st.get("thresholds", [[30, 70]])
    low, high = 30, 70
    if isinstance(th, (list, tuple)) and th:
        first = th[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            low, high = first[0], first[1]
        elif len(th) >= 2 and not any(isinstance(i, (list, tuple, dict)) for i in th[:2]):
            low, high = th[0], th[1]
    tol = _first(st.get("threshold_tolerance", 10))
    return f"k{_num(k)}_d{_num(d)}_s{_num(s)}_l{_num(low)}_u{_num(high)}_tol{_num(tol)}"


def _lookback_key(signal_json: Any, tf: str = "15m") -> str:
    signals = _signal_blocks(signal_json)
    lb = _first(signals.get("lookback", {}).get(tf, {}).get("entry_lookback_units"))
    return f"u{_num(lb)}"


def add_human_readable_signal_keys(df: pl.DataFrame, timeframes: tuple[str, ...] = ("5m", "15m")) -> pl.DataFrame:
    if df is None or df.is_empty() or "signal_json" not in df.columns:
        return df

    out = df
    for tf in timeframes:
        out = out.with_columns(
            pl.col("signal_json").map_elements(lambda x, _tf=tf: _stoch_key(x, _tf), return_dtype=pl.Utf8).alias(f"stoch_key ({tf})"),
            pl.col("signal_json").map_elements(lambda x, _tf=tf: _lookback_key(x, _tf), return_dtype=pl.Utf8).alias(f"lookback_key ({tf})"),
        )
    return out


def build_era_recency_weight_map(eras: list[Any], decay_days: float = RECENCY_DECAY_DAYS) -> dict[int, float]:
    clean = []
    for e in eras:
        try:
            clean.append(int(e))
        except Exception:
            pass

    clean = sorted(set(clean))
    if not clean:
        return {}

    latest = datetime.strptime(str(clean[-1]), "%Y%m%d").date()
    out = {}
    for era in clean:
        era_dt = datetime.strptime(str(era), "%Y%m%d").date()
        age_days = max((latest - era_dt).days, 0)
        out[int(era)] = float(1.0 / (1.0 + math.log1p(age_days / decay_days)))
    return out


def add_perf_cols(df: pl.DataFrame) -> pl.DataFrame:
    ret = ((pl.col("balance") - 100) / 100) / (pl.col("total_pos") + 1e-9)
    return df.with_columns(
        ret.alias("return_per_trade"),
        (ret * ret.exp() / (pl.col("max_drawdown") + 1e-9)).alias("alpha_per_trade"),
    )



def _ensure_master_cols(df: pl.DataFrame) -> pl.DataFrame:
    if df is None or df.is_empty():
        return df

    out = df
    if "signal_scope" not in out.columns:
        out = out.with_columns(pl.lit("").alias("signal_scope"))
    if "signal_layer" not in out.columns:
        out = out.with_columns(pl.lit(0, dtype=pl.Int64).alias("signal_layer"))

    out = out.with_columns(
        pl.col("signal_scope").cast(pl.Utf8, strict=False).fill_null(""),
        pl.col("signal_layer").cast(pl.Int64, strict=False).fill_null(0),
    )
    return out

def _ensure_trade_mgmt_cols(df: pl.DataFrame) -> pl.DataFrame:
    if df is None or df.is_empty():
        return df

    out = df

    # Core columns: always normalize them.
    core_defaults = {
        "exit_window_h": -1,
        "SL": None,
        "TP": None,
        "use_trailing_sl": None,
        "trailing_sl_pct": None,
        "trailing_sl_interval": -1,
        "trailing_sl_stop_at_pos": None,
        "use_limit_entry": None,
        "limit_order_expiry_bars": -1,
        "trade_window_interval": -1,
    }

    for c, default in core_defaults.items():
        if c not in out.columns:
            if default is None:
                out = out.with_columns(pl.lit(None).alias(c))
            else:
                out = out.with_columns(pl.lit(default).alias(c))
        else:
            out = out.with_columns(pl.col(c).alias(c))

    # Optional v2 columns: keep them if present, otherwise create nulls.
    optional_defaults = {
        "trade_overlap": None,
        "trade_flip_on_entry": None,
    }

    for c, default in optional_defaults.items():
        if c not in out.columns:
            out = out.with_columns(pl.lit(None).alias(c))
        else:
            out = out.with_columns(pl.col(c).alias(c))

    return out

def _trade_mgmt_key_cols(df: pl.DataFrame) -> list[str]:
    """
    Build version-aware grouping keys.
    Optional columns only participate if they exist in the source frame
    and have at least one non-null value.
    """
    key_cols = [
        "signal_layer",
        "signal_scope",
        "signal_scope_id",
        "regime_id",
        "side",
        "exit_window_h",
        "SL",
        "TP",
        "use_trailing_sl",
        "trailing_sl_pct",
        "trailing_sl_interval",
        "trailing_sl_stop_at_pos",
        "use_limit_entry",
        "limit_order_expiry_bars",
        "trade_window_interval",
    ]

    for c in TRADE_MGMT_OPTIONAL_COLS:
        if c in df.columns:
            try:
                has_any = bool(df.select(pl.col(c).is_not_null().any()).item())
            except Exception:
                has_any = False
            if has_any:
                key_cols.append(c)

    return key_cols

def _round_df(df: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
    if df is None or df.is_empty():
        return df

    out = df
    for c, dtype in out.schema.items():
        if dtype in (pl.Float32, pl.Float64):
            decimals = 5 if "median_return" in c.lower() else 4
            out = out.with_columns(pl.col(c).cast(pl.Float64, strict=False).round(decimals).alias(c))
    return out


def _dominance_metrics(group_rows: int, elite_rows: int, parent_rows: int) -> tuple[float, float]:
    elite_dominance = float(elite_rows / max(group_rows, 1))
    global_dominance = float(group_rows / max(parent_rows, 1))
    return elite_dominance, global_dominance


def _weighted_joint_score(df: pl.DataFrame, metric_weights: dict[str, float], score_name: str) -> pl.DataFrame:
    if df is None or df.is_empty():
        return df

    out = df
    needed = list(metric_weights.keys())

    for c in needed:
        if c not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(c))

    zcols = []
    for c in needed:
        zc = f"__{c}_z"
        zcols.append(zc)
        out = out.with_columns(((pl.col(c) - pl.col(c).mean()) / (pl.col(c).std() + 1e-9)).alias(zc))

    expr = None
    for c in needed:
        term = pl.lit(float(metric_weights[c])) * pl.col(f"__{c}_z")
        expr = term if expr is None else expr + term

    out = out.with_columns(expr.alias(score_name))
    out = out.drop(zcols)
    return out


def _add_joint_scores(df_summary: pl.DataFrame) -> pl.DataFrame:
    if df_summary is None or df_summary.is_empty():
        return df_summary
    out = _weighted_joint_score(df_summary, ELITE_JOINT_WEIGHTS, "elite_joint_score")
    out = _weighted_joint_score(out, GLOBAL_JOINT_WEIGHTS, "global_joint_score")
    return out


def _era_consistency_from_coverage(
    df_eff: pl.DataFrame,
    eras: list[Any],
    era_weight_map: dict[int, float],
    min_return: float,
) -> tuple[float, float, float, int, int]:
    """
    Old logic:
    "Did it ever work in this era?"

    An era counts as a hit if there is at least one row in that era whose
    session return is >= min_return.

    This is binary per era, not per-trade density.
    """
    if df_eff is None or df_eff.is_empty() or not eras:
        return 0.0, 0.0, 0.0, 0, 0

    per_era_hits: list[float] = []
    weighted_num = 0.0
    weighted_den = 0.0
    hit_eras = 0
    qualifying_rows_total = 0

    session_return = (pl.col("balance") - 100) / 100

    for era in eras:
        era_df = df_eff.filter(pl.col("era_int") == int(era))
        if era_df.is_empty():
            continue

        qualifying_rows = int(era_df.filter(session_return >= float(min_return)).height)
        qualifying_rows_total += qualifying_rows

        era_hit = 1.0 if qualifying_rows > 0 else 0.0
        per_era_hits.append(era_hit)

        if era_hit > 0:
            hit_eras += 1

        w = float(era_weight_map.get(int(era), 1.0))
        weighted_num += era_hit * w
        weighted_den += w

    era_consistency_score = float(sum(per_era_hits) / max(len(per_era_hits), 1))
    recency_weighted_era_consistency_score = float(weighted_num / max(weighted_den, 1e-9))
    hit_rate = float(hit_eras / max(len(per_era_hits), 1))

    return era_consistency_score, recency_weighted_era_consistency_score, hit_rate, hit_eras, qualifying_rows_total


def _compute_group_report(
    df_group: pl.DataFrame,
    parent_total_rows: int,
    top_pct: float = TOP_PCT,
    trade_floor: int = TRADE_FLOOR,
    min_return: float = MIN_RETURN,
) -> tuple[Optional[dict[str, Any]], Optional[pl.DataFrame], Optional[dict[str, Any]]]:
    if df_group is None or df_group.is_empty():
        return None, None, None

    df_group = _ensure_master_cols(df_group)
    df_group = _ensure_trade_mgmt_cols(df_group)
    df_group = df_group.filter(pl.col("total_pos") >= trade_floor)
    if df_group.is_empty():
        return None, None, None

    df_eff = add_perf_cols(df_group)
    eras = sorted(df_eff["era_int"].unique().to_list()) if "era_int" in df_eff.columns else []
    era_weight_map = build_era_recency_weight_map(eras)

    by_era_df = (
        df_eff.group_by("era_int")
        .agg(
            pl.len().alias("rows"),
            pl.col("balance").median().alias("median_balance"),
            pl.col("max_drawdown").mean().alias("avg_dd"),
            (pl.col("win_pos").mean() / (pl.col("total_pos").mean() + 1e-9)).alias("avg_win_rate"),
            pl.col("total_pos").mean().alias("avg_total_pos"),
            pl.col("max_consecutive_losses").mean().alias("mean_max_consecutive_losses"),
            pl.col("max_consecutive_losses").median().alias("median_max_consecutive_losses"),
            pl.col("max_consecutive_losses").max().alias("max_max_consecutive_losses"),
        )
        .with_columns(
            (((pl.col("median_balance") - 100) / 100) / (pl.col("avg_total_pos") + 1e-9)).alias("median_return_per_trade"),
            (
                (((pl.col("median_balance") - 100) / 100) / (pl.col("avg_total_pos") + 1e-9))
                * ((((pl.col("median_balance") - 100) / 100) / (pl.col("avg_total_pos") + 1e-9)).exp())
                / (pl.col("avg_dd") + 1e-9)
            ).alias("alpha_per_trade"),
            pl.col("era_int")
            .map_elements(lambda x: float(era_weight_map.get(int(x), 1.0)), return_dtype=pl.Float64)
            .alias("era_recency_weight"),
        )
        .sort(["era_int"], descending=False)
    )

    elite_rows = []
    for era in eras:
        era_df = df_eff.filter(pl.col("era_int") == era)
        elite_rows.append(era_df.sort("alpha_per_trade", descending=True).head(max(1, int(era_df.height * top_pct))))

    elite_all = pl.concat(elite_rows) if elite_rows else pl.DataFrame()

    if elite_all.is_empty():
        elite_metrics = {
            "elite_rows": 0,
            "elite_mean_max_consecutive_losses": 0.0,
            "elite_median_max_consecutive_losses": 0.0,
            "elite_max_max_consecutive_losses": 0,
            "elite_median_balance": 0.0,
            "elite_avg_max_drawdown": 0.0,
            "elite_median_alpha": 0.0,
            "elite_median_return": 0.0,
        }
    else:
        elite_metrics = elite_all.select(
            pl.len().alias("elite_rows"),
            pl.col("max_consecutive_losses").mean().alias("elite_mean_max_consecutive_losses"),
            pl.col("max_consecutive_losses").median().alias("elite_median_max_consecutive_losses"),
            pl.col("max_consecutive_losses").max().alias("elite_max_max_consecutive_losses"),
            pl.col("balance").median().alias("elite_median_balance"),
            pl.col("max_drawdown").mean().alias("elite_avg_max_drawdown"),
            pl.col("alpha_per_trade").median().alias("elite_median_alpha"),
            pl.col("return_per_trade").median().alias("elite_median_return"),
        ).to_dicts()[0]

    era_consistency_score, recency_weighted_era_consistency_score, era_hit_rate, hit_eras, qualifying_rows_total = _era_consistency_from_coverage(
        df_eff=df_eff,
        eras=eras,
        era_weight_map=era_weight_map,
        min_return=min_return,
    )

    global_metrics = df_eff.select(
        pl.len().alias("global_rows"),
        pl.col("max_consecutive_losses").mean().alias("global_mean_max_consecutive_losses"),
        pl.col("max_consecutive_losses").median().alias("global_median_max_consecutive_losses"),
        pl.col("max_consecutive_losses").max().alias("global_max_max_consecutive_losses"),
        pl.col("balance").median().alias("global_median_balance"),
        pl.col("max_drawdown").mean().alias("global_avg_max_drawdown"),
        pl.col("alpha_per_trade").median().alias("global_median_alpha"),
        pl.col("return_per_trade").median().alias("global_median_return"),
        pl.col("total_pos").mean().alias("global_avg_total_pos"),
    ).to_dicts()[0]

    best_row = df_eff.sort(["alpha_per_trade", "balance", "total_pos"], descending=[True, True, True]).head(1)
    if best_row.is_empty():
        best_metrics = {
            "best_regime_id": None,
            "best_era_int": None,
            "best_side": None,
            "best_balance": None,
            "best_max_drawdown": None,
            "best_total_pos": None,
            "best_SL": None,
            "best_TP": None,
        }
    else:
        r = best_row.to_dicts()[0]
        best_metrics = {
            "best_regime_id": r.get("regime_id"),
            "best_era_int": r.get("era_int"),
            "best_side": r.get("side"),
            "best_balance": r.get("balance"),
            "best_max_drawdown": r.get("max_drawdown"),
            "best_total_pos": r.get("total_pos"),
            "best_SL": r.get("SL"),
            "best_TP": r.get("TP"),
        }

    group_rows = int(df_eff.height)
    elite_dominance_score, global_dominance_score = _dominance_metrics(group_rows, int(elite_metrics["elite_rows"]), parent_total_rows)

    summary_common = {
        "rows": group_rows,
        "era_count": int(len(eras)),
        "filtered_trade_floor": int(trade_floor),
        "min_return": float(min_return),
        **global_metrics,
        **elite_metrics,
        "era_consistency_score": float(era_consistency_score),
        "recency_weighted_era_consistency_score": float(recency_weighted_era_consistency_score),
        "era_hit_rate": float(era_hit_rate),
        "hit_eras": int(hit_eras),
        "qualifying_rows_total": int(qualifying_rows_total),
        "elite_dominance_score": float(elite_dominance_score),
        "global_dominance_score": float(global_dominance_score),
        "alpha_lift": float(elite_metrics["elite_median_alpha"] - global_metrics["global_median_alpha"]),
        "return_lift": float(elite_metrics["elite_median_return"] - global_metrics["global_median_return"]),
        "drawdown_gap": float(global_metrics["global_avg_max_drawdown"] - elite_metrics["elite_avg_max_drawdown"]),
        **best_metrics,
    }

    loss_common = {
        "global_mean_max_consecutive_losses": float(global_metrics["global_mean_max_consecutive_losses"]),
        "global_median_max_consecutive_losses": float(global_metrics["global_median_max_consecutive_losses"]),
        "global_max_max_consecutive_losses": int(global_metrics["global_max_max_consecutive_losses"]),
        "elite_mean_max_consecutive_losses": float(elite_metrics["elite_mean_max_consecutive_losses"]),
        "elite_median_max_consecutive_losses": float(elite_metrics["elite_median_max_consecutive_losses"]),
        "elite_max_max_consecutive_losses": int(elite_metrics["elite_max_max_consecutive_losses"]),
        "streak_gap_mean": float(global_metrics["global_mean_max_consecutive_losses"] - elite_metrics["elite_mean_max_consecutive_losses"]),
        "streak_gap_median": float(global_metrics["global_median_max_consecutive_losses"] - elite_metrics["elite_median_max_consecutive_losses"]),
    }

    return summary_common, by_era_df, loss_common

def _build_scope_regime_combo_df(
    df_scope: pl.DataFrame,
    signal_layer: int,
    scope_text: str,
    top_pct: float = TOP_PCT,
    trade_floor: int = TRADE_FLOOR,
) -> pl.DataFrame:
    """
    Version-safe regime/combo map.

    Backward compatibility:
      - old master files without trade_overlap / trade_flip_on_entry still work
      - those columns are not used in the key unless they actually exist with data

    Forward compatibility:
      - new master files with those columns automatically include them in the key

    Output always keeps the full schema, but missing optional columns remain null.
    """
    schema = {
        "signal_layer": pl.Int64,
        "signal_scope": pl.Utf8,
        "signal_scope_id": pl.Utf8,
        "regime_id": pl.Int32,
        "side": pl.Int8,
        "exit_window_h": pl.Int32,
        "SL": pl.Float32,
        "TP": pl.Float32,
        "use_trailing_sl": pl.Boolean,
        "trailing_sl_pct": pl.Float32,
        "trailing_sl_interval": pl.Int32,
        "trailing_sl_stop_at_pos": pl.Boolean,
        "use_limit_entry": pl.Boolean,
        "limit_order_expiry_bars": pl.Int32,
        "trade_overlap": pl.Boolean,
        "trade_flip_on_entry": pl.Boolean,
        "trade_window_interval": pl.Int32,
        "scope_rows": pl.Int64,
        "era_hits": pl.Int64,
        "elite_rows": pl.Int64,
        "elite_era_hits": pl.Int64,
        "elite_row_rate": pl.Float64,
        "elite_era_rate": pl.Float64,
        "elite_hit": pl.Boolean,
        "elite_median_alpha": pl.Float64,
        "elite_median_return": pl.Float64,
    }

    if df_scope is None or df_scope.is_empty():
        return pl.DataFrame(schema=schema)

    out = _ensure_master_cols(df_scope)
    out = _ensure_trade_mgmt_cols(out)
    out = out.filter(pl.col("total_pos") >= trade_floor)

    if out.is_empty():
        return pl.DataFrame(schema=schema)

    out = out.with_columns(
        pl.lit(int(signal_layer), dtype=pl.Int64).alias("signal_layer"),
        pl.lit(str(scope_text), dtype=pl.Utf8).alias("signal_scope"),
        pl.col("signal_scope_id").cast(pl.Utf8, strict=False).fill_null("").alias("signal_scope_id"),
        pl.col("regime_id").cast(pl.Int32, strict=False).fill_null(-1).alias("regime_id"),
        pl.col("side").cast(pl.Int8, strict=False).fill_null(-1).alias("side"),
    )

    key_cols = _trade_mgmt_key_cols(out)

    # Make sure all key columns exist and are type-stable.
    for c in key_cols:
        if c not in out.columns:
            out = out.with_columns(pl.lit(None).alias(c))

    base = out.select(key_cols + (["era_int"] if "era_int" in out.columns else []))

    scope_agg = base.group_by(key_cols).agg(
        pl.len().alias("scope_rows"),
        pl.col("era_int").n_unique().alias("era_hits") if "era_int" in base.columns else pl.lit(1, dtype=pl.Int64).alias("era_hits"),
    )

    df_eff = add_perf_cols(out)
    eras = sorted(df_eff["era_int"].unique().to_list()) if "era_int" in df_eff.columns else []

    elite_parts: list[pl.DataFrame] = []
    for era in eras:
        era_df = df_eff.filter(pl.col("era_int") == int(era))
        if era_df.is_empty():
            continue
        elite_parts.append(
            era_df.sort("alpha_per_trade", descending=True).head(max(1, int(era_df.height * top_pct)))
        )

    elite_all = pl.concat(elite_parts) if elite_parts else pl.DataFrame()

    if elite_all.is_empty():
        elite_agg = pl.DataFrame(
            schema={
                **{c: schema[c] for c in key_cols},
                "elite_rows": pl.Int64,
                "elite_era_hits": pl.Int64,
                "elite_median_alpha": pl.Float64,
                "elite_median_return": pl.Float64,
            }
        )
    else:
        elite_agg = (
            elite_all
            .select(key_cols + ["era_int", "alpha_per_trade", "return_per_trade"])
            .group_by(key_cols)
            .agg(
                pl.len().alias("elite_rows"),
                pl.col("era_int").n_unique().alias("elite_era_hits"),
                pl.col("alpha_per_trade").median().alias("elite_median_alpha"),
                pl.col("return_per_trade").median().alias("elite_median_return"),
            )
        )

    out = (
        scope_agg
        .join(elite_agg, on=key_cols, how="left")
        .with_columns(
            pl.col("elite_rows").cast(pl.Int64, strict=False).fill_null(0).alias("elite_rows"),
            pl.col("elite_era_hits").cast(pl.Int64, strict=False).fill_null(0).alias("elite_era_hits"),
            pl.col("elite_median_alpha").cast(pl.Float64, strict=False).fill_null(0.0).alias("elite_median_alpha"),
            pl.col("elite_median_return").cast(pl.Float64, strict=False).fill_null(0.0).alias("elite_median_return"),
            (pl.col("elite_rows") > 0).fill_null(False).alias("elite_hit"),
            (pl.col("elite_rows") / (pl.col("scope_rows") + 1e-9)).alias("elite_row_rate"),
            (pl.col("elite_era_hits") / (pl.col("era_hits") + 1e-9)).alias("elite_era_rate"),
        )
    )

    # Fill schema columns that were not part of the key on old masters.
    for c, dtype in schema.items():
        if c not in out.columns:
            out = out.with_columns(pl.lit(None).cast(dtype).alias(c))

    return out.select(list(schema.keys())).sort(
        ["elite_hit", "elite_era_hits", "elite_rows", "scope_rows", "regime_id", "side"],
        descending=[True, True, True, True, False, False],
    )

    return out

def _signal_group_report(
    df_master: pl.DataFrame,
    signal_layer: int,
    scope_text: str,
    top_pct: float = TOP_PCT,
    trade_floor: int = TRADE_FLOOR,
    min_return: float = MIN_RETURN,
) -> tuple[Optional[dict[str, Any]], Optional[pl.DataFrame], Optional[dict[str, Any]], Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Build one signal-scope report plus:
      - per-scope trade-management bundle
      - per-scope regime/combo map
    """
    if df_master is None or df_master.is_empty():
        return None, None, None, None, None, None, None

    df_master = _ensure_master_cols(df_master)
    scope_text = str(scope_text or "").strip()
    if not scope_text:
        return None, None, None, None, None, None, None

    layer_df = df_master.filter(pl.col("signal_layer") == int(signal_layer))
    if layer_df.is_empty():
        return None, None, None, None, None, None, None

    if "|" in scope_text:
        group_df = layer_df.filter(pl.col("signal_scope").cast(pl.Utf8, strict=False) == scope_text)
    else:
        group_df = layer_df.filter(_scope_contains_token_expr(scope_text))

    if group_df.is_empty():
        return None, None, None, None, None, None, None

    summary_common, _, loss_common = _compute_group_report(
        group_df,
        parent_total_rows=int(layer_df.height),
        top_pct=top_pct,
        trade_floor=trade_floor,
        min_return=min_return,
    )
    if summary_common is None or loss_common is None:
        return None, None, None, None, None, None, None

    meta = _scope_meta(scope_text)
    summary_row = {
        "signal_layer": int(signal_layer),
        "signal_scope": scope_text,
        "scope_kind": "combo" if "|" in scope_text else "token",
        "scope_size": int(meta["scope_size"]),
        "families": meta["families_text"],
        "timeframes": meta["timeframes_text"],
        **summary_common,
        "scope_dir": None,
    }

    loss_report = {
        "signal_layer": int(signal_layer),
        "signal_scope": scope_text,
        "scope_kind": "combo" if "|" in scope_text else "token",
        "scope_size": int(meta["scope_size"]),
        "families": meta["families_text"],
        "timeframes": meta["timeframes_text"],
        **loss_common,
        "scope_dir": None,
    }

    scope_trade_summary_df, scope_trade_by_era_df, scope_trade_loss_df = _build_trade_summary_bundle(
        df_master=group_df,
        top_pct=top_pct,
        trade_floor=trade_floor,
        min_return=min_return,
    )

    scope_regime_map_df = _build_scope_regime_combo_df(
        df_scope=group_df,
        signal_layer=signal_layer,
        scope_text=scope_text,
        top_pct=top_pct,
        trade_floor=trade_floor,
    )

    return (
        summary_row,
        None,
        loss_report,
        scope_trade_summary_df,
        scope_trade_by_era_df,
        scope_trade_loss_df,
        scope_regime_map_df,
    )

def _trade_feature_report(
    df_master: pl.DataFrame,
    feature_col: str,
    feature_value: Any,
    top_pct: float = TOP_PCT,
    trade_floor: int = TRADE_FLOOR,
    min_return: float = MIN_RETURN,
) -> tuple[Optional[dict[str, Any]], Optional[pl.DataFrame], Optional[dict[str, Any]]]:
    if df_master is None or df_master.is_empty():
        return None, None, None

    if feature_col not in df_master.columns:
        return None, None, None

    df_group = df_master.filter(pl.col(feature_col) == feature_value)
    if df_group.is_empty():
        return None, None, None

    feature_name_df = df_master.filter(pl.col(feature_col).is_not_null())
    if feature_name_df.is_empty():
        feature_name_df = df_group

    summary_common, by_era_df, loss_common = _compute_group_report(
        df_group,
        parent_total_rows=int(feature_name_df.height),
        top_pct=top_pct,
        trade_floor=trade_floor,
        min_return=min_return,
    )
    if summary_common is None or by_era_df is None or loss_common is None:
        return None, None, None

    feature_value_text = _num(feature_value)
    feature_key = f"{feature_col}={feature_value_text}"

    summary_row = {
        "feature_name": feature_col,
        "feature_value": feature_value_text,
        "feature_key": feature_key,
        **summary_common,
    }

    loss_row = {
        "feature_name": feature_col,
        "feature_value": feature_value_text,
        "feature_key": feature_key,
        **loss_common,
    }

    by_era_df = by_era_df.with_columns(
        pl.lit(feature_col).alias("feature_name"),
        pl.lit(feature_value_text).alias("feature_value"),
        pl.lit(feature_key).alias("feature_key"),
    )

    return summary_row, by_era_df, loss_row


def _sorted_unique_values(series: pl.Series) -> list[Any]:
    values = [v for v in series.to_list() if v is not None]
    cleaned: list[Any] = []
    for v in values:
        if isinstance(v, float) and math.isnan(v):
            continue
        cleaned.append(v)

    def sort_key(x: Any):
        if isinstance(x, bool):
            return (0, int(x))
        if isinstance(x, int):
            return (1, x)
        if isinstance(x, float):
            return (2, x)
        return (3, str(x))

    return sorted(set(cleaned), key=sort_key)



def _write_signal_scope_reports(
    session_dir: Path,
    signal_layer: int,
    scope_text: str,
    scope_trade_summary_df: pl.DataFrame,
    scope_trade_by_era_df: pl.DataFrame,
    scope_trade_loss_df: pl.DataFrame,
    scope_regime_map_df: pl.DataFrame,
) -> Path:
    scope_root = _signal_root(session_dir) / f"layer_{int(signal_layer)}" / _safe_slug(scope_text)
    scope_root.mkdir(parents=True, exist_ok=True)

    scope_trade_summary_df = scope_trade_summary_df.with_columns(pl.lit(str(scope_root)).alias("scope_dir"))
    scope_trade_by_era_df = scope_trade_by_era_df.with_columns(pl.lit(str(scope_root)).alias("scope_dir"))
    scope_trade_loss_df = scope_trade_loss_df.with_columns(pl.lit(str(scope_root)).alias("scope_dir"))
    scope_regime_map_df = scope_regime_map_df.with_columns(pl.lit(str(scope_root)).alias("scope_dir"))

    _write_csv(scope_trade_summary_df, scope_root / "trade_management_feature_summary.csv")
    _write_csv(scope_trade_by_era_df, scope_root / "trade_management_feature_by_era.csv")
    _write_csv(scope_trade_loss_df, scope_root / "trade_management_feature_loss_summary.csv")
    _write_csv(scope_regime_map_df, scope_root / "regime_combo_map.csv")
    return scope_root


def _collect_signal_candidates(df_master: pl.DataFrame) -> list[tuple[int, str]]:
    if df_master is None or df_master.is_empty() or "signal_scope" not in df_master.columns:
        return []

    df_master = _ensure_master_cols(df_master)

    if "signal_layer" not in df_master.columns:
        df_master = df_master.with_columns(pl.lit(0, dtype=pl.Int64).alias("signal_layer"))

    pairs = (
        df_master.select(
            pl.col("signal_layer").cast(pl.Int64, strict=False).fill_null(0).alias("signal_layer"),
            pl.col("signal_scope").cast(pl.Utf8, strict=False).fill_null("").alias("signal_scope"),
        )
        .unique()
        .sort(["signal_layer", "signal_scope"])
    )

    out: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()

    for row in pairs.to_dicts():
        layer = int(row.get("signal_layer") or 0)
        scope_text = str(row.get("signal_scope") or "").strip()
        if not scope_text:
            continue

        candidates = [scope_text]
        candidates.extend(_atomic_signal_tokens(scope_text))

        for cand in candidates:
            key = (layer, cand)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)

    out.sort(key=lambda x: (x[0], x[1].count("|"), len(x[1]), x[1]))
    return out

def _build_trade_summary_bundle(
    df_master: pl.DataFrame,
    top_pct: float,
    trade_floor: int,
    min_return: float,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    trade_summary_rows: list[dict[str, Any]] = []
    trade_by_era_rows: list[dict[str, Any]] = []
    trade_loss_rows: list[dict[str, Any]] = []

    for feature_col in TRADE_MGMT_FEATURE_COLS:
        if feature_col not in df_master.columns:
            continue

        values = _sorted_unique_values(df_master.select(pl.col(feature_col)).to_series())
        for feature_value in values:
            summary_row, by_era_df, loss_row = _trade_feature_report(
                df_master=df_master,
                feature_col=feature_col,
                feature_value=feature_value,
                top_pct=top_pct,
                trade_floor=trade_floor,
                min_return=min_return,
            )
            if summary_row is None or by_era_df is None or loss_row is None:
                continue

            trade_summary_rows.append(summary_row)
            trade_by_era_rows.extend(by_era_df.to_dicts())
            trade_loss_rows.append(loss_row)

    trade_summary_df = pl.DataFrame(trade_summary_rows) if trade_summary_rows else pl.DataFrame(
        schema={
            "feature_name": pl.Utf8,
            "feature_value": pl.Utf8,
            "feature_key": pl.Utf8,
        }
    )

    if not trade_summary_df.is_empty():
        trade_summary_df = _add_joint_scores(trade_summary_df)
        trade_summary_df = trade_summary_df.sort(["elite_joint_score", "global_joint_score", "elite_median_alpha"], descending=[True, True, True])

    trade_by_era_df = pl.DataFrame(trade_by_era_rows) if trade_by_era_rows else pl.DataFrame(
        schema={
            "era_int": pl.Int64,
            "rows": pl.Int64,
            "median_balance": pl.Float64,
            "avg_dd": pl.Float64,
            "avg_win_rate": pl.Float64,
            "avg_total_pos": pl.Float64,
            "mean_max_consecutive_losses": pl.Float64,
            "median_max_consecutive_losses": pl.Float64,
            "max_max_consecutive_losses": pl.Int64,
            "median_return_per_trade": pl.Float64,
            "alpha_per_trade": pl.Float64,
            "era_recency_weight": pl.Float64,
            "feature_name": pl.Utf8,
            "feature_value": pl.Utf8,
            "feature_key": pl.Utf8,
        }
    )

    trade_loss_df = pl.DataFrame(trade_loss_rows) if trade_loss_rows else pl.DataFrame(
        schema={
            "feature_name": pl.Utf8,
            "feature_value": pl.Utf8,
            "feature_key": pl.Utf8,
            "global_mean_max_consecutive_losses": pl.Float64,
            "global_median_max_consecutive_losses": pl.Float64,
            "global_max_max_consecutive_losses": pl.Int64,
            "elite_mean_max_consecutive_losses": pl.Float64,
            "elite_median_max_consecutive_losses": pl.Float64,
            "elite_max_max_consecutive_losses": pl.Int64,
            "streak_gap_mean": pl.Float64,
            "streak_gap_median": pl.Float64,
        }
    )

    return _round_df(trade_summary_df), _round_df(trade_by_era_df), _round_df(trade_loss_df)



def export_session_analysis(
    session_dir: str | Path,
    df_master: Optional[pl.DataFrame] = None,
    trade_floor: int = TRADE_FLOOR,
    top_pct: float = TOP_PCT,
    min_return: float = MIN_RETURN,
    overwrite: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Writes:
      analysis/signal/signal_scope_summary.csv

      analysis/trade_management/trade_management_feature_summary.csv
      analysis/trade_management/trade_management_feature_by_era.csv
      analysis/trade_management/trade_management_feature_loss_summary.csv

    Per-scope trade-management reports are also written inside each signal layer folder.
    Signal-layer folders do not contain signal summary files.
    """
    session_dir = Path(session_dir)
    signal_root = _signal_root(session_dir)
    trade_root = _trade_mgmt_root(session_dir)
    signal_root.mkdir(parents=True, exist_ok=True)
    trade_root.mkdir(parents=True, exist_ok=True)

    if df_master is None:
        df_master = load_master_df(session_dir)

    df_master = _ensure_master_cols(df_master)
    df_master = _ensure_trade_mgmt_cols(df_master)

    if df_master.is_empty():
        empty = pl.DataFrame()
        _write_csv(empty, signal_root / "signal_scope_summary.csv")
        _write_csv(empty, trade_root / "trade_management_feature_summary.csv")
        _write_csv(empty, trade_root / "trade_management_feature_by_era.csv")
        _write_csv(empty, trade_root / "trade_management_feature_loss_summary.csv")
        return empty, empty

    trade_summary_df, trade_by_era_df, trade_loss_df = _build_trade_summary_bundle(
        df_master=df_master,
        top_pct=top_pct,
        trade_floor=trade_floor,
        min_return=min_return,
    )
    _write_csv(trade_summary_df, trade_root / "trade_management_feature_summary.csv")
    _write_csv(trade_by_era_df, trade_root / "trade_management_feature_by_era.csv")
    _write_csv(trade_loss_df, trade_root / "trade_management_feature_loss_summary.csv")

    signal_summary_rows: list[dict[str, Any]] = []

    candidates = _collect_signal_candidates(df_master)
    for signal_layer, scope_text in candidates:
        scope_summary, _, scope_loss, scope_trade_summary_df, scope_trade_by_era_df, scope_trade_loss_df, scope_regime_map_df = _signal_group_report(
            df_master=df_master,
            signal_layer=signal_layer,
            scope_text=scope_text,
            top_pct=top_pct,
            trade_floor=trade_floor,
            min_return=min_return,
        )

        if (
            scope_summary is None
            or scope_loss is None
            or scope_trade_summary_df is None
            or scope_trade_by_era_df is None
            or scope_trade_loss_df is None
            or scope_regime_map_df is None
        ):
            continue

        scope_root = _write_signal_scope_reports(
            session_dir=session_dir,
            signal_layer=signal_layer,
            scope_text=scope_text,
            scope_trade_summary_df=scope_trade_summary_df,
            scope_trade_by_era_df=scope_trade_by_era_df,
            scope_trade_loss_df=scope_trade_loss_df,
            scope_regime_map_df=scope_regime_map_df,
        )

        signal_summary_rows.append({**scope_summary, "scope_dir": str(scope_root)})

    signal_summary_df = pl.DataFrame(signal_summary_rows) if signal_summary_rows else pl.DataFrame(
        schema={
            "signal_layer": pl.Int64,
            "signal_scope": pl.Utf8,
            "scope_kind": pl.Utf8,
            "scope_size": pl.Int64,
            "families": pl.Utf8,
            "timeframes": pl.Utf8,
            "scope_dir": pl.Utf8,
        }
    )

    if not signal_summary_df.is_empty():
        signal_summary_df = _add_joint_scores(signal_summary_df)
        signal_summary_df = signal_summary_df.sort(
            ["signal_layer", "elite_joint_score", "global_joint_score", "elite_median_alpha"],
            descending=[False, True, True, True],
        )
        signal_summary_df = signal_summary_df.with_columns(
            pl.col("scope_dir").cast(pl.Utf8).fill_null("").alias("scope_dir")
        )

    signal_summary_df = _round_df(signal_summary_df)
    _write_csv(signal_summary_df, signal_root / "signal_scope_summary.csv")

    return signal_summary_df, trade_summary_df

if __name__ == "__main__":

    export_session_analysis(
        session_dir=PATH_DIR
    )
