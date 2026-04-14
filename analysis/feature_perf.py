from __future__ import annotations

import json
import math
from datetime import datetime
from itertools import combinations
from typing import Any, Iterable

import polars as pl

TOP_PCT = 0.1
TRADE_FLOOR = 40
MIN_RETURN = 0.05
RECENCY_DECAY_DAYS = 90.0


def _loads_maybe(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return x


def _leaf(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return _leaf(v[0])
        return json.dumps(v, separators=(",", ":"), sort_keys=True)
    if isinstance(v, dict):
        return json.dumps(v, separators=(",", ":"), sort_keys=True)
    return str(v)


def _scalar(v: Any) -> Any:
    v = _loads_maybe(v)
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return _scalar(v[0])
    return v


def _fmt_num(v: Any) -> str:
    v = _scalar(v)
    if v is None:
        return "na"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else str(v)
    return str(v)


def _thresholds_to_key(v: Any) -> str:
    v = _scalar(v)
    if v is None:
        return "na"

    if isinstance(v, (list, tuple)):
        # common case: [30, 70]
        if len(v) == 2 and not any(isinstance(i, (list, tuple, dict)) for i in v):
            return f"l{_fmt_num(v[0])}_u{_fmt_num(v[1])}"

        # nested case: [[30, 70]]
        parts = []
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                parts.append(f"l{_fmt_num(item[0])}_u{_fmt_num(item[1])}")
        if parts:
            return "__".join(parts)

    return _fmt_num(v)


def _stoch_key_row(row: dict[str, Any], timeframe: str = "15m", include_tolerance: bool = False) -> str:
    k = _scalar(row.get(f"signal__stochastic__{timeframe}__k"))
    d = _scalar(row.get(f"signal__stochastic__{timeframe}__d"))
    s = _scalar(row.get(f"signal__stochastic__{timeframe}__s"))
    thr = _thresholds_to_key(row.get(f"signal__stochastic__{timeframe}__thresholds"))
    tol = _scalar(row.get(f"signal__stochastic__{timeframe}__threshold_tolerance"))

    parts = [
        f"k{_fmt_num(k)}",
        f"d{_fmt_num(d)}",
        f"s{_fmt_num(s)}",
        thr,
    ]

    if include_tolerance and tol is not None:
        parts.append(f"tol{_fmt_num(tol)}")

    return "_".join(parts)


def _lookback_key_row(row: dict[str, Any], timeframe: str = "15m") -> str:
    lb = _scalar(row.get(f"signal__lookback__{timeframe}__entry_lookback_units"))
    return f"lb{_fmt_num(lb)}"


def prepare_master_df(df_master: pl.DataFrame) -> pl.DataFrame:
    if df_master is None or df_master.is_empty():
        return df_master

    has_signal_flat = any(c.startswith("signal__") for c in df_master.columns)
    if has_signal_flat:
        return df_master

    if "signal_json" not in df_master.columns:
        return df_master

    sig_map = (
        df_master
        .select("signal_json")
        .unique()
        .drop_nulls()
        .to_series()
        .to_list()
    )

    if not sig_map:
        return df_master

    sig_df = pl.from_dicts([flatten_signal_json(s) for s in sig_map])

    return df_master.join(
        pl.DataFrame({"signal_json": sig_map}).hstack(sig_df),
        on="signal_json",
        how="left",
    )


def add_human_readable_signal_keys(
    df_master: pl.DataFrame,
    timeframes: tuple[str, ...] = ("15m",),
    include_stoch_tolerance: bool = False,
) -> pl.DataFrame:
    df_master = prepare_master_df(df_master)

    if df_master is None or df_master.is_empty():
        return df_master

    out = df_master

    for tf in timeframes:
        stoch_cols = [
            f"signal__stochastic__{tf}__k",
            f"signal__stochastic__{tf}__d",
            f"signal__stochastic__{tf}__s",
            f"signal__stochastic__{tf}__thresholds",
        ]
        if all(c in out.columns for c in stoch_cols):
            out = out.with_columns(
                pl.struct(stoch_cols + [f"signal__stochastic__{tf}__threshold_tolerance"])
                .map_elements(
                    lambda row, _tf=tf: _stoch_key_row(row, timeframe=_tf, include_tolerance=include_stoch_tolerance),
                    return_dtype=pl.Utf8,
                )
                .alias(f"stoch_key ({tf})")
            )

        lookback_col = f"signal__lookback__{tf}__entry_lookback_units"
        if lookback_col in out.columns:
            out = out.with_columns(
                pl.struct([lookback_col])
                .map_elements(
                    lambda row, _tf=tf: _lookback_key_row(
                        {f"signal__lookback__{_tf}__entry_lookback_units": row[lookback_col]},
                        timeframe=_tf,
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias(f"lookback_key ({tf})")
            )

    return out


def _flatten(obj: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}__{k}" if prefix else str(k)
            _flatten(v, key, out)
    else:
        out[prefix] = _leaf(obj)

    return out


def _get_path(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur = obj
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def flatten_signal_json(s: str | dict | None) -> dict[str, Any]:
    obj = _loads_maybe(s)
    if not isinstance(obj, dict):
        return {}

    out: dict[str, Any] = {"signal_version": obj.get("version")}

    for k in ("search_mode", "cycle_idx", "block_cycle_idx", "block_idx", "block_refine_idx", "block_refine_rounds"):
        if k in obj:
            out[k] = _leaf(obj.get(k))

    seed = obj.get("seed")
    if isinstance(seed, dict):
        for k in (
            "pair",
            "BASE_MINUTES",
            "sl_tp_in_pct",
            "min_rr",
            "sl_tp_interval_months",
            "trailing_sl_pct",
            "trailing_sl_interval",
            "trailing_sl_stop_at_pos",
            "use_trailing_sl",
            "exit_windows_h",
            "trade_window_interval",
            "limit_order_expiry_bars",
            "use_limit_entry",
            "SL",
            "TP",
            "limit_order_expiry_h",
            "exit_window_h",
            "search_mode",
        ):
            if k in seed:
                out[f"seed__{k}"] = _leaf(seed.get(k))

    signal_block = None
    source_name = None
    for path, name in (
        (("seed", "signal_structure"), "seed.signal_structure"),
        (("incumbent", "regime_cfg", "signal_structure"), "incumbent.regime_cfg.signal_structure"),
        (("signal_structure",), "signal_structure"),
        (("signals",), "signals"),
    ):
        candidate = _get_path(obj, path)
        if isinstance(candidate, dict):
            signal_block = candidate
            source_name = name
            break

    if signal_block is not None:
        out["signal_source"] = source_name
        out.update(_flatten(signal_block, prefix="signal"))

    return out


def prepare_master_df(df_master: pl.DataFrame) -> pl.DataFrame:
    if df_master is None or df_master.is_empty():
        return df_master

    has_signal_flat = any(c.startswith("signal__") for c in df_master.columns)
    if has_signal_flat:
        return df_master

    if "signal_json" not in df_master.columns:
        return df_master

    sig_map = (
        df_master
        .select("signal_json")
        .unique()
        .drop_nulls()
        .to_series()
        .to_list()
    )

    if not sig_map:
        return df_master

    sig_df = pl.from_dicts([flatten_signal_json(s) for s in sig_map])

    return df_master.join(
        pl.DataFrame({"signal_json": sig_map}).hstack(sig_df),
        on="signal_json",
        how="left",
    )


def resolve_feature_col(df: pl.DataFrame, feature_col: str) -> str:
    candidates = [feature_col]

    if "__by_timeframe__" in feature_col:
        candidates.append(feature_col.replace("__by_timeframe__", "__"))

    if feature_col.startswith("signal__"):
        candidates.append(feature_col[len("signal__"):])

    if not feature_col.startswith("signal__"):
        candidates.append("signal__" + feature_col)

    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    for c in unique_candidates:
        if c in df.columns:
            return c

    raise ColumnNotFoundError(
        f"Feature column not found: {feature_col}\n"
        f"Tried: {unique_candidates}\n"
        f"Available signal columns: {[c for c in df.columns if c.startswith('signal__')][:40]}"
    )


def build_era_recency_weight_map(eras: Iterable[Any], decay_days: float = RECENCY_DECAY_DAYS) -> dict[int, float]:
    eras_clean = []
    for e in eras:
        try:
            eras_clean.append(int(e))
        except Exception:
            continue

    eras_clean = sorted(set(eras_clean))
    if not eras_clean:
        return {}

    latest_era = eras_clean[-1]
    latest_dt = datetime.strptime(str(latest_era), "%Y%m%d").date()

    weights = {}
    for era in eras_clean:
        era_dt = datetime.strptime(str(era), "%Y%m%d").date()
        age_days = max((latest_dt - era_dt).days, 0)
        weight = 1.0 / (1.0 + math.log1p(age_days / decay_days))
        weights[int(era)] = float(weight)

    return weights


def add_perf_cols(df: pl.DataFrame) -> pl.DataFrame:
    ret = ((pl.col("balance") - 100) / 100) / (pl.col("total_pos") + 1e-9)
    return df.with_columns([
        ret.alias("return_per_trade"),
        (ret * ret.exp() / (pl.col("max_drawdown") + 1e-9)).alias("alpha_per_trade"),
    ])


def _feature_value_expr(col_name: str) -> pl.Expr:
    return pl.col(col_name).cast(pl.Utf8, strict=False).fill_null("NULL")


def feature_performance(
    df_master: pl.DataFrame,
    feature_col: str,
    top_pct: float = TOP_PCT,
    trade_floor: int = TRADE_FLOOR,
    min_return: float = MIN_RETURN,
    verbose: bool = True,
):
    """
    Single-feature performance analysis.

    Returns:
        perf_by_era, top_elite_consistency
    """
    df_master = prepare_master_df(df_master)
    feature_col = resolve_feature_col(df_master, feature_col)

    if df_master is None or df_master.is_empty():
        return None, None

    df_filtered = df_master.filter(pl.col("total_pos") >= trade_floor)
    if df_filtered.is_empty():
        if verbose:
            print(f"No strategies found with total_pos >= {trade_floor}")
        return None, None

    df_consistency = df_filtered.filter(((pl.col("balance") - 100) / 100) >= min_return)
    df_eff_era = add_perf_cols(df_filtered)

    eras = sorted(df_eff_era["era_int"].unique().to_list())
    era_weight_map = build_era_recency_weight_map(eras, decay_days=RECENCY_DECAY_DAYS)
    total_era_weight = sum(era_weight_map.values()) if era_weight_map else 0.0

    perf_by_era = (
        df_filtered
        .group_by(["era_int", feature_col])
        .agg([
            pl.col("balance").median().alias("median_balance"),
            pl.col("max_drawdown").mean().alias("avg_dd"),
            (pl.col("win_pos").mean() / (pl.col("total_pos").mean() + 1e-9)).alias("avg_win_rate"),
            pl.col("total_pos").mean().alias("avg_total_pos"),
        ])
        .with_columns([
            (((pl.col("median_balance") - 100) / 100) / (pl.col("avg_total_pos") + 1e-9)).alias("median_return_per_trade"),
            (
                (((pl.col("median_balance") - 100) / 100) / (pl.col("avg_total_pos") + 1e-9))
                * ((((pl.col("median_balance") - 100) / 100) / (pl.col("avg_total_pos") + 1e-9)).exp())
                / (pl.col("avg_dd") + 1e-9)
            ).alias("alpha_per_trade"),
        ])
        .sort(["era_int", "alpha_per_trade"], descending=[False, True])
    )

    top_elite_rows = []
    for era in eras:
        era_df = df_eff_era.filter(pl.col("era_int") == era)
        cutoff = max(1, int(len(era_df) * top_pct))
        top_elite_rows.append(
            era_df.sort("alpha_per_trade", descending=True).head(cutoff)
        )

    top_elite_per_era = pl.concat(top_elite_rows) if top_elite_rows else pl.DataFrame()
    total_pool_size = max(top_elite_per_era.height, 1)

    consistency_presence = (
        df_consistency
        .select(["era_int", feature_col])
        .unique()
        .with_columns(
            pl.col("era_int").map_elements(
                lambda x: float(era_weight_map.get(int(x), 1.0)),
                return_dtype=pl.Float64,
            ).alias("era_recency_weight")
        )
    )

    global_stats = (
        df_eff_era
        .group_by(feature_col)
        .agg([
            pl.col("alpha_per_trade").median().alias("global_median_alpha"),
            pl.col("return_per_trade").median().alias("global_median_return"),
        ])
    )

    elite_quality = (
        top_elite_per_era
        .group_by(feature_col)
        .agg([
            (pl.len().cast(pl.Float64) / total_pool_size).alias("dominance_score"),
            pl.col("alpha_per_trade").median().alias("elite_median_alpha"),
            pl.col("return_per_trade").median().alias("elite_median_return"),
            pl.col("max_drawdown").mean().alias("avg_max_drawdown"),
        ])
    )

    elite_consistency = (
        consistency_presence
        .group_by(feature_col)
        .agg([
            pl.col("era_recency_weight").sum().alias("weighted_era_hits"),
            pl.col("era_int").n_unique().alias("_eras_found"),
        ])
        .with_columns([
            (pl.col("weighted_era_hits") / (total_era_weight + 1e-9)).alias("recency_weighted_era_consistency_score"),
            (pl.col("_eras_found") / max(len(eras), 1)).alias("era_consistency_score"),
        ])
    )

    top_elite_consistency = (
        elite_quality
        .join(elite_consistency, on=feature_col, how="left")
        .join(global_stats, on=feature_col, how="left")
        .with_columns([
            pl.col("weighted_era_hits").fill_null(0.0),
            pl.col("recency_weighted_era_consistency_score").fill_null(0.0),
            pl.col("era_consistency_score").fill_null(0.0),
            pl.col("elite_median_alpha").fill_null(0.0),
            pl.col("elite_median_return").fill_null(0.0),
            pl.col("global_median_alpha").fill_null(0.0),
            pl.col("global_median_return").fill_null(0.0),
            (pl.col("elite_median_alpha") - pl.col("global_median_alpha")).alias("alpha_lift"),
            (pl.col("elite_median_return") - pl.col("global_median_return")).alias("return_lift"),
        ])
        .drop("_eras_found")
        .sort(
            [
                "recency_weighted_era_consistency_score",
                "era_consistency_score",
                "dominance_score",
                "elite_median_alpha",
            ],
            descending=True,
        )
    )

    if verbose:
        print(f"\n--- {feature_col.upper()} PERFORMANCE BY ERA (FLOOR: {trade_floor}) ---")
        print(
            perf_by_era.select([
                "era_int",
                feature_col,
                "median_balance",
                "avg_dd",
                "avg_win_rate",
                "avg_total_pos",
                "median_return_per_trade",
                "alpha_per_trade",
            ])
        )

        with pl.Config(tbl_rows=200, tbl_width_chars=200):
            print(f"\n--- TOP ELITE PER ERA DOMINANCE: {feature_col} ---")
            print(top_elite_consistency)

    return perf_by_era, top_elite_consistency


def top_two_feature_combos(
    df_master: pl.DataFrame,
    features: list[str],
    top_pct: float = TOP_PCT,
    trade_floor: int = TRADE_FLOOR,
    min_return: float = MIN_RETURN,
    verbose: bool = True,
) -> pl.DataFrame | None:
    """
    Two-feature combo analysis.

    Returns a single long table across all feature pairs.
    """
    df_master = prepare_master_df(df_master)

    if df_master is None or df_master.is_empty():
        return None

    df_base = df_master.filter(pl.col("total_pos") >= trade_floor)
    if df_base.is_empty():
        if verbose:
            print(f"No strategies found with total_pos >= {trade_floor}")
        return None

    df_perf = add_perf_cols(df_base)
    df_consistency = df_base.filter(((pl.col("balance") - 100) / 100) >= min_return)

    eras = sorted(df_perf["era_int"].unique().to_list())
    if not eras:
        return None

    pair_tables = []

    for f1_raw, f2_raw in combinations(features, 2):
        try:
            f1 = resolve_feature_col(df_perf, f1_raw)
            f2 = resolve_feature_col(df_perf, f2_raw)
        except Exception:
            continue

        if f1 == f2:
            continue

        elite_rows = []
        for era in eras:
            era_df = df_perf.filter(pl.col("era_int") == era)
            if era_df.is_empty():
                continue

            cutoff = max(1, int(len(era_df) * top_pct))
            elite = (
                era_df
                .sort("alpha_per_trade", descending=True)
                .head(cutoff)
                .with_columns([
                    pl.lit(f1_raw).alias("feature_1"),
                    pl.lit(f1).alias("feature_col_1"),
                    _feature_value_expr(f1).alias("feature_value_1"),
                    pl.lit(f2_raw).alias("feature_2"),
                    pl.lit(f2).alias("feature_col_2"),
                    _feature_value_expr(f2).alias("feature_value_2"),
                ])
            )
            elite_rows.append(elite)

        if not elite_rows:
            continue

        elite_all = pl.concat(elite_rows)
        total_elite = max(elite_all.height, 1)

        consistency_rows = (
            df_consistency
            .with_columns([
                pl.lit(f1_raw).alias("feature_1"),
                pl.lit(f1).alias("feature_col_1"),
                _feature_value_expr(f1).alias("feature_value_1"),
                pl.lit(f2_raw).alias("feature_2"),
                pl.lit(f2).alias("feature_col_2"),
                _feature_value_expr(f2).alias("feature_value_2"),
            ])
            .select([
                "era_int",
                "feature_1",
                "feature_col_1",
                "feature_value_1",
                "feature_2",
                "feature_col_2",
                "feature_value_2",
            ])
            .unique()
        )

        final_table = (
            elite_all
            .group_by([
                "feature_1",
                "feature_col_1",
                "feature_value_1",
                "feature_2",
                "feature_col_2",
                "feature_value_2",
            ])
            .agg([
                (pl.len() / total_elite).alias("dominance_score"),
                pl.col("alpha_per_trade").median().alias("elite_median_alpha"),
                pl.col("return_per_trade").median().alias("elite_median_return"),
            ])
        )

        if not consistency_rows.is_empty():
            consistency_table = (
                consistency_rows
                .group_by([
                    "feature_1",
                    "feature_col_1",
                    "feature_value_1",
                    "feature_2",
                    "feature_col_2",
                    "feature_value_2",
                ])
                .agg([
                    pl.col("era_int").n_unique().alias("eras_hit"),
                ])
                .with_columns(
                    (pl.col("eras_hit") / len(eras)).alias("era_consistency_score")
                )
                .drop("eras_hit")
            )

            final_table = (
                final_table
                .join(
                    consistency_table,
                    on=[
                        "feature_1",
                        "feature_col_1",
                        "feature_value_1",
                        "feature_2",
                        "feature_col_2",
                        "feature_value_2",
                    ],
                    how="left",
                )
                .with_columns(
                    pl.col("era_consistency_score").fill_null(0.0)
                )
            )
        else:
            final_table = final_table.with_columns(
                pl.lit(0.0).alias("era_consistency_score")
            )

        pair_tables.append(final_table)

    if not pair_tables:
        if verbose:
            print("No pair results found.")
        return None

    out = pl.concat(pair_tables).sort(
        ["era_consistency_score", "dominance_score", "elite_median_alpha"],
        descending=True,
    )

    if verbose:
        print("\n=== TOP 2-FEATURE VALUE COMBINATIONS ===")
        print(out.head(40))

    return out


def list_signal_columns(df_master: pl.DataFrame) -> list[str]:
    df_master = prepare_master_df(df_master)
    return [c for c in df_master.columns if c.startswith("signal__")]