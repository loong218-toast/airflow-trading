import json
import math
import logging
from pathlib import Path
from typing import List
from itertools import product

from etl.grid import _expand_sl_tp, _prune_by_min_rr

# Assuming logger is initialized globally in your project
logger = logging.getLogger(__name__)

def _list(x, default):
    if x is None:
        return list(default)
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]

def _stoch_opts(cfg, use_stoch_list):
    out = []

    ks = _list(cfg.get("stoch_k"), [12]) if "stoch_k" in cfg else _list(cfg.get("k"), [12])
    ds = _list(cfg.get("stoch_d"), [3]) if "stoch_d" in cfg else _list(cfg.get("d"), [3])
    ss = _list(cfg.get("stoch_s"), [3]) if "stoch_s" in cfg else _list(cfg.get("s"), [3])
    ths = _list(cfg.get("stoch_thresholds"), [[20, 80]]) if "stoch_thresholds" in cfg else _list(cfg.get("thresholds"), [[20, 80]])

    for use_stoch in use_stoch_list:
        if not use_stoch:
            out.append({
                "use_stochastic": False,
                "stoch_key": "OFF",
                "col": None,
                "low": None,
                "high": None,
            })
            continue

        for k, d, s, t in product(ks, ds, ss, ths):
            if not isinstance(t, (list, tuple)) or len(t) != 2:
                raise ValueError(f"Invalid stoch_threshold entry: {t!r}")

            low = float(t[0])
            high = float(t[1])

            out.append({
                "use_stochastic": True,
                "stoch_key": f"k{k}_d{d}_s{s}_l{low:g}_u{high:g}",
                "col": f"stoch_k{k}_d{d}_s{s}",
                "low": low,
                "high": high,
            })

    return out

def _bbw_opts(cfg, use_bbw_list):
    out = []

    periods = [int(x) for x in _list(cfg.get("bbw_periods"), [96]) if int(x) > 0]
    stds = [float(x) for x in _list(cfg.get("bbw_std"), [2.5])]
    ths = [float(x) for x in _list(cfg.get("bbw_thresholds"), [50])]

    for use_bbw in use_bbw_list:
        if not use_bbw:
            out.append({
                "use_bbw": False,
                "bbw_periods": 0,
                "bbw_std": 0.0,
                "bbw_thresholds": 0.0,
            })
            continue

        for p, s, t in product(periods, stds, ths):
            out.append({
                "use_bbw": True,
                "bbw_periods": p,
                "bbw_std": s,
                "bbw_thresholds": t,
            })

    return out

def _write_batch(path: Path, batch_id: int, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf8") as f:
        json.dump({"batch_id": batch_id, "regimes": rows}, f, indent=2)

def _ma_types_for_periods(run_cfg: dict, count: int) -> list[str]:
    if count <= 0:
        return []

    raw = run_cfg.get("ma_types", None)

    if raw is None:
        return ["sma"] * count

    if isinstance(raw, str):
        v = raw.strip().lower() or "sma"
        return [v] * count

    vals = _list(raw, [])
    if not vals:
        return ["sma"] * count

    if len(vals) == 1:
        v = str(vals[0]).strip().lower() or "sma"
        return [v] * count

    out = [str(x).strip().lower() or "sma" for x in vals]
    if len(out) < count:
        out.extend(["sma"] * (count - len(out)))
    return out[:count]

def generate_configs(session_dir: Path, run_cfg: dict) -> list[Path]:
    cfg_dir = session_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    exit_windows = _list(run_cfg.get("exit_windows"), [24])
    lookbacks = _list(run_cfg.get("entry_lookback_units"), [24])

    sl_vals, tp_vals = _expand_sl_tp(run_cfg)
    combos = _prune_by_min_rr(sl_vals, tp_vals, float(run_cfg.get("min_rr", 3.0)))

    ma_periods = run_cfg.get("ma_periods", []) or []
    if not isinstance(ma_periods, list):
        ma_periods = list(ma_periods)

    ma_periods_sorted = sorted({int(x) for x in ma_periods if int(x) > 0})
    ma_types_sorted = _ma_types_for_periods(run_cfg, len(ma_periods_sorted))

    n_bits = len(ma_periods_sorted)
    max_ma_int = (1 << n_bits) if n_bits > 0 else 1

    ma_reversion_list = _list(run_cfg.get("ma_reversion"), [False])
    use_stoch_list = _list(run_cfg.get("use_stochastic"), [False, True])
    use_bbw_list = _list(run_cfg.get("use_bbw"), [False, True])

    stoch_opts = _stoch_opts(run_cfg, use_stoch_list)
    bbw_opts = _bbw_opts(run_cfg, use_bbw_list)

    use_sl_decay_list = _list(run_cfg.get("use_sl_decay"), [False])
    sl_decay_pct_list = _list(run_cfg.get("sl_decay_pct"), [0.0])
    sl_decay_interval_list = _list(run_cfg.get("sl_decay_interval"), [12])
    sl_decay_stop_at_pos_list = _list(run_cfg.get("sl_decay_stop_at_pos"), [True])

    limit_order_expiry_h_list = _list(run_cfg.get("limit_order_expiry_h"), [24])

    trade_window_interval_list = _list(run_cfg.get("trade_window_interval"), [0])

    all_regime_configs = []
    idx = 0

    for ma_rev in ma_reversion_list:
        for ma_int in range(0, max_ma_int):
            for st in stoch_opts:
                for bw in bbw_opts:
                    for lb_h in lookbacks:
                        for exit_h in exit_windows:
                            for use_decay in use_sl_decay_list:
                                for decay_pct in sl_decay_pct_list:
                                    for decay_interval in sl_decay_interval_list:
                                        for decay_stop in sl_decay_stop_at_pos_list:
                                            for limit_h in limit_order_expiry_h_list:
                                                for trade_window_interval in trade_window_interval_list:
                                                    regime = {
                                                        "regime_id": f"{idx:05d}",
                                                        "ma_int": int(ma_int),
                                                        "ma_reversion": bool(ma_rev),
                                                        "ma_periods": ma_periods_sorted,
                                                        "ma_types": ma_types_sorted,
                                                        "use_stochastic": bool(st["use_stochastic"]),
                                                        "stoch_key": st["stoch_key"],
                                                        "stoch_col": st["col"],
                                                        "stoch_lower": st["low"],
                                                        "stoch_upper": st["high"],
                                                        "stoch_threshold_tolerance": float(run_cfg.get("stoch_threshold_tolerance", 50)),
                                                        "use_bbw": bool(bw["use_bbw"]),
                                                        "bbw_periods": int(bw["bbw_periods"]),
                                                        "bbw_std": float(bw["bbw_std"]),
                                                        "bbw_thresholds": float(bw["bbw_thresholds"]),
                                                        "entry_lookback_units": int(lb_h),
                                                        "exit_window_h": int(exit_h),
                                                        "use_sl_decay": bool(use_decay),
                                                        "sl_decay_pct": float(decay_pct),
                                                        "sl_decay_interval": int(decay_interval),
                                                        "sl_decay_stop_at_pos": bool(decay_stop),
                                                        "limit_order_expiry_h": int(limit_h),
                                                        "trade_window_interval": int(trade_window_interval),
                                                        "use_limit_entry": bool(run_cfg.get("use_limit_entry", True)),
                                                    }
                                                    all_regime_configs.append(regime)
                                                    idx += 1

    total_regimes = len(all_regime_configs)
    batch_size = int(run_cfg.get("BATCH_SIZE", 150))
    saved_batch_paths = []

    logger.info("=" * 60)
    logger.info("🚀 GRID CONFIG GENERATION (BATCH MODE)")
    logger.info("📈 Total Unique Regimes: %d", total_regimes)
    logger.info("📦 Batch Size: %d | Expected Tasks: %d", batch_size, math.ceil(total_regimes / batch_size))
    logger.info("🧪 SL/TP combos per regime: %d", len(combos))
    logger.info("📊 Total Backtests to be run: %d", total_regimes * len(combos))
    logger.info("=" * 60)

    for i in range(0, total_regimes, batch_size):
        batch_slice = all_regime_configs[i : i + batch_size]
        batch_num = i // batch_size
        batch_payload = {"batch_id": batch_num, "regimes": batch_slice}
        batch_filename = cfg_dir / f"batch_{batch_num:04d}.json"

        with open(batch_filename, "w", encoding="utf8") as f:
            json.dump(batch_payload, f, indent=2)

        saved_batch_paths.append(batch_filename)
        if len(saved_batch_paths) % 20 == 0 or i + batch_size >= total_regimes:
            logger.info("📝 Written %d batch files...", len(saved_batch_paths))

    return saved_batch_paths

def list_pending_config_paths(session_dir: Path) -> List[str]:
    cfg_dir = session_dir / "configs"
    results_dir = session_dir / "results"
    batch_files = sorted(cfg_dir.glob("batch_*.json"))
    pending_batches = []

    for batch_path in batch_files:
        try:
            with open(batch_path, "r", encoding="utf8") as f:
                batch_data = json.load(f)

            regimes = batch_data.get("regimes", [])

            is_batch_complete = True
            for r in regimes:
                regime_id = r.get("regime_id")
                result_file = results_dir / f"cfg_{regime_id}_summary.json"
                if not result_file.exists():
                    is_batch_complete = False
                    break
            if not is_batch_complete:
                pending_batches.append(str(batch_path))
        except Exception:
            pending_batches.append(str(batch_path))
    logger.info("🔍 Checked %d batches: %d still pending.", len(batch_files), len(pending_batches))
    return pending_batches