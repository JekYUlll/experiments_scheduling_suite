from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.plots.strategy_compare_plot import StrategySeries, plot_strategy_comparison
from experiments_scheduling_suite.src.preprocessing.split import time_split
from experiments_scheduling_suite.src.utils.io import load_yaml, read_csv_with_timestamp


@dataclass
class RunMeta:
    run_id: str
    run_dir: Path
    missing_name: str
    missing_label: str
    impute_name: str
    cfg: Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="跨调度/插值策略绘制预测曲线对比图。")
    parser.add_argument("--run-prefix", type=str, default=None, help="仅处理以该前缀开头的 run_id")
    parser.add_argument("--models", type=str, default=None, help="逗号分隔的模型名，默认自动扫描")
    parser.add_argument("--horizons", type=str, default="1", help="逗号分隔的 horizon，例如 1,2,3")
    parser.add_argument("--missingness", type=str, default=None, help="仅保留指定调度算法名（逗号分隔）")
    parser.add_argument("--imputations", type=str, default=None, help="仅保留指定插值算法名（逗号分隔）")
    parser.add_argument("--reference-run", type=str, default=None, help="指定参考 run_id（用于 true 曲线）")
    parser.add_argument("--reference-missingness", type=str, default=None, help="参考调度算法名（用于选择 true 曲线）")
    parser.add_argument("--reference-imputer", type=str, default=None, help="参考插值算法名（用于选择 true 曲线）")
    parser.add_argument("--no-inverse", action="store_true", help="不做反归一化，直接使用标准化值")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录（默认 reports/_aggregate/figures/strategy_preds）")
    return parser.parse_args()


def _format_missingness(mcfg: Dict) -> str:
    name = mcfg.get("name", "missing")
    if name == "mcar":
        return f"mcar_p{mcfg.get('p_missing', '')}"
    if name == "block":
        return f"block_b{mcfg.get('n_blocks', '')}"
    if name == "duty_cycle":
        return f"duty_cycle_p{mcfg.get('period_steps', '')}_on{mcfg.get('on_steps', '')}"
    if name == "round_robin":
        return f"round_robin_k{mcfg.get('budget_k', '')}"
    if name == "info_priority":
        return f"info_priority_k{mcfg.get('budget_k', '')}"
    return name


def _scan_runs(reports_dir: Path, run_prefix: Optional[str]) -> List[RunMeta]:
    runs: List[RunMeta] = []
    for run_dir in sorted(reports_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        if run_prefix and not run_dir.name.startswith(run_prefix):
            continue
        cfg_path = run_dir / "config_used.yaml"
        cfg = load_yaml(cfg_path)
        if not cfg:
            continue
        missing_cfg = cfg.get("missingness", {})
        impute_cfg = cfg.get("imputation", {})
        runs.append(
            RunMeta(
                run_id=run_dir.name,
                run_dir=run_dir,
                missing_name=str(missing_cfg.get("name", "missing")),
                missing_label=_format_missingness(missing_cfg),
                impute_name=str(impute_cfg.get("name", "impute")),
                cfg=cfg,
            )
        )
    return runs


def _collect_models(runs: Iterable[RunMeta]) -> List[str]:
    models = set()
    for run in runs:
        preds_dir = run.run_dir / "preds"
        if not preds_dir.exists():
            continue
        for path in preds_dir.glob("*_h*.csv"):
            name = path.stem
            if "_h" not in name:
                continue
            model, _ = name.rsplit("_h", 1)
            models.add(model)
    return sorted(models)


def _load_pred_df(run: RunMeta, model: str, horizon: int) -> Optional[pd.DataFrame]:
    path = run.run_dir / "preds" / f"{model}_h{horizon}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    return df


def _get_scaler(run: RunMeta) -> Optional[Tuple[float, float]]:
    cfg = run.cfg
    paths = cfg.get("paths", {})
    processed_root = Path(paths.get("processed_dir", "data/processed"))
    processed_dir = PROJECT_ROOT / processed_root / run.run_id
    imputed_path = processed_dir / "imputed.csv"
    if not imputed_path.exists():
        return None
    df = read_csv_with_timestamp(imputed_path)
    split_cfg = cfg.get("split", {})
    if not split_cfg:
        return None
    splits = time_split(df, **split_cfg)
    target_col = cfg.get("run", {}).get("target", "wind_speed_ms")
    if target_col not in df.columns:
        return None
    train_df = df.iloc[splits["train"]]
    mean = float(train_df[target_col].mean())
    std = float(train_df[target_col].std()) or 1.0
    return mean, std


def _apply_inverse(df: pd.DataFrame, scaler: Optional[Tuple[float, float]]) -> pd.DataFrame:
    if scaler is None:
        return df
    mean, std = scaler
    out = df.copy()
    out["y_true"] = out["y_true"] * std + mean
    out["y_pred"] = out["y_pred"] * std + mean
    return out


def _pick_reference(runs: List[RunMeta], args: argparse.Namespace) -> Optional[RunMeta]:
    if args.reference_run:
        for run in runs:
            if run.run_id == args.reference_run:
                return run
    if args.reference_missingness or args.reference_imputer:
        for run in runs:
            if args.reference_missingness and run.missing_name != args.reference_missingness:
                continue
            if args.reference_imputer and run.impute_name != args.reference_imputer:
                continue
            return run
    return runs[0] if runs else None


def main() -> None:
    args = parse_args()
    reports_dir = PROJECT_ROOT / "reports"
    runs = _scan_runs(reports_dir, args.run_prefix)

    if args.missingness:
        allow_missing = {s.strip() for s in args.missingness.split(",") if s.strip()}
        runs = [r for r in runs if r.missing_name in allow_missing]
    if args.imputations:
        allow_impute = {s.strip() for s in args.imputations.split(",") if s.strip()}
        runs = [r for r in runs if r.impute_name in allow_impute]

    if not runs:
        raise SystemExit("No runs matched the filters.")

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else _collect_models(runs)
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    out_dir = args.out_dir or (reports_dir / "_aggregate" / "figures" / "strategy_preds")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 颜色映射：插值算法
    imputer_names = sorted({r.impute_name for r in runs})
    cmap = plt.get_cmap("tab10", max(1, len(imputer_names)))
    imputer_colors = {name: cmap(i) for i, name in enumerate(imputer_names)}
    # 线型映射：调度算法（含参数）
    missing_labels = sorted({r.missing_label for r in runs})
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (1, 1))]
    missing_styles = {name: styles[i % len(styles)] for i, name in enumerate(missing_labels)}

    for model in models:
        for h in horizons:
            ref_run = _pick_reference(runs, args)
            if ref_run is None:
                continue
            ref_df = _load_pred_df(ref_run, model, h)
            if ref_df is None:
                continue
            scaler_cache: Dict[str, Optional[Tuple[float, float]]] = {}
            if not args.no_inverse:
                scaler_cache[ref_run.run_id] = _get_scaler(ref_run)
                ref_df = _apply_inverse(ref_df, scaler_cache[ref_run.run_id])

            ref_df = ref_df.dropna(subset=["timestamp"]).set_index("timestamp")
            timestamps = ref_df.index.to_numpy()
            y_true = ref_df["y_true"].to_numpy()

            series: List[StrategySeries] = []
            for run in runs:
                df = _load_pred_df(run, model, h)
                if df is None:
                    continue
                if not args.no_inverse:
                    if run.run_id not in scaler_cache:
                        scaler_cache[run.run_id] = _get_scaler(run)
                    df = _apply_inverse(df, scaler_cache[run.run_id])
                df = df.set_index("timestamp")
                aligned = ref_df[["y_true"]].join(df[["y_pred"]], how="left")
                y_pred = aligned["y_pred"].to_numpy()
                if np.isfinite(y_pred).sum() < max(10, int(len(y_pred) * 0.2)):
                    continue
                series.append(
                    StrategySeries(
                        label=f"{run.missing_label}|{run.impute_name}",
                        missingness=run.missing_label,
                        imputation=run.impute_name,
                        y_pred=y_pred,
                    )
                )

            if not series:
                continue
            out_path = out_dir / f"{model}_h{h}.png"
            title = f"{model} - strategy comparison (H={h})"
            plot_strategy_comparison(timestamps, y_true, series, imputer_colors, missing_styles, out_path, title, h)
            print(f"Saved strategy comparison -> {out_path}")


if __name__ == "__main__":
    main()
