from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from experiments_scheduling_suite.src.plots.style import apply_style


def _gap_lengths(mask: np.ndarray) -> List[int]:
    lengths: List[int] = []
    current = 0
    for missing in mask:
        if missing:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def plot_missingness_heatmap(
    mask_df: pd.DataFrame,
    out_path: Path,
    max_cols_per_panel: int = 20,
    max_time_points: int = 1200,
) -> None:
    """时间×变量的缺失热力图（自动降采样 + 分面，避免标签重叠）。"""
    apply_style()
    work = mask_df.copy()
    n_time = len(work)
    stride = max(1, int(np.ceil(n_time / max_time_points)))
    if stride > 1:
        work = work.iloc[::stride].reset_index(drop=True)

    cols = list(work.columns)
    n_cols = len(cols)
    if n_cols == 0:
        return

    # 按列分面，避免标签拥挤
    chunks = [cols[i : i + max_cols_per_panel] for i in range(0, n_cols, max_cols_per_panel)]
    n_panels = len(chunks)
    fig_h = max(3.5, sum(max(2.2, 0.25 * len(chunk)) for chunk in chunks))
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, fig_h), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, chunk in zip(axes, chunks):
        data = work[chunk].to_numpy().T
        ax.imshow(data, aspect="auto", cmap="gray_r", interpolation="nearest")
        ax.set_yticks(range(len(chunk)))
        ax.set_yticklabels(chunk, fontsize=7)
        ax.grid(False)

    # x 轴刻度简化
    x_max = work.shape[0] - 1
    if x_max > 0:
        ticks = np.linspace(0, x_max, num=6, dtype=int)
        for ax in axes:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(t * stride)) for t in ticks], fontsize=8)
    axes[-1].set_xlabel("time index")

    title = "Missingness heatmap (1=observed, 0=missing)"
    if stride > 1:
        title += f" (downsample x{stride})"
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_gap_distribution(mask_df: pd.DataFrame, out_path: Path) -> None:
    """缺失段长度分布（直方图）。"""
    apply_style()
    gaps: List[int] = []
    for col in mask_df.columns:
        gaps.extend(_gap_lengths(mask_df[col].to_numpy() == 0))
    fig, ax = plt.subplots()
    if gaps:
        ax.hist(gaps, bins=40, alpha=0.8)
    ax.set_xlabel("gap length (steps)")
    ax.set_ylabel("count")
    ax.set_title("Gap length distribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_overlay(original: pd.DataFrame, masked: pd.DataFrame, imputed: pd.DataFrame, target: str, out_path: Path) -> None:
    """原始/遮罩/插补序列叠加示例。"""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(original["timestamp"], original[target], label="original", linewidth=1.0)
    ax.plot(masked["timestamp"], masked[target], label="masked", linewidth=1.0, alpha=0.7)
    ax.plot(imputed["timestamp"], imputed[target], label="imputed", linewidth=1.2)
    ax.set_title(f"Overlay - {target}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, cols: List[str], out_path: Path) -> None:
    """关键特征分布直方图。"""
    apply_style()
    n = len(cols)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            ax.text(0.5, 0.5, "无有效数据", ha="center", va="center", fontsize=9)
            ax.set_title(col)
            ax.set_axis_off()
            continue
        # 如果取值范围过小，避免直方图分箱失败
        vmin = float(series.min())
        vmax = float(series.max())
        if np.isclose(vmin, vmax):
            ax.axvline(vmin, color="tab:blue")
            ax.set_title(f"{col} (常数)")
            continue
        # 动态调整分箱数量，避免“Too many bins”错误
        bins = min(40, max(5, int(np.sqrt(len(series)))))
        ax.hist(series, bins=bins, alpha=0.8)
        ax.set_title(col)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _kde_1d(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    if values.size < 2:
        return None
    std = float(np.std(values))
    if std == 0.0:
        return None
    n = values.size
    bw = 1.06 * std * (n ** (-1 / 5))
    bw = max(bw, std * 1e-3)
    diffs = (grid[:, None] - values[None, :]) / bw
    kernel = np.exp(-0.5 * diffs**2)
    norm = n * bw * np.sqrt(2 * np.pi)
    return kernel.sum(axis=1) / norm


def plot_feature_kde_comparison(
    original: pd.DataFrame,
    masked: pd.DataFrame,
    imputed: pd.DataFrame,
    cols: List[str],
    out_path: Path,
    max_cols: int = 8,
    min_observed_ratio: float = 0.01,
) -> None:
    """原始/遮罩/插补的 KDE 分布对比。"""
    apply_style()
    # 仅使用三份数据都包含的列，避免 mask-aware 特征导致 KeyError
    common_cols = [c for c in cols if c in original.columns and c in masked.columns and c in imputed.columns]
    observed_ratio = {}
    for col in common_cols:
        series = pd.to_numeric(masked[col], errors="coerce")
        observed_ratio[col] = float(series.notna().mean())
    candidates = [c for c in common_cols if observed_ratio.get(c, 0.0) >= min_observed_ratio]
    if candidates:
        candidates = sorted(candidates, key=lambda c: observed_ratio.get(c, 0.0), reverse=True)
        use_cols = candidates[:max_cols]
    else:
        # 如果全部是全缺失，退而选观测比例最高的列并标注
        use_cols = sorted(common_cols, key=lambda c: observed_ratio.get(c, 0.0), reverse=True)[:max_cols]
    if not use_cols:
        return
    n = len(use_cols)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    axes = np.array(axes).reshape(-1)
    legend_handles = [
        Line2D([0], [0], color="tab:blue", label="original"),
        Line2D([0], [0], color="tab:orange", label="masked"),
        Line2D([0], [0], color="tab:green", label="imputed"),
    ]
    for ax, col in zip(axes, use_cols):
        s_ori = pd.to_numeric(original[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        s_mask = pd.to_numeric(masked[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        s_imp = pd.to_numeric(imputed[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        all_vals = np.concatenate([s_ori, s_mask, s_imp]) if s_ori.size + s_mask.size + s_imp.size > 0 else np.array([])
        if all_vals.size < 2:
            ax.text(0.5, 0.5, "无有效数据", ha="center", va="center", fontsize=9)
            ax.set_title(col)
            ax.set_axis_off()
            continue
        vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
        if np.isclose(vmin, vmax):
            ax.axvline(vmin, color="tab:blue", label="original")
            ax.set_title(f"{col} (常数)")
            continue
        grid = np.linspace(vmin, vmax, 200)
        notes: List[str] = []
        for series, label, color in [
            (s_ori, "original", "tab:blue"),
            (s_mask, "masked", "tab:orange"),
            (s_imp, "imputed", "tab:green"),
        ]:
            if series.size < 2:
                notes.append(f"{label}=all-missing")
                continue
            if np.isclose(float(np.std(series)), 0.0):
                ax.axvline(float(series[0]), color=color, linestyle="--", linewidth=1.2, alpha=0.8)
                notes.append(f"{label}=constant")
                continue
            density = _kde_1d(series, grid)
            if density is not None:
                ax.plot(grid, density, label=label, color=color, linewidth=1.5)
        ax.set_title(col)
        if observed_ratio.get(col, 0.0) < min_observed_ratio:
            notes.append("masked=very-sparse")
        if notes:
            ax.text(0.98, 0.95, "; ".join(notes), ha="right", va="top", fontsize=7, transform=ax.transAxes)
    for ax in axes[n:]:
        ax.axis("off")
    fig.legend(handles=legend_handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_pretrain_viz(
    original: pd.DataFrame,
    masked: pd.DataFrame,
    imputed: pd.DataFrame,
    mask_df: pd.DataFrame,
    target: str,
    out_dir: Path,
    feature_cols: Optional[List[str]] = None,
) -> None:
    """统一入口：生成预训练可视化。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_missingness_heatmap(mask_df, out_dir / "missingness_heatmap.png")
    plot_gap_distribution(mask_df, out_dir / "gap_length_hist.png")
    plot_overlay(original, masked, imputed, target, out_dir / "overlay.png")
    if feature_cols:
        plot_feature_distributions(imputed, feature_cols, out_dir / "feature_distributions.png")
        plot_feature_kde_comparison(original, masked, imputed, feature_cols, out_dir / "feature_kde_comparison.png")
