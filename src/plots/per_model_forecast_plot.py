from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from experiments_scheduling_suite.src.plots.style import apply_style


def plot_predictions(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: list[int],
    out_path: Path,
    title: str,
) -> None:
    """将 h=1/2/3 预测与真实值画在同一张图中，并补充残差子图。"""
    apply_style()
    n_h = len(horizons)
    fig, axes = plt.subplots(n_h + 1, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [1] * n_h + [0.8]})
    for i, h in enumerate(horizons):
        ax = axes[i]
        ax.plot(timestamps, y_true[:, i], label="true", linewidth=1.0)
        ax.plot(timestamps, y_pred[:, i], label="pred", linewidth=1.0)
        ax.set_title(f"H={h}")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    # 残差子图：展示 pred-true，放大 y 轴以便观察细小差异
    ax_res = axes[-1]
    residuals = []
    for i, h in enumerate(horizons):
        res = y_pred[:, i] - y_true[:, i]
        residuals.append(res)
        ax_res.plot(timestamps, res, label=f"H={h}", linewidth=0.9)
    ax_res.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax_res.set_title("Residuals (pred - true)")
    ax_res.grid(alpha=0.3)
    # 用分位数缩放 y 轴，避免被极端值拉爆
    all_res = np.concatenate([r[~np.isnan(r)] for r in residuals]) if residuals else np.array([0.0])
    if all_res.size > 0:
        max_abs = np.nanpercentile(np.abs(all_res), 99)
        if not np.isfinite(max_abs) or max_abs <= 0:
            max_abs = np.nanmax(np.abs(all_res)) if np.isfinite(all_res).any() else 1.0
        max_abs = max(max_abs, 1e-6)
        ax_res.set_ylim(-max_abs, max_abs)
    ax_res.legend(fontsize=7, ncol=min(3, max(1, len(horizons))))
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
