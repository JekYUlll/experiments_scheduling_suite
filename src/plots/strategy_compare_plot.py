from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments_scheduling_suite.src.plots.style import apply_style


@dataclass
class StrategySeries:
    """单条策略曲线信息。"""

    label: str
    missingness: str
    imputation: str
    y_pred: np.ndarray


def plot_strategy_comparison(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    series: List[StrategySeries],
    imputer_colors: Dict[str, str],
    missing_styles: Dict[str, str],
    out_path: Path,
    title: str,
    horizon: int,
) -> None:
    """绘制不同调度/插值策略下的预测曲线对比图。"""
    apply_style()
    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(13, 4))

    # 主图：全段对比
    ax_main.plot(timestamps, y_true, label="true", linewidth=1.2, color="black")
    for item in series:
        ax_main.plot(
            timestamps,
            item.y_pred,
            label=item.label,
            linewidth=0.9,
            alpha=0.9,
            color=imputer_colors.get(item.imputation, "tab:blue"),
            linestyle=missing_styles.get(item.missingness, "-"),
        )
    ax_main.set_title(f"Overlay (H={horizon})")
    ax_main.grid(alpha=0.3)

    # 缩放图：取中间片段，方便观察细微差异
    n = len(timestamps)
    start = int(n * 0.4)
    end = int(n * 0.5)
    if n > 0:
        ax_main.axvspan(timestamps[start], timestamps[end - 1], color="gray", alpha=0.15, linestyle="--")
    ax_zoom.plot(timestamps[start:end], y_true[start:end], label="true", linewidth=1.2, color="black")
    for item in series:
        ax_zoom.plot(
            timestamps[start:end],
            item.y_pred[start:end],
            linewidth=0.9,
            alpha=0.9,
            color=imputer_colors.get(item.imputation, "tab:blue"),
            linestyle=missing_styles.get(item.missingness, "-"),
        )
    ax_zoom.set_title(f"Zoom-in (H={horizon})")
    ax_zoom.grid(alpha=0.3)

    # 图例：颜色=插值算法，线型=调度算法
    imputer_handles = [
        Line2D([0], [0], color=imputer_colors[name], linestyle="-", lw=1.5)
        for name in imputer_colors
    ]
    missing_handles = [
        Line2D([0], [0], color="gray", linestyle=missing_styles[name], lw=1.5)
        for name in missing_styles
    ]
    true_handles = [Line2D([0], [0], color="black", linestyle="-", lw=1.5)]
    # 右侧留白放图例，避免遮挡曲线
    fig.legend(imputer_handles, list(imputer_colors.keys()), loc="center right", bbox_to_anchor=(0.98, 0.72), fontsize=7, frameon=False, title="Imputation")
    fig.legend(missing_handles, list(missing_styles.keys()), loc="center right", bbox_to_anchor=(0.98, 0.36), fontsize=7, frameon=False, title="Missingness")
    fig.legend(true_handles, ["true"], loc="center right", bbox_to_anchor=(0.98, 0.14), fontsize=7, frameon=False, title="Series")

    fig.suptitle(title)
    fig.subplots_adjust(left=0.06, right=0.82, top=0.88, bottom=0.12, wspace=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
