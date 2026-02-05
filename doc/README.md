# 实验文档（详细版）

本文件面向研究复现与学术交流，完整描述实验背景、设计、方法与结果，并在合适处引用已产出的图像结果。  
所有路径均以 `experiments_scheduling_suite/` 为根目录。

---

## 1. 实验背景与问题定义

极端环境下的气象与雪物理观测常受到传感器调度约束与不规则缺测的影响。  
本实验旨在系统评估：**不同调度/缺失策略与不同插值算法的组合**，在多种预测模型下对短时序列预测性能的影响。  
研究问题可表述为：

1) 缺失机制（调度策略）如何影响模型性能与稳定性？  
2) 插值策略在不同缺失模式下的表现差异是否显著？  
3) 不同模型（深度/传统/基线）对缺失结构的敏感度是否一致？  
4) 在“事件段”与“非事件段”上的误差分布是否具有结构性差异？

本实验采用合成的 wind-blown snow 数据作为标准化测试基准，提供可控的频率与缺失机制以支持系统性比较。

---

## 2. 数据与合成流程

实验数据由脚本 [`scripts/00_generate_data.py`](../scripts/00_generate_data.py) 生成（默认 synthetic wind-blown snow），包含风速、温湿度、气压、雪粒径/速度分箱通量等多变量观测。  
数据被重采样至 `base_freq`（默认 1s），并进行特征工程：

- 风向角转为 `sin/cos` 两个分量  
- 稳定度标记转为 one-hot  

### 2.1 合成数据生成规则（关键机制）

合成数据生成器位于：[`src/data/generator/synthetic_windblown.py`](../src/data/generator/synthetic_windblown.py)，核心规则如下：

- **时间网格**：固定步长（默认 1s），生成规则时间序列。  
- **动态平滑**：关键气象变量采用带步长缩放的 AR(1) 过程，保证短时连续性。  
- **日变化**：温度/风速/辐射包含正弦日周期项。  
- **风暴状态**：二态马尔可夫过程（storm_state）控制风速抬升、降水增强、辐射衰减。  
- **输运阈值**：使用经验阈值 `Ut` 与风速计算超越程度 `ratio`，决定雪输运强度。  
- **供给反馈**：`snow_supply` 随降水增加、随输运消耗；影响最终雪通量。  
- **质量/数目通量**：先生成总雪质量通量，再按粒径/速度分箱用 Dirichlet 分配，确保守恒。  
- **观测质量字段**：`quality_flag`、`data_source`、`missing_reason` 仅作为元信息；在数值化阶段会被丢弃。  

> 该设计确保序列既平滑可学，又包含风暴事件与输运阈值带来的非线性变化。

### 2.2 可用变量与处理约束

输入 CSV 至少需要包含：  
`timestamp, air_temperature_c, relative_humidity, air_pressure_pa, wind_speed_ms, wind_direction_deg, solar_radiation_wm2, snow_mass_flux_kg_m2_s, snow_number_flux_m2_s`  

注意事项：  
- 所有**非数值列**会在 `_select_numeric` 阶段（见 [`scripts/01_prepare_dataset.py`](../scripts/01_prepare_dataset.py)）被转换/剔除。  
- 若某列全部为 NaN，会被直接删除，避免插补后仍为 NaN。  
- 目标列默认 **始终可观测**（由 missingness 配置 `target_always_observed` 控制）。  

示例的缺失热图可查看：  
[`../reports/synthetic_windblown_1s_mcar_p0.2_linear/figures/pretrain/missingness_heatmap.png`](../reports/synthetic_windblown_1s_mcar_p0.2_linear/figures/pretrain/missingness_heatmap.png)

![Missingness heatmap](../reports/synthetic_windblown_1s_mcar_p0.2_linear/figures/pretrain/missingness_heatmap.png)

> 图中每个像素表示“时间×变量”的观测状态（黑=观测，白=缺失）。  
> 当时间点过多时会自动按 `max_time_points` 下采样，并按列分面展示以减少标签拥挤。

---

## 3. 实验设计

### 3.1 全量因子设计

本实验采用**全量组合**设计：

- 调度/缺失策略（Missingness）
- 插值算法（Imputation）
- 预测模型（Model）
- 预测步长（Horizon）

每个组合产生一个 `run_id`，其结构为：  
`{dataset}_{freq}_{missingness}{params}_{imputer}`

例如：  
`synthetic_windblown_1s_mcar_p0.2_linear`

### 3.2 数据处理流程

1) 缺失模拟（按策略生成 mask）  
2) 插值（或 mask-aware 衍生特征）  
3) 归一化（仅用训练集统计量）  
4) 窗口化（lookback + horizons）  

### 3.3 训练与评估

训练脚本：[`scripts/03_train_models.py`](../scripts/03_train_models.py)  
评估脚本：[`scripts/04_evaluate.py`](../scripts/04_evaluate.py)  

指标包括：
- RMSE  
- MAE  
- MAPE  

此外还有：
- 事件段 vs 非事件段误差对比  
- 缺失结构 vs 误差散点相关  
- 排名一致性与敏感性分析  

---

## 4. 调度/缺失策略（Missingness）

### 4.1 调度/缺失策略实现细节

实现路径：[`src/missingness/`](../src/missingness/)。记 x_t^j 为第 j 个传感器在时间 t 的观测值，
定义观测指示变量 m_t^j ∈ {0,1}（1=观测，0=缺失）。默认设置 m_t^target=1（`target_always_observed=true`），
即目标列始终可观测。

- **MCAR（完全随机缺失）**（[`mcar.py`](../src/missingness/mcar.py)）：
  该机制对应 Missing Completely At Random 假设，即对每个传感器 j、每个时间点 t 的观测指示变量 m_t^j 独立抽样，
  令 m_t^j ~ Bernoulli(1-p_j)，并假设缺失与时间、变量身份及真实取值无关。实现上由 `p_missing` 指定全局缺失率，
  如配置 `per_variable` 则为不同变量赋予不同 p_j，使缺失率可“整体一致”或“按变量异质”地控制；由于在时间与变量维度上
  均为独立同分布抽样，该机制不引入结构性模式，常作为无结构缺失的基准模型。

- **Block（连续块缺失）**（[`block.py`](../src/missingness/block.py)）：
  该机制通过显式构造连续缺失区间来模拟设备故障或通信中断：对每个变量生成 n_blocks 个缺失块，并在
  [min_len_steps, max_len_steps] 内均匀采样缺失长度，从而形成具有长缺口与强时间自相关的缺失结构；当 `per_variable=false` 时，
  同一缺失块对所有变量同步生效，使缺失在变量维度上呈现显著相关性，与 MCAR 的独立性假设形成对照。

- **Duty Cycle（占空比采样）**（[`duty_cycle.py`](../src/missingness/duty_cycle.py)）：
  该机制刻画周期性开关采样（硬件占空比模型），对每个变量设定周期 P 与开启时长 O，使 m_t^j 在每个周期内仅有 O 步为观测，
  同时允许随机相位以避免严格同相；可选 `budget_k` 进一步限制同一时刻最多开启的变量数，从而在占空比约束之外叠加观测预算约束，
  最终形成具有明显周期性的缺失结构。

- **Round Robin（轮询调度）**（[`round_robin.py`](../src/missingness/round_robin.py)）：
  该策略在预算约束下进行轮询调度，以 `min_on_steps` 作为调度块长度，并在每个块内选择 `budget_k` 个传感器按 `sensor_order` 轮换开启，
  其余传感器关闭；该设计保证长期覆盖的公平性，同时在任意时刻只保留预算内变量，使缺失在变量间表现为结构性互斥。

- **Info Priority（信息优先级调度）**（[`info_priority.py`](../src/missingness/info_priority.py)）：
  该机制是数据驱动的启发式调度，先在训练集上计算每个变量的相关性权重 w_j = max_lag |corr(x_{t-lag}^j, y_t)|（`lag_steps` 定义滞后集合），
  再在每个调度块中选取权重最高的 top-k 变量作为观测对象；可选 `refresh_steps` 周期性更新权重以适配时变相关性。
  由于权重仅由训练集估计，因此在引入“信息优先级”采样的同时尽量避免信息泄露。

统一约束：
- 缺失 mask 仅作用于输入传感器；目标列是否缺失由 `target_always_observed` 控制。
- 缺失 mask 在插补前应用，确保插补算法在相同缺失条件下比较。

示例：不同 missingness 家族的敏感性曲线  
[`../reports/_aggregate/figures/sensitivity_mcar_H1.png`](../reports/_aggregate/figures/sensitivity_mcar_H1.png)  
[`../reports/_aggregate/figures/sensitivity_block_H1.png`](../reports/_aggregate/figures/sensitivity_block_H1.png)  
[`../reports/_aggregate/figures/sensitivity_duty_cycle_H1.png`](../reports/_aggregate/figures/sensitivity_duty_cycle_H1.png)

![Sensitivity mcar](../reports/_aggregate/figures/sensitivity_mcar_H1.png)
![Sensitivity block](../reports/_aggregate/figures/sensitivity_block_H1.png)
![Sensitivity duty_cycle](../reports/_aggregate/figures/sensitivity_duty_cycle_H1.png)

> 这些曲线展示在同一 missingness 家族内，参数变化对误差的影响趋势。

---

## 5. 插值算法（Imputation）

### 5.1 插值算法实现细节

实现路径：[`src/imputation/`](../src/imputation/)。设观测序列为 x_t^j，缺失位置由 m_t^j 标记，插值目标为估计缺失值 x_t^j。

- **Mask-aware（缺失感知特征）**（[`maskaware_features.py`](../src/imputation/maskaware_features.py)）：
  该策略不直接估计缺失值，而是把缺失模式显式编码为输入特征，从而将“插补”转化为“缺失建模”。
  具体实现中保留原始缺失位置，并为每个变量加入 is_missing（缺失指示）与 time-since-last-seen, TSLS（距离上次观测的时间间隔）等特征，
  使模型能够在训练和预测时利用缺失结构本身携带的信息，这在缺失机制与目标相关时尤其重要。

- **Linear（线性插值）**（[`linear.py`](../src/imputation/linear.py)）：
  对每个变量沿时间轴做分段线性插值，并在序列边界采用前向/后向填充以避免残留缺失。
  该方法等价于在缺失区间内假设局部线性演化，偏差小、实现简单，是最常用的基础插补基线。

- **Spline（三次样条插值）**（[`spline.py`](../src/imputation/spline.py)）：
  采用三次样条（order=3, cubic spline）恢复更高阶的平滑连续性，在观测足够密集时可拟合更细的曲率变化。
  当有效样本不足时退化为线性插值以保证数值稳定，且最终结果裁剪到观测值范围以抑制过冲，从而在平滑性与稳健性之间取得折中。

- **Kalman（卡尔曼滤波/平滑）**（[`kalman.py`](../src/imputation/kalman.py)）：
  逐变量使用一维局部水平状态空间模型进行 Kalman 滤波与平滑，可视为在最小均方估计（MMSE）意义下对噪声与动态变化的权衡。
  该方法对平稳或弱趋势序列的缺失补全较为稳健，同时能在噪声较大时提供一定的去噪效果。

- **GP（高斯过程插值）**（[`gp.py`](../src/imputation/gp.py)）：
  使用高斯过程回归（Gaussian Process Regression）对每个变量进行非参数插补，核函数为 RBF + WhiteKernel，
  可以表达平滑且非线性的变化趋势，但计算开销较高；当可用样本过少时会回退到线性插值以避免不稳定。

> 注意：在插补前，所有非数值列会被剔除；全为 NaN 的列也会被直接丢弃。

示例：插值误差评估（如有生成）可在：  
[`../reports/_aggregate/tables/imputation_error_long.csv`](../reports/_aggregate/tables/imputation_error_long.csv)

---

## 6. 预测模型（Model）

### 6.1 任务定义与特征构建

本实验采用多步监督学习形式：给定长度为 `lookback` 的历史窗口，预测未来多步目标值。
默认目标变量为 `wind_speed_ms`（可在 [`configs/base.yaml`](../configs/base.yaml) 的 `run.target` 中修改）。

- **输入特征**：重采样后的所有数值传感器变量（包含目标变量的历史轨迹），见 [`scripts/01_prepare_dataset.py`](../scripts/01_prepare_dataset.py)。
  - 可选特征工程：风向 `sin/cos` 分量与稳定度 one-hot（`features.use_wind_dir_sincos` / `features.onehot_stability_flag`）。
  - 若使用 Mask-aware 插补，会额外加入 `is_missing_*` 与 `tsls_*` 特征。
- **标准化**：仅使用训练集统计量对输入特征进行标准化（`normalize.method=standard`）。
- **输出形式**：直接多步预测，输出向量对应 `horizons=[1,2,3]`。

### 6.2 模型家族与结构简介（全称与功能）

- **Naive Persistence Baseline（持久性基线）**：
  假设短期内状态不变，预测 ŷ(t+h)=y(t)。用于衡量“不可忽略的最低基线”。
- **MLP (Multi-Layer Perceptron)**：
  将时间窗口展平为向量后进行多输出回归，适合作为弱非线性基线。
- **TCN (Temporal Convolutional Network)**：
  因果膨胀卷积 + 残差结构，能够在固定感受野内捕捉长依赖。
- **LSTM (Long Short-Term Memory)**：
  循环网络模型，利用门控机制建模长短期依赖；输出取最后时刻隐藏状态。
- **Transformer (Encoder-only)**：
  基于自注意力的序列建模，采用位置编码与多头注意力捕捉全局依赖。
- **Informer (Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting)**：
  本实验采用简化实现，加入下采样（distilling）以减少序列长度并提高效率。
- **XGBoost (Extreme Gradient Boosting)**：
  树模型的梯度提升方法，对每个 horizon 独立拟合回归器，适合强特征基线比较。

### 6.3 参数设置与训练策略

- **模型超参**：来自 [`configs/models/`](../configs/models/)；例如 LSTM 的 `hidden_size/num_layers/dropout`，
  Transformer/Informer 的 `d_model/nhead/num_layers/dim_feedforward` 等。
- **训练超参**：来自 [`configs/base.yaml`](../configs/base.yaml) 的 `training` 部分：
  `epochs=10`、`batch_size=64`、`lr=1e-3`、`weight_decay=0`、默认 CPU。
- **优化与损失**：神经模型使用 Adam + MSELoss；XGBoost/MLP 使用各自库的默认优化策略。
- **可复现性**：统一使用固定随机种子（`run.seed`）。

> 所有模型输出均为多步预测（H=1/2/3），用于跨策略与插补方法的统一对比。

## 7. 结果展示与分析

### 7.A 图表清单与作用（汇总）

| 图表（示例文件名） | 输出路径 | 作用 | 生成脚本 |
| --- | --- | --- | --- |
| `missingness_heatmap.png` | `reports/<run_id>/figures/pretrain/` | 展示缺失掩码的时间×变量结构（黑=观测、白=缺失），用于直观检查缺失模式 | `scripts/02_visualize_pretrain.py` |
| `gap_length_hist.png` | `reports/<run_id>/figures/pretrain/` | 缺失段长度分布，用于判断缺口尺度与长尾程度 | `scripts/02_visualize_pretrain.py` |
| `overlay.png` | `reports/<run_id>/figures/pretrain/` | 原始/遮罩/插补的目标序列叠加，用于快速检查插补是否产生系统偏移 | `scripts/02_visualize_pretrain.py` |
| `feature_distributions.png` | `reports/<run_id>/figures/pretrain/` | 插补后特征分布直方图，用于粗查异常分布或常数列 | `scripts/02_visualize_pretrain.py` |
| `feature_kde_comparison.png` | `reports/<run_id>/figures/pretrain/` | 原始/遮罩/插补 KDE 对比（选取观测充足的变量），用于评估插补是否改变分布形状 | `scripts/02_visualize_pretrain.py` |
| `<model>_pred_h123.png` | `reports/<run_id>/figures/preds/` | 单模型的 H=1/2/3 预测曲线，用于比较真实值与预测随时间的偏差 | `scripts/05_plot_predictions.py` |
| `summary_<run_id>.png` | `reports/<run_id>/figures/` | 汇总图（主图+局部放大+雷达图），用于快速比较多模型整体表现 | `scripts/06_plot_summary.py` |
| `heatmap_<model>_H*.png` | `reports/_aggregate/figures/` | Missingness × Imputation 组合热力图，用于观察策略组合效应 | `scripts/09_plot_cross_strategy.py` |
| `boxplot_H*.png` | `reports/_aggregate/figures/` | 跨策略误差分布箱线图，用于比较稳健性与离群点 | `scripts/09_plot_cross_strategy.py` |
| `sensitivity_<family>_H*.png` | `reports/_aggregate/figures/` | 单一缺失家族的参数敏感性曲线，用于量化参数变化对误差的影响 | `scripts/09_plot_cross_strategy.py` |
| `scatter_*_H*.png` | `reports/_aggregate/figures/` | 缺失结构特征（如最大缺口、共缺失率）与误差的相关性散点 | `scripts/09_plot_cross_strategy.py` |
| `rank_corr_heatmap_H*.png` | `reports/_aggregate/figures/` | 策略组合的模型排名一致性（相关矩阵） | `scripts/09_plot_cross_strategy.py` |
| `event_vs_nonevent_H*.png` | `reports/_aggregate/figures/` | 事件段 vs 非事件段误差对比（柱状） | `scripts/10_event_based_eval.py` |
| `strategy_preds/<model>_h*.png` | `reports/_aggregate/figures/strategy_preds/` | 多策略预测对比曲线（颜色=插值，线型=调度） | `scripts/13_plot_strategy_predictions.py` |
| `posthoc_overview.png` | `reports/_aggregate/figures/` | 汇总拼图，快速浏览关键后处理图 | `scripts/12_make_posthoc_report.py` |


### 7.0 结果简要观察（基于当前生成图）

以下观察基于 [`reports/_aggregate/figures`](../reports/_aggregate/figures) 的默认汇总图（H=1 为例）：

- **整体误差分布**：见 [`../reports/_aggregate/figures/boxplot_H1.png`](../reports/_aggregate/figures/boxplot_H1.png) / `H2/H3`，
  随预测步长增大，中位数整体上升且箱体变宽，说明不确定性随 horizon 增加。
- **事件段 vs 非事件段**：见 [`../reports/_aggregate/figures/event_vs_nonevent_H1.png`](../reports/_aggregate/figures/event_vs_nonevent_H1.png)，
  事件段 RMSE 略高于非事件段，且 H 越大差距越明显。
- **MCAR 敏感性**：见 [`../reports/_aggregate/figures/sensitivity_mcar_H1.png`](../reports/_aggregate/figures/sensitivity_mcar_H1.png)，
  TCN 对缺失率上升更敏感；Naive/XGBoost 相对稳定但在高缺失率出现回升。
- **Block 敏感性**：见 [`../reports/_aggregate/figures/sensitivity_block_H1.png`](../reports/_aggregate/figures/sensitivity_block_H1.png)，
  block 长度增大时误差上界提高，说明长缺口对模型更不利。
- **Duty-cycle 敏感性**：见 [`../reports/_aggregate/figures/sensitivity_duty_cycle_H1.png`](../reports/_aggregate/figures/sensitivity_duty_cycle_H1.png)，
  不同占空比对误差存在非线性影响，部分模型在中等占空比达到更低误差。
- **Round-robin 敏感性**：见 [`../reports/_aggregate/figures/sensitivity_round_robin_H1.png`](../reports/_aggregate/figures/sensitivity_round_robin_H1.png)，
  预算 k 增大时 Informer 误差下降更明显，表明对“观测数量”更敏感。
- **Info-priority**：见 [`../reports/_aggregate/figures/sensitivity_info_priority_H1.png`](../reports/_aggregate/figures/sensitivity_info_priority_H1.png)，
  当前 sweep 仅包含单一参数点（k=3），曲线退化为竖线，建议扩展参数栅格。
- **热力图**：见 [`../reports/_aggregate/figures/heatmap_informer_H1.png`](../reports/_aggregate/figures/heatmap_informer_H1.png) 与
  [`../reports/_aggregate/figures/heatmap_lstm_H1.png`](../reports/_aggregate/figures/heatmap_lstm_H1.png)，
  info_priority_k3_minon3 在多数插补下颜色更冷，表现更稳；
  round_robin 低 k + spline 的组合更容易出现较高误差。
- **排名一致性**：见 [`../reports/_aggregate/figures/rank_corr_heatmap_H1.png`](../reports/_aggregate/figures/rank_corr_heatmap_H1.png)，
  同一策略家族内部相关性高，跨家族（如 block vs round_robin）一致性较低。

> 注意：以上为定性观察，严谨结论需结合统计显著性检验与更长序列。

### 7.1 汇总图（overview）

整体汇总图可见：  
[`../reports/_aggregate/figures/posthoc_overview.png`](../reports/_aggregate/figures/posthoc_overview.png)

![Overview](../reports/_aggregate/figures/posthoc_overview.png)

该图汇总了模型总体表现与缺失结构的影响趋势。

### 7.2 模型热力图（策略 × 插值）

以 Informer 为例（H=1/2/3）：  
[`../reports/_aggregate/figures/heatmap_informer_H1.png`](../reports/_aggregate/figures/heatmap_informer_H1.png)  
[`../reports/_aggregate/figures/heatmap_informer_H2.png`](../reports/_aggregate/figures/heatmap_informer_H2.png)  
[`../reports/_aggregate/figures/heatmap_informer_H3.png`](../reports/_aggregate/figures/heatmap_informer_H3.png)

![Heatmap informer H1](../reports/_aggregate/figures/heatmap_informer_H1.png)

热力图展示 “Missingness × Imputation” 的组合效应。

### 7.3 Boxplot（跨策略误差分布）

[`../reports/_aggregate/figures/boxplot_H1.png`](../reports/_aggregate/figures/boxplot_H1.png)  
[`../reports/_aggregate/figures/boxplot_H2.png`](../reports/_aggregate/figures/boxplot_H2.png)  
[`../reports/_aggregate/figures/boxplot_H3.png`](../reports/_aggregate/figures/boxplot_H3.png)

![Boxplot H1](../reports/_aggregate/figures/boxplot_H1.png)

该图可用于观察不同策略组合的总体误差分布与离群点。

### 7.4 散点相关（缺失结构 vs 误差）

[`../reports/_aggregate/figures/scatter_max_gap_len_H1.png`](../reports/_aggregate/figures/scatter_max_gap_len_H1.png)  
[`../reports/_aggregate/figures/scatter_p95_gap_len_H1.png`](../reports/_aggregate/figures/scatter_p95_gap_len_H1.png)  
[`../reports/_aggregate/figures/scatter_co_missingness_mean_H1.png`](../reports/_aggregate/figures/scatter_co_missingness_mean_H1.png)

![Scatter max gap](../reports/_aggregate/figures/scatter_max_gap_len_H1.png)
![Scatter p95 gap](../reports/_aggregate/figures/scatter_p95_gap_len_H1.png)
![Scatter co-missing](../reports/_aggregate/figures/scatter_co_missingness_mean_H1.png)

该类图用于分析缺失结构特征与预测误差之间的相关性趋势。

### 7.5 排名相关（策略一致性）

[`../reports/_aggregate/figures/rank_corr_heatmap_H1.png`](../reports/_aggregate/figures/rank_corr_heatmap_H1.png)

![Rank corr](../reports/_aggregate/figures/rank_corr_heatmap_H1.png)

展示各策略组合在模型排名上的一致性。

### 7.6 事件段 vs 非事件段

[`../reports/_aggregate/figures/event_vs_nonevent_H1.png`](../reports/_aggregate/figures/event_vs_nonevent_H1.png)

![Event vs non-event](../reports/_aggregate/figures/event_vs_nonevent_H1.png)

该图用于比较事件段与非事件段的误差差异（具体事件定义见脚本参数）。

### 7.7 跨策略预测曲线对比

以 Informer (H=1) 为例：  
[`../reports/_aggregate/figures/strategy_preds/informer_h1.png`](../reports/_aggregate/figures/strategy_preds/informer_h1.png)

![Strategy comparison informer H1](../reports/_aggregate/figures/strategy_preds/informer_h1.png)

颜色代表插值算法，线型代表调度策略，黑色为真实值。

---

## 8. 可复现流程

单次实验（示例）：

相关脚本：[`00_generate_data.py`](../scripts/00_generate_data.py)、[`01_prepare_dataset.py`](../scripts/01_prepare_dataset.py)、[`03_train_models.py`](../scripts/03_train_models.py)
```
PYTHONPATH=/home/zhangzhuyu/_code/microclimate_demo \
python experiments_scheduling_suite/scripts/00_generate_data.py \
  --config experiments_scheduling_suite/configs/base.yaml \
  --dataset experiments_scheduling_suite/configs/datasets/synthetic.yaml

PYTHONPATH=/home/zhangzhuyu/_code/microclimate_demo \
python experiments_scheduling_suite/scripts/01_prepare_dataset.py \
  --config experiments_scheduling_suite/configs/base.yaml \
  --dataset experiments_scheduling_suite/configs/datasets/synthetic.yaml \
  --missingness experiments_scheduling_suite/configs/missingness/mcar.yaml \
  --imputation experiments_scheduling_suite/configs/imputation/linear.yaml

PYTHONPATH=/home/zhangzhuyu/_code/microclimate_demo \
python experiments_scheduling_suite/scripts/03_train_models.py --run-id <RUN_ID>
```

全量实验（建议在 tmux 中运行）：

脚本：[`run_full_sweep.sh`](../scripts/run_full_sweep.sh)
```
bash experiments_scheduling_suite/scripts/run_full_sweep.sh
```

---

## 9. 讨论与局限

1) **合成数据偏差**：合成分布与真实观测仍有差距。  
2) **插值过冲风险**：样条插值在稀疏缺失时可能出现过冲或不稳定。  
3) **短步长高重合**：在 H=1/2/3 的短步长下，持久性基线可能非常强。  
4) **事件定义敏感**：事件段划分依赖阈值定义，影响对比结果。  

---

## 10. 后续扩展建议

- 引入真实站点数据作为外部验证  
- 增加更长的预测 horizon  
- 增加物理约束或多任务损失  
- 进行缺失结构的可解释性分析（如 SHAP、因果扰动）  

---

## 11. 参考输出路径索引（便于检索）

- 预训练可视化：`../reports/<RUN_ID>/figures/pretrain/`  
- 预测输出：`../reports/<RUN_ID>/preds/`  
- 单次总结图：`../reports/<RUN_ID>/figures/summary_<RUN_ID>.png`  
- 聚合分析：[`../reports/_aggregate/figures/`](../reports/_aggregate/figures/)  
