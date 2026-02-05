# 实验套件说明（Wind-blown Snow / 调度 / 插值 / 多模型预测）

本目录是独立实验套件，所有脚本均以 `experiments_scheduling_suite/` 为根目录运行。

## 快速开始（单次实验）

1) 生成/准备原始数据  
```
PYTHONPATH=/home/zhangzhuyu/_code/microclimate_demo \
python experiments_scheduling_suite/scripts/00_generate_data.py \
  --config experiments_scheduling_suite/configs/base.yaml \
  --dataset experiments_scheduling_suite/configs/datasets/synthetic.yaml
```

2) 准备数据集（选择调度 + 插值）  
```
PYTHONPATH=/home/zhangzhuyu/_code/microclimate_demo \
python experiments_scheduling_suite/scripts/01_prepare_dataset.py \
  --config experiments_scheduling_suite/configs/base.yaml \
  --dataset experiments_scheduling_suite/configs/datasets/synthetic.yaml \
  --missingness experiments_scheduling_suite/configs/missingness/mcar.yaml \
  --imputation experiments_scheduling_suite/configs/imputation/linear.yaml
```

3) 训练模型  
```
PYTHONPATH=/home/zhangzhuyu/_code/microclimate_demo \
python experiments_scheduling_suite/scripts/03_train_models.py --run-id <RUN_ID>
```

4) 评估与画图  
```
PYTHONPATH=/home/zhangzhuyu/_code/microclimate_demo \
python experiments_scheduling_suite/scripts/04_evaluate.py --run-id <RUN_ID>

PYTHONPATH=/home/zhangzhuyu/_code/microclimate_demo \
python experiments_scheduling_suite/scripts/05_plot_predictions.py --run-id <RUN_ID>

PYTHONPATH=/home/zhangzhuyu/_code/microclimate_demo \
python experiments_scheduling_suite/scripts/06_plot_summary.py --run-id <RUN_ID>
```

## 全量实验（所有调度 × 插值 × 模型）

直接运行一键脚本（包含后期聚合与分析）：  
```
bash experiments_scheduling_suite/scripts/run_full_sweep.sh
```

该脚本会完成：
- 00 生成数据  
- 01→06 全组合训练、评估、绘图  
- 08→12 聚合、横向对比、事件评估、显著性检验、报告  
- 13 跨调度/插值策略预测对比图

## 跨策略预测对比图

用于比较同一模型在不同调度/插值策略下的预测曲线：  
```
PYTHONPATH=/home/zhangzhuyu/_code/microclimate_demo \
python experiments_scheduling_suite/scripts/13_plot_strategy_predictions.py \
  --run-prefix synthetic_windblown_1s --horizons 1,2,3
```

说明：
- 颜色 = 插值算法  
- 线型 = 调度算法  
- 真实值为黑色实线（图例中 “Series: true”）

## 主要输出

- `experiments_scheduling_suite/reports/<RUN_ID>/preds/` 预测 CSV  
- `experiments_scheduling_suite/reports/<RUN_ID>/tables/` 指标表  
- `experiments_scheduling_suite/reports/<RUN_ID>/figures/` 预测图 / 总结图  
- `experiments_scheduling_suite/reports/_aggregate/` 聚合分析结果与图表  

## 注意事项

- 插补后若仍存在 NaN，会在 `01_prepare_dataset.py` 末尾自动兜底填充（避免 MLP 等模型报错）。  
- spline 插值已做过冲裁剪，避免极端值影响训练。  
- `base_freq` 使用小写（如 `1s`）避免 pandas 警告。  

如果需要断点续跑/跳过已完成 run，可继续扩展 `run_full_sweep.sh`。  

---

# 附录：插值与调度算法清单

下面按配置文件名列出当前可用的插值与调度算法，并附简要说明。  

## 插值算法（`experiments_scheduling_suite/configs/imputation/`）

- `none_maskaware.yaml`（maskaware）  
  不做数值插补，只添加 `is_missing_*` 与 `tsls_*` 特征，并将缺失值用 `fill_value` 填充（默认 0）。适合希望模型“感知缺失模式”的场景。  

- `linear.yaml`（linear）  
  逐列线性插值，可设 `limit_direction`。速度快、稳健，是常用基线。  

- `spline.yaml`（spline）  
  逐列样条插值（order 可调）。对曲线平滑更友好，但易过冲，已在代码中加入“裁剪到观测范围”的保护。  

- `kalman.yaml`（kalman）  
  逐列一维 Kalman 平滑插补。能抑制噪声并补齐缺失，但可能过度平滑。  

- `gp.yaml`（gp）  
  逐列高斯过程插补（RBF + WhiteKernel）。适合小规模数据，代价较高。  

## 调度/缺失模拟（`experiments_scheduling_suite/configs/missingness/`）

- `mcar.yaml`（mcar）  
  随机独立缺失（MCAR），按 `p_missing` 控制缺失比例。  

- `block.yaml`（block）  
  块状缺失，按 `n_blocks` 与 `min_len_steps/max_len_steps` 生成连续缺失段。  

- `duty_cycle.yaml`（duty_cycle）  
  周期性采样/失采，按 `period_steps` 与 `on_steps` 控制“开/关”节奏。  

- `round_robin.yaml`（round_robin）  
  多变量轮询采样，按 `budget_k` 和 `min_on_steps` 控制每次观测的变量数与最短开启时长。  

- `info_priority.yaml`（info_priority）  
  信息优先级策略：根据与目标列的相关性选择 top-k 变量观测，可设置 `lag_steps` 与 `refresh_steps`。  
