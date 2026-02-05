#!/usr/bin/env bash
set -euo pipefail

# 进入脚本目录并定位工程根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SUITE_ROOT}/.." && pwd)"

# 保证 Python 可以找到 experiments_scheduling_suite 包
export PYTHONPATH="${REPO_ROOT}"

BASE_CFG="${SUITE_ROOT}/configs/base.yaml"
DATASET_CFG="${SUITE_ROOT}/configs/datasets/synthetic.yaml"

# 1) 生成/准备原始数据（只需要一次）
python "${SUITE_ROOT}/scripts/00_generate_data.py" \
  --config "${BASE_CFG}" \
  --dataset "${DATASET_CFG}"

# 2) 定义全量组合（所有模型 / 所有插补 / 所有调度）
# 缺失策略包含参数扫点（保证敏感性曲线有多点）
MISSINGNESS_CFGS=(
  mcar_p0.05.yaml
  mcar_p0.1.yaml
  mcar.yaml
  mcar_p0.3.yaml
  mcar_p0.4.yaml
  block_b5.yaml
  block.yaml
  block_b20.yaml
  block_b40.yaml
  duty_cycle_p20_on2.yaml
  duty_cycle.yaml
  duty_cycle_p20_on10.yaml
  round_robin_k1_minon3.yaml
  round_robin.yaml
  round_robin_k3_minon3.yaml
  info_priority_k1_minon3.yaml
  info_priority.yaml
  info_priority_k3_minon3.yaml
)
IMPUTATION_LIST=(none_maskaware linear spline kalman gp)
MODEL_LIST=(lstm transformer informer tcn xgboost naive mlp)
HORIZON_LIST=(1 2 3)
IMPUTER_NAME_LIST=(maskaware linear spline kalman gp)

# 3) 逐组合运行：准备数据 -> 预训练可视化 -> 训练 -> 评估 -> 预测图 -> 总结图
for miss_cfg in "${MISSINGNESS_CFGS[@]}"; do
  MISS_CFG="${SUITE_ROOT}/configs/missingness/${miss_cfg}"
  for imp in "${IMPUTATION_LIST[@]}"; do
    IMP_CFG="${SUITE_ROOT}/configs/imputation/${imp}.yaml"

    # 用 Python 生成与系统一致的 RUN_ID
    RUN_ID=$(python - <<PY
from pathlib import Path
from experiments_scheduling_suite.src.utils.io import load_yaml, build_run_id
base = load_yaml(Path("${BASE_CFG}"))
dataset = load_yaml(Path("${DATASET_CFG}")).get("dataset", {})
missing = load_yaml(Path("${MISS_CFG}")).get("missingness", {})
impute = load_yaml(Path("${IMP_CFG}")).get("imputation", {})
print(build_run_id(dataset, base, missing, impute))
PY
)

    echo "=== RUN_ID: ${RUN_ID} (missingness=${miss_cfg}, imputation=${imp}) ==="

    # A) 准备数据（缺失 + 插补 + 窗口化）
    python "${SUITE_ROOT}/scripts/01_prepare_dataset.py" \
      --config "${BASE_CFG}" \
      --dataset "${DATASET_CFG}" \
      --missingness "${MISS_CFG}" \
      --imputation "${IMP_CFG}" \
      --run-id "${RUN_ID}"

    # B) 预训练可视化
    python "${SUITE_ROOT}/scripts/02_visualize_pretrain.py" \
      --run-id "${RUN_ID}"

    # C) 训练所有模型
    python "${SUITE_ROOT}/scripts/03_train_models.py" \
      --run-id "${RUN_ID}" \
      --models "$(IFS=,; echo "${MODEL_LIST[*]}")"

    # D) 评估
    python "${SUITE_ROOT}/scripts/04_evaluate.py" \
      --run-id "${RUN_ID}"

    # E) 预测图
    python "${SUITE_ROOT}/scripts/05_plot_predictions.py" \
      --run-id "${RUN_ID}"

    # F) 总结大图
    python "${SUITE_ROOT}/scripts/06_plot_summary.py" \
      --run-id "${RUN_ID}"
  done
done

# 4) 聚合与后期分析
AGG_DIR="${SUITE_ROOT}/reports/_aggregate"
METRICS_LONG="${AGG_DIR}/tables/metrics_long.csv"
FEATURES="${AGG_DIR}/tables/missingness_features.csv"

python "${SUITE_ROOT}/scripts/08_collect_results.py" --with_missingness_features

python "${SUITE_ROOT}/scripts/09_plot_cross_strategy.py" \
  --in "${METRICS_LONG}" --out "${AGG_DIR}/figures" --mode heatmap --metric rmse
python "${SUITE_ROOT}/scripts/09_plot_cross_strategy.py" \
  --in "${METRICS_LONG}" --out "${AGG_DIR}/figures" --mode boxplot --metric rmse
python "${SUITE_ROOT}/scripts/09_plot_cross_strategy.py" \
  --in "${METRICS_LONG}" --out "${AGG_DIR}/figures" --mode sensitivity --metric rmse
python "${SUITE_ROOT}/scripts/09_plot_cross_strategy.py" \
  --in "${METRICS_LONG}" --out "${AGG_DIR}/figures" --mode ranking --metric rmse
python "${SUITE_ROOT}/scripts/09_plot_cross_strategy.py" \
  --in "${METRICS_LONG}" --out "${AGG_DIR}/figures" --mode scatter --metric rmse --features "${FEATURES}"

python "${SUITE_ROOT}/scripts/10_event_based_eval.py" --mode event_eval
python "${SUITE_ROOT}/scripts/10_event_based_eval.py" --mode imputation_error

# 5) 显著性检验（按模型/插值/步长）
for model in "${MODEL_LIST[@]}"; do
  for h in "${HORIZON_LIST[@]}"; do
    for imputer in "${IMPUTER_NAME_LIST[@]}"; do
      python "${SUITE_ROOT}/scripts/11_significance_tests.py" \
        --model "${model}" --horizon "${h}" --imputer "${imputer}"
    done
  done
done

# 6) 汇总报告
python "${SUITE_ROOT}/scripts/12_make_posthoc_report.py"

# 7) 跨调度/插值策略预测对比图
RUN_PREFIX=$(python - <<PY
from pathlib import Path
from experiments_scheduling_suite.src.utils.io import load_yaml
base = load_yaml(Path("${BASE_CFG}"))
dataset = load_yaml(Path("${DATASET_CFG}")).get("dataset", {})
freq = str(base.get("run", {}).get("base_freq", "1S")).lower().replace("/", "")
name = dataset.get("name", "dataset")
print(f"{name}_{freq}")
PY
)

python "${SUITE_ROOT}/scripts/13_plot_strategy_predictions.py" \
  --run-prefix "${RUN_PREFIX}" \
  --horizons 1,2,3

echo "All experiments completed."
