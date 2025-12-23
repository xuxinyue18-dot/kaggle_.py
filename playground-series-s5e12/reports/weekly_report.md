# 周报｜Playground Series S5E12

日期：2025-12-23  
仓库：`xuxinyue18-dot/kaggle_.py`（项目合集）

## 1. 本周目标

- 将 S5E12 项目整理为统一目录结构，便于后续继续往同一仓库追加其他 Kaggle 项目
- 提供可复用的 baseline 脚本与数据下载说明，避免上传原始数据

## 2. 本周完成

- 统一目录结构：`data/`、`notebooks/`、`src/`、`reports/`
- 添加数据下载说明：`playground-series-s5e12/data/README.md`
- 迁移并整理训练脚本（输出统一写入 `reports/`）
  - `src/baseline_hgb.py`：HGB baseline + 简单特征工程
  - `src/objective_weight_model.py`：entropy + CRITIC 融合的 objective_score 特征
  - `src/cat_lgbm_models.py`：CatBoost / LightGBM baseline（需在有可用 wheels 的环境运行）
- 增加 `.gitignore`：默认忽略 `*.csv/*.zip` 等数据与提交文件，避免把大文件推到 GitHub
- 保留 Kaggle kernel 入口脚本（`.py` + metadata）：`playground-series-s5e12/notebooks/`

## 3. 运行方式

1) 下载数据到 `playground-series-s5e12/data/raw/`（不进 git）

- 需要：`train.csv`、`test.csv`（可选：`sample_submission.csv`）
- 参考：`playground-series-s5e12/data/README.md`

2) 运行脚本（示例）

```bash
python playground-series-s5e12/src/baseline_hgb.py
python playground-series-s5e12/src/objective_weight_model.py
python playground-series-s5e12/src/cat_lgbm_models.py --model catboost
```

## 4. 实验与结果（待补充）

说明：当前仓库以“结构与可复现”为主；具体 CV 分数需要在有数据文件的环境里运行后再补齐。

- HGB baseline：CV AUC = TBD
- Objective score（entropy+CRITIC）：CV AUC = TBD
- CatBoost / LightGBM：CV AUC = TBD

## 5. 风险与注意事项

- Termux 环境可能无法安装 `catboost/lightgbm` 的预编译 wheel，建议在 Kaggle Notebook / Colab / PC 上运行对应脚本
- GitHub 对单文件大小有建议上限（50MB），本仓库默认不上传原始数据与提交文件

## 6. 下周计划

- 在 Kaggle Notebook 跑通三套 baseline，记录 CV 与 leaderboard 分数
- 将结果写回本周报的“实验与结果”部分
