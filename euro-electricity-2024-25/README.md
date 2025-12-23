# European Electricity Price & Generation 2024-25 (PL pumped storage)

本项目是本仓库（Kaggle 项目合集）中的一个子项目。

## Layout

- `data/`：数据下载说明（默认不提交原始数据）
- `notebooks/`：Kaggle kernel / notebook 入口脚本
- `src/`：训练与特征工程代码
- `reports/`：周报与实验记录（建议优先提交 Markdown）

## What this project does

聚焦 `PL`（波兰）市场的抽水蓄能行为，用“交易周报”的口径自动产出：

- `reports/weekly_report.pdf`：两页周报（本周 vs 上周、结构/抽蓄、回测摘要）
- `reports/weekly_review.md`：自动复盘（结论句 + 可能驱动 + 风险提示 + 下周关注点）
- `backtest_results/`：滚动回测（方向三分类 + 周均价分位区间）

## Quick start

1) 下载数据（Kaggle dataset）

数据集：`patrikpetovsky/european-electricity-price-and-generation-2024-25`

```bash
mkdir -p kaggle_datasets/european-electricity-price-and-generation-2024-25
kaggle datasets download -d patrikpetovsky/european-electricity-price-and-generation-2024-25 -p kaggle_datasets/european-electricity-price-and-generation-2024-25 --unzip
```

2) 放置数据（不提交到 git）

把 `entsoe_data_2024_2025.csv` 放到 `euro-electricity-2024-25/data/raw/`（默认被忽略）。

3) 一键生成周报/回测/复盘

```bash
cd euro-electricity-2024-25
python generate_report.py
```

常用参数：

```bash
python generate_report.py --train-window-weeks 26 --review-weeks 8
```
