# European Electricity Price & Generation 2024-25 (PL pumped storage)

本项目是本仓库（Kaggle 项目合集）中的一个子项目。

## Goal（面向“交易研究助理”交付物）

把“电力市场研究助理”日常要做的交付物做成可复现的一键流水线：

- 周报：价格水平/波动/异常（尖峰、负价）+ 抽蓄行为与结构驱动
- 预测：下周均价方向（Up/Down/Flat）+ 风险区间（P10/P50/P90）
- 复盘：滚动回测逐周复盘（命中/未命中、偏差原因提示、下周关注点）

## Layout

- `data/`：数据下载说明（默认不提交原始数据）
- `notebooks/`：Kaggle kernel / notebook 入口脚本
- `src/`：训练与特征工程代码
- `reports/`：周报与实验记录（建议优先提交 Markdown）

附加输出目录：

- `backtest_results/`：滚动回测结果（图 + 逐周预测 CSV）

## What this project does

聚焦 `PL`（波兰）市场的抽水蓄能行为，用“交易周报”的口径自动产出：

- `reports/weekly_report.pdf`：两页周报（本周 vs 上周、结构/抽蓄、回测摘要）
- `reports/weekly_review.md`：自动复盘（结论句 + 可能驱动 + 风险提示 + 下周关注点）
- `backtest_results/`：滚动回测（方向三分类 + 周均价分位区间）

## Data（为什么选 PL）

数据来自 Kaggle：`patrikpetovsky/european-electricity-price-and-generation-2024-25`，核心字段包括：

- `price`：日前市场电价（€/MWh）
- 多种发电类型出力（MW）
- 抽水蓄能：发电侧 + 用电/吸纳侧（并非所有国家都同时具备）

本项目选择 `PL` 的原因：在这份数据里，`PL` 基本是唯一一个同时具备以下两列且覆盖完整的市场，因此可以严谨构建“抽蓄净出力/套利行为”的分析：

- `_hydro_pumped_storage_actual_aggregated_`（抽蓄发电）
- `_hydro_pumped_storage_actual_consumption_`（抽蓄用电/吸纳）

## Methods（核心方法与口径）

### 1) 时间口径（周报口径）

- 将 `datetime` 解析为 UTC 后转换到市场时区 `Europe/Warsaw`
- 按“本地周一 00:00”为周起点做周频汇总（更贴近交易周报）
- 默认仅用“完整周”（`n_hours>=160`）参与建模/回测/复盘，避免最后一周数据不满导致指标失真

### 2) 风险定义（Risk Flags）

- `neg_flag`：`price < 0`（负价小时）
- `spike_flag`：`price > rolling_P99(price, window=8 weeks)`
  - 采用滚动阈值且 **shift=1**（只用历史，不用未来），避免信息泄漏

### 3) 抽蓄行为指标（Pumped-storage Features）

- `pumped_net = pumped_gen - pumped_consume`
- `pumped_mode`：`pump / generate / idle / both`

用于回答交易问题：抽蓄是否在“低价吸纳、高价发电”（套利直觉），以及是否出现“高价吸纳/低价发电”等反常行为提示。

### 4) 结构与压力（Generation Mix + Ramps）

用 aggregated 口径（对 PL 更完整的一套）构建周频结构特征：

- `total_gen_mw`、`renewable_share`、`gas_share`、`coal_share`、`nuclear_share`
- `wind_ramp/solar_ramp/renewable_ramp`（用 diff(abs) 近似“爬坡/波动压力”）

## Analysis pipeline（完整流程路线）

入口脚本：`generate_report.py` → `src/euro_electricity_2024_25/pipeline.py`

1. 读取数据 → 过滤 `country=PL`
2. 构造小时特征（风险标记、抽蓄净出力、结构占比、爬坡压力）
3. 聚合为周频表（每周一行）：价格统计 + 风险小时数 + 抽蓄行为 + 结构/压力
4. 构造监督学习目标：`next_week_price_mean`（下周均价）与 `delta`
5. 模型与回测（Walk-forward 滚动回测，每周滚动一次）
   - 方向：三分类（Up/Down/Flat），基于阈值 `θ=0.25*std(delta_history)`
   - 区间：分位回归输出 P10/P50/P90（用于风险区间）
6. 自动生成产物
   - `reports/weekly_report.pdf`（2页）
   - `reports/weekly_review.md`（逐周复盘：结论句 + 归因提示 + 风险提示 + 下周关注点）
   - `backtest_results/*`（回测图 + 逐周预测 CSV）

## Outputs（生成哪些文件）

运行 `python generate_report.py` 后主要输出：

- `reports/weekly_report.pdf`
- `reports/weekly_review.md`
- `reports/weekly_features_pl.csv`（周频特征表，便于你继续扩展）
- `backtest_results/walk_forward_predictions.csv`（每周预测、覆盖与方向结果）
- `backtest_results/interval_backtest.png`
- `backtest_results/direction_confusion.png`
- `backtest_results/coverage_over_time.png`
- `backtest_results/pred_error_over_time.png`

## Expected results（预期你能讲出来什么）

这套项目的“预期结果”不是追求某个 Kaggle 分数，而是面试可讲、能落地：

- 在周报里清晰展示：本周 vs 上周价格、波动区间、负价/尖峰小时、抽蓄净出力与行为切换
- 在复盘里能解释：哪些周区间未覆盖/方向未命中，并给出“更像交易员”的归因提示与下周关注点
- 给出“风险区间”而不是只给点预测，体现风险管理思维（coverage/误差曲线是硬证据）

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

更多参数：

```bash
python generate_report.py --train-window-weeks 52 --review-weeks 8 --spike-window-weeks 8 --min-week-hours 160
```
