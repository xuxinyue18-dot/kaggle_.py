# PL 抽蓄套利 / 风险提示周报（自动生成）

- 市场：PL
- 时区：Europe/Warsaw
- 覆盖范围：2024-01-01 00:00:00+01:00 → 2025-06-30 00:00:00+02:00
- Spike 定义：过去 8 周小时价 99 分位阈值（滚动、仅用历史，shift=1）

## 最新周（week_start_local=2025-06-23 00:00:00+02:00）概览

- 均价：78.59 €/MWh
- 环比（均价）：+0.00 €/MWh
- 波动区间（P10–P90）：0.12 → 127.25 €/MWh
- 负价小时：9
- Spike 小时：4
- 抽蓄发电小时：68
- 抽蓄吸纳小时：153
- 抽蓄净出力均值：-46.33 MW

## 套利直觉验证（条件价格）

- 吸纳（pump）时均价：50.60 €/MWh
- 发电（generate）时均价：163.77 €/MWh
- 简单价差（generate - pump）：113.17 €/MWh

## 风险提示（最新周）

- 高价吸纳小时（pump & price>P90）：0
- 低价发电小时（generate & price<P10）：0

## 行为与价格相关性（全样本）

- corr(pumped_net, price)：0.395
- corr(pumped_consume, price)：-0.290
- corr(pumped_gen, price)：0.293
