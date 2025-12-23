# kaggle_.py

Kaggle 项目合集仓库（monorepo）。每个比赛/实验放在一个独立子目录中，目录结构尽量统一，方便后续继续添加其他项目。

## Projects

- `playground-series-s5e12/`：Playground Series S5E12（二分类：`diagnosed_diabetes`）基线与一些特征工程尝试。

## Directory layout (per project)

- `data/`：数据说明与下载方式（可选，不上传原始数据）
- `notebooks/`：Kaggle kernel / notebook 入口脚本等
- `src/`：可复用的训练/特征工程代码
- `reports/weekly_report.pdf`：阶段性汇报

## Quick start (S5E12)

1) 下载数据（不进 git）

- 放到：`playground-series-s5e12/data/raw/`
- 需要包含：`train.csv`、`test.csv`（可选：`sample_submission.csv`）

如果你有 Kaggle CLI（推荐）：

```bash
# 先配置 kaggle.json 后再执行
mkdir -p playground-series-s5e12/data/raw
cd playground-series-s5e12/data/raw
kaggle competitions download -c playground-series-s5e12
unzip -o playground-series-s5e12.zip
```

2) 运行一个 baseline（示例）

```bash
python playground-series-s5e12/src/baseline_hgb.py
```

默认输出会写到 `playground-series-s5e12/reports/`（CSV 已被 `.gitignore` 忽略，避免把提交文件推上 GitHub）。
