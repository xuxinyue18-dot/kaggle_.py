# Playground Series S5E12

二分类任务：预测 `diagnosed_diabetes`。

## Layout

- `data/`：数据下载说明（不提交原始数据）
- `src/`：训练脚本
- `notebooks/`：Kaggle kernel 入口脚本（`.py` + metadata）
- `reports/`：周报与本地生成的提交文件（提交文件 CSV 默认被 git 忽略）

## Run

先按 `data/README.md` 下载数据到 `data/raw/`，然后运行：

```bash
python src/baseline_hgb.py
python src/objective_weight_model.py
python src/cat_lgbm_models.py --model catboost
```
