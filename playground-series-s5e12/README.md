# Playground Series S5E12

二分类任务：预测 `diagnosed_diabetes`。

## Results

离线验证（ROC-AUC）摘要：

- HGB baseline（特征工程版）：0.7150 ± 0.0027（3-fold；200k 分层抽样）
- Objective score（entropy+CRITIC）：0.6947 ± 0.0005（2-fold；30k 分层抽样）
- CatBoost：0.7183 ± 0.0039（3-fold；200k 分层抽样）

详见周报：`reports/weekly_report.md`

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
