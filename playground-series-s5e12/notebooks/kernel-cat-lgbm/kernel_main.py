"""
Kaggle script kernel for Playground Series S5E12 using CatBoost or LightGBM.
Default: LightGBM with stratified CV sample to keep runtime manageable.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

TARGET = "diagnosed_diabetes"
CAT_COLS = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
]
SEED = 42

# Prefer Kaggle input path; fallback to local project root when running offline.
DEFAULT_DATA_DIR = Path("/kaggle/input/playground-series-s5e12")
LOCAL_DATA_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DEFAULT_DATA_DIR if DEFAULT_DATA_DIR.exists() else LOCAL_DATA_DIR


def stratified_sample(df: pd.DataFrame, rows: int | None) -> pd.DataFrame:
    if rows and len(df) > rows:
        frac = rows / len(df)
        return (
            df.groupby(TARGET, group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=SEED))
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )
    return df


def encode_categoricals(
    train: pd.DataFrame, test: pd.DataFrame, cat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_enc = train.copy()
    test_enc = test.copy()
    for col in cat_cols:
        enc = LabelEncoder()
        combined = pd.concat([train_enc[col], test_enc[col]], axis=0).astype(str)
        enc.fit(combined)
        train_enc[col] = enc.transform(train_enc[col].astype(str))
        test_enc[col] = enc.transform(test_enc[col].astype(str))
    return train_enc, test_enc


def run_catboost(train: pd.DataFrame, test: pd.DataFrame, cv_rows: int | None, folds: int) -> pd.Series:
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise SystemExit("catboost is not installed. Please install and retry.") from exc

    train = train.copy()
    sample = stratified_sample(train, cv_rows)
    X_sample = sample.drop(columns=[TARGET, "id"])
    y_sample = sample[TARGET]
    cat_idx = [X_sample.columns.get_loc(c) for c in CAT_COLS]

    print(f"CatBoost CV on {len(sample):,} rows ({folds} folds)...")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_sample, y_sample), 1):
        train_pool = Pool(
            X_sample.iloc[tr_idx],
            y_sample.iloc[tr_idx],
            cat_features=cat_idx,
        )
        val_pool = Pool(
            X_sample.iloc[va_idx],
            y_sample.iloc[va_idx],
            cat_features=cat_idx,
        )
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            learning_rate=0.06,
            depth=8,
            iterations=1000,
            l2_leaf_reg=3.0,
            subsample=0.9,
            bootstrap_type="Bernoulli",
            random_seed=SEED,
            od_type="Iter",
            od_wait=50,
            task_type="CPU",
            verbose=False,
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
        auc = model.get_best_score()["validation"]["AUC"]
        aucs.append(auc)
        print(f"  Fold {fold}: AUC {auc:.4f} (best_iter={model.tree_count_})")

    print(f"CatBoost CV AUC: mean {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    X_full = train.drop(columns=[TARGET, "id"])
    y_full = train[TARGET]
    full_pool = Pool(X_full, y_full, cat_features=[X_full.columns.get_loc(c) for c in CAT_COLS])
    full_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            learning_rate=0.05,
        depth=8,
        iterations=1600,
        l2_leaf_reg=3.0,
        subsample=0.9,
        bootstrap_type="Bernoulli",
        random_seed=SEED,
        task_type="CPU",
        verbose=200,
    )
    print("Training CatBoost on full data...")
    full_model.fit(full_pool)
    test_no_id = test.drop(columns=["id"])
    test_pool = Pool(test_no_id, cat_features=[test_no_id.columns.get_loc(c) for c in CAT_COLS])
    preds = full_model.predict_proba(test_pool)[:, 1]
    return pd.Series(preds, name=TARGET)


def run_lightgbm(train: pd.DataFrame, test: pd.DataFrame, cv_rows: int | None, folds: int) -> pd.Series:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm is not installed. Please install and retry.") from exc

    train = train.copy()
    test = test.copy()
    train_enc, test_enc = encode_categoricals(train, test, CAT_COLS)
    sample = stratified_sample(train_enc, cv_rows)
    X_sample = sample.drop(columns=[TARGET, "id"])
    y_sample = sample[TARGET]

    print(f"LightGBM CV on {len(sample):,} rows ({folds} folds)...")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    aucs = []
    best_iters = []

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 255,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "seed": SEED,
        "verbose": -1,
    }

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_sample, y_sample), 1):
        dtrain = lgb.Dataset(
            X_sample.iloc[tr_idx],
            label=y_sample.iloc[tr_idx],
            categorical_feature=CAT_COLS,
            free_raw_data=True,
        )
        dval = lgb.Dataset(
            X_sample.iloc[va_idx],
            label=y_sample.iloc[va_idx],
            categorical_feature=CAT_COLS,
            free_raw_data=True,
        )
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1200,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred = model.predict(X_sample.iloc[va_idx], num_iteration=model.best_iteration)
        auc = roc_auc_score(y_sample.iloc[va_idx], pred)
        aucs.append(auc)
        best_iters.append(model.best_iteration or 0)
        print(f"  Fold {fold}: AUC {auc:.4f} (best_iter={model.best_iteration})")

    print(f"LightGBM CV AUC: mean {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    best_iter = int(np.mean(best_iters)) if best_iters else 400

    X_full = train_enc.drop(columns=[TARGET, "id"])
    y_full = train_enc[TARGET]
    dfull = lgb.Dataset(
        X_full,
        label=y_full,
        categorical_feature=CAT_COLS,
        free_raw_data=True,
    )
    print("Training LightGBM on full data...")
    final_model = lgb.train(
        params,
        dfull,
        num_boost_round=max(best_iter, 300),
    )
    preds = final_model.predict(test_enc.drop(columns=["id"]), num_iteration=final_model.best_iteration)
    return pd.Series(preds, name=TARGET)


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    model = "catboost"  # switch to CatBoost (enable GPU in kernel settings)
    cv_rows = 200_000
    folds = 3

    preds = run_catboost(train, test, cv_rows=cv_rows, folds=folds)

    submission = pd.DataFrame({"id": test["id"], TARGET: preds})
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
