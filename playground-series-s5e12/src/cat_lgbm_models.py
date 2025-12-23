"""
CatBoost / LightGBM baselines for Playground Series S5E12 (diabetes prediction).

Default: CatBoost, 3-fold CV on a stratified 200k sample to keep runtime reasonable,
then train on full data and write submission to kaggle/playground-series-s5e12/.

Note: catboost/lightgbm must be installed. On this Termux environment, prebuilt wheels
are unavailable, so install/run this script in an environment that provides them
(e.g., Kaggle Notebook/Colab/local x86_64 with pip wheels).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
RAW_DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data" / "raw"))
REPORTS_DIR = PROJECT_DIR / "reports"
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


def stratified_sample(df: pd.DataFrame, rows: int | None) -> pd.DataFrame:
    """Optionally take a stratified sample to speed up CV."""
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
    """Label-encode categoricals jointly across train+test for LightGBM."""
    train_enc = train.copy()
    test_enc = test.copy()
    for col in cat_cols:
        enc = LabelEncoder()
        combined = pd.concat([train_enc[col], test_enc[col]], axis=0).astype(str)
        enc.fit(combined)
        train_enc[col] = enc.transform(train_enc[col].astype(str))
        test_enc[col] = enc.transform(test_enc[col].astype(str))
    return train_enc, test_enc


def run_catboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cv_rows: int | None,
    folds: int,
) -> pd.Series:
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
            learning_rate=0.1,
            depth=8,
            iterations=700,
            l2_leaf_reg=3.0,
            subsample=0.8,
            colsample_bylevel=0.8,
            random_seed=SEED,
            od_type="Iter",
            od_wait=40,
            verbose=False,
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
        auc = model.get_best_score()["validation"]["AUC"]
        aucs.append(auc)
        print(f"  Fold {fold}: AUC {auc:.4f} (best_iter={model.tree_count_})")

    print(f"CatBoost CV AUC: mean {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Train on full data.
    X_full = train.drop(columns=[TARGET, "id"])
    y_full = train[TARGET]
    full_pool = Pool(X_full, y_full, cat_features=[X_full.columns.get_loc(c) for c in CAT_COLS])
    full_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        learning_rate=0.08,
        depth=8,
        iterations=800,
        l2_leaf_reg=3.0,
        subsample=0.85,
        colsample_bylevel=0.85,
        random_seed=SEED,
        verbose=200,
    )
    print("Training CatBoost on full data...")
    full_model.fit(full_pool)
    test_no_id = test.drop(columns=["id"])
    test_pool = Pool(test_no_id, cat_features=[test_no_id.columns.get_loc(c) for c in CAT_COLS])
    preds = full_model.predict_proba(test_pool)[:, 1]
    return pd.Series(preds, name=TARGET)


def run_lightgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cv_rows: int | None,
    folds: int,
) -> pd.Series:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm is not installed. Please install and retry.") from exc

    train = train.copy()
    test = test.copy()
    # Label-encode categoricals for LightGBM.
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
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 30,
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
            early_stopping_rounds=80,
            verbose_eval=False,
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
        num_boost_round=max(best_iter, 200),
        verbose_eval=False,
    )
    preds = final_model.predict(test_enc.drop(columns=["id"]), num_iteration=final_model.best_iteration)
    return pd.Series(preds, name=TARGET)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["catboost", "lightgbm"],
        default="catboost",
        help="Which model to run.",
    )
    parser.add_argument(
        "--cv-rows",
        type=int,
        default=200_000,
        help="Stratified sample size for CV (None for full data).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save submission CSV. Defaults to kaggle/.../submission_<model>.csv",
    )
    args = parser.parse_args()

    train_path = RAW_DATA_DIR / "train.csv"
    test_path = RAW_DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(
            f"Missing data files. Expected:\n- {train_path}\n- {test_path}\n"
            "See playground-series-s5e12/data/README.md for download instructions."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = REPORTS_DIR / f"submission_{args.model}.csv"

    if args.model == "catboost":
        preds = run_catboost(train, test, args.cv_rows, args.folds)
    else:
        preds = run_lightgbm(train, test, args.cv_rows, args.folds)

    submission = pd.DataFrame({"id": test["id"], TARGET: preds})
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
