import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
# Directory that contains train.csv / test.csv.
RAW_DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data" / "raw"))
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
REPORTS_DIR = PROJECT_DIR / "reports"
SUBMISSION_PATH = REPORTS_DIR / "submission_hgb_fe.csv"
TARGET = "diagnosed_diabetes"
SEED = 42
CV_ROWS = 200_000  # stratified sample size for CV to keep memory/runtime reasonable
TUNE_ROWS = 80_000  # rows used for small hyperparameter sweep (stratified)
DO_TUNE = True


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple ratios/buckets to give the tree model more signal."""
    df = df.copy()

    hdl = df["hdl_cholesterol"].clip(lower=1e-3)
    bmi = df["bmi"].clip(lower=1e-3)
    screen = df["screen_time_hours_per_day"] + 1.0

    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["mean_arterial_pressure"] = (2 * df["diastolic_bp"] + df["systolic_bp"]) / 3
    df["chol_hdl_ratio"] = df["cholesterol_total"] / hdl
    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / hdl
    df["trig_hdl_ratio"] = df["triglycerides"] / hdl
    df["activity_screen_ratio"] = df["physical_activity_minutes_per_week"] / screen
    df["alcohol_per_day"] = df["alcohol_consumption_per_week"] / 7.0
    df["bmi_waist_product"] = df["bmi"] * df["waist_to_hip_ratio"]
    df["bp_systolic_to_bmi"] = df["systolic_bp"] / bmi
    df["bp_diastolic_to_bmi"] = df["diastolic_bp"] / bmi

    df["age_bin"] = pd.cut(
        df["age"],
        bins=[-np.inf, 30, 40, 50, 60, 70, np.inf],
        labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"],
    )
    df["bmi_bin"] = pd.cut(
        df["bmi"],
        bins=[-np.inf, 18.5, 25, 30, 35, np.inf],
        labels=["under", "normal", "over", "obese", "severe"],
    )

    # Simple crossed categoricals to capture interactions.
    df["age_bmi_cross"] = df["age_bin"].astype(str) + "|" + df["bmi_bin"].astype(str)
    df["gender_smoking"] = df["gender"].astype(str) + "|" + df["smoking_status"].astype(str)

    return df


def to_float32_array(x):
    """Cast array-like input to float32 to speed up downstream model."""
    return np.asarray(x, dtype=np.float32)


def build_pipeline(cat_cols, num_cols, model_params=None):
    """One-hot encode categoricals, pass numeric features through, and cast to float32 for speed."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    to_float32 = FunctionTransformer(to_float32_array)

    default_params = dict(
        learning_rate=0.05,
        max_depth=8,
        max_iter=400,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=1e-3,
        validation_fraction=0.15,
        n_iter_no_change=20,
        early_stopping=True,
        random_state=SEED,
        class_weight="balanced",
    )
    params = {**default_params, **(model_params or {})}
    model = HistGradientBoostingClassifier(**params)

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("to_float32", to_float32),
            ("model", model),
        ]
    )


def main():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise SystemExit(
            f"Missing data files. Expected:\n- {TRAIN_PATH}\n- {TEST_PATH}\n"
            "See playground-series-s5e12/data/README.md for download instructions."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train = add_features(pd.read_csv(TRAIN_PATH))
    test = add_features(pd.read_csv(TEST_PATH))

    cat_cols = [
        "gender",
        "ethnicity",
        "education_level",
        "income_level",
        "smoking_status",
        "employment_status",
        "age_bin",
        "bmi_bin",
        "age_bmi_cross",
        "gender_smoking",
    ]
    num_cols = [c for c in train.columns if c not in cat_cols + [TARGET, "id"]]

    X = train.drop(columns=[TARGET, "id"])
    y = train[TARGET]

    best_params = {}
    if DO_TUNE:
        # Light hyperparameter sweep on a stratified subsample.
        candidates = [
            dict(learning_rate=0.05, max_iter=450, max_depth=8, max_leaf_nodes=63, min_samples_leaf=25),
            dict(learning_rate=0.03, max_iter=650, max_depth=9, max_leaf_nodes=127, min_samples_leaf=20),
            dict(learning_rate=0.02, max_iter=800, max_depth=10, max_leaf_nodes=127, min_samples_leaf=30),
        ]
        frac = min(1.0, TUNE_ROWS / len(train))
        sample = (
            train.groupby(TARGET, group_keys=False).sample(frac=frac, random_state=SEED)
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )
        X_tune = sample.drop(columns=[TARGET, "id"])
        y_tune = sample[TARGET]
        cv_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        best_score = -np.inf
        for params in candidates:
            pipe = build_pipeline(cat_cols, num_cols, params)
            scores = cross_val_score(pipe, X_tune, y_tune, cv=cv_tune, scoring="roc_auc", n_jobs=1)
            mean_score = scores.mean()
            print(f"Tune params {params} -> CV AUC {mean_score:.4f} ± {scores.std():.4f}")
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        print(f"Selected params: {best_params} (CV {best_score:.4f})")

    pipeline = build_pipeline(cat_cols, num_cols, best_params)

    # Use a stratified subsample for CV to keep runtime manageable.
    if CV_ROWS and len(train) > CV_ROWS:
        frac = CV_ROWS / len(train)
        sampled = (
            train.groupby(TARGET, group_keys=False).sample(frac=frac, random_state=SEED)
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )
        X_cv = sampled.drop(columns=[TARGET, "id"])
        y_cv = sampled[TARGET]
        print(f"CV using stratified sample of {len(sampled):,} rows out of {len(train):,}.")
    else:
        X_cv, y_cv = X, y
        print(f"CV using full training data of {len(train):,} rows.")

    # 3-fold stratified CV to estimate ROC-AUC; single-threaded to avoid excess memory use.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        pipeline,
        X_cv,
        y_cv,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,  # single-threaded to avoid huge memory use on dense design matrices
    )
    print(f"CV ROC-AUC: mean {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit on full training data.
    pipeline.fit(X, y)

    # Predict probabilities for the positive class.
    test_pred = pipeline.predict_proba(test.drop(columns=["id"]))[:, 1]
    submission = pd.DataFrame({"id": test["id"], TARGET: test_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
