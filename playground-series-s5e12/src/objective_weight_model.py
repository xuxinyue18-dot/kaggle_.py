"""
HistGradientBoosting baseline with an additional composite feature built from
objective weights (entropy + CRITIC). The composite score summarizes numeric
features without using the target, then the supervised model learns on top.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
RAW_DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data" / "raw"))
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
REPORTS_DIR = PROJECT_DIR / "reports"
SUBMISSION_PATH = REPORTS_DIR / "submission_hgb_obj_weight.csv"
TARGET = "diagnosed_diabetes"
SEED = 42

# Smaller CV to keep runtime reasonable while trying the new feature idea.
CV_ROWS = 30_000
CV_FOLDS = 2
# Optionally cap rows used for final training to keep runtime reasonable.
TRAIN_ROWS = 150_000
ALPHA = 0.5  # blend factor between entropy and CRITIC weights


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

    return df


def stratified_sample(df: pd.DataFrame, rows: int, target: str, seed: int):
    """Stratified down-sample to a target row count if needed."""
    if rows and len(df) > rows:
        frac = rows / len(df)
        sampled = (
            df.groupby(target, group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=seed))
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
        return sampled
    return df


def to_float32_array(x):
    """Cast array-like input to float32 to speed up downstream model."""
    return np.asarray(x, dtype=np.float32)


def compute_objective_weights(df: pd.DataFrame, num_cols, alpha: float = ALPHA):
    """Compute blended entropy + CRITIC weights on numeric columns only."""
    num = df[num_cols].copy()

    # Min-max normalize to keep values positive for entropy; guard zero spans.
    eps = 1e-12
    min_ = num.min()
    span = (num.max() - min_).replace(to_replace=0, value=1.0)
    norm = ((num - min_) / span).clip(lower=eps)

    # Entropy weights.
    p = norm / norm.sum(axis=0)
    k = 1.0 / np.log(len(norm))
    entropy = -k * (p * np.log(p)).sum(axis=0)
    diff = 1.0 - entropy
    w_entropy = diff / diff.sum()

    # CRITIC weights: contrast intensity = std * (1 - |corr| summed).
    z = (num - num.mean()) / num.std(ddof=0).replace(to_replace=0, value=1.0)
    sd = z.std(ddof=0)
    corr = z.corr().abs()
    np.fill_diagonal(corr.values, 0.0)
    contrast = sd * (1.0 - corr).sum(axis=0)
    w_critic = contrast / contrast.sum()

    # Blend and renormalize.
    w = alpha * w_entropy + (1.0 - alpha) * w_critic
    w = w / w.sum()

    stats = {"min": min_, "span": span}
    return w, stats


class ObjectiveScoreAdder(BaseEstimator, TransformerMixin):
    """Fit blended objective weights on numeric columns and add a composite score."""

    def __init__(self, num_cols, alpha: float = ALPHA):
        self.num_cols = tuple(num_cols)
        self.alpha = alpha
        self.weights_ = None
        self.stats_ = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        # Keep column names if present
        if hasattr(X, "columns"):
            df.columns = X.columns
        w, stats = compute_objective_weights(df, list(self.num_cols), alpha=self.alpha)
        self.weights_ = w
        self.stats_ = stats
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if hasattr(X, "columns"):
            df.columns = X.columns
        norm = ((df[list(self.num_cols)] - self.stats_["min"]) / self.stats_["span"]).clip(lower=0.0)
        norm = norm.fillna(0.0)
        df["objective_score"] = norm.dot(self.weights_)
        return df


def build_pipeline(cat_cols, num_cols):
    """Add objective score, one-hot encode categoricals, and cast to float32."""
    score_adder = ObjectiveScoreAdder(num_cols, alpha=ALPHA)
    num_cols_with_score = num_cols + ["objective_score"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols_with_score),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    to_float32 = FunctionTransformer(to_float32_array)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=7,
        max_iter=80,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=1e-3,
        validation_fraction=0.15,
        n_iter_no_change=20,
        early_stopping=True,
        random_state=SEED,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("objective_score", score_adder),
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
    ]
    num_cols = [c for c in train.columns if c not in cat_cols + [TARGET, "id"]]

    X = train.drop(columns=[TARGET, "id"])
    y = train[TARGET]

    pipeline = build_pipeline(cat_cols, num_cols)

    # Use a stratified subsample for CV to keep runtime manageable.
    train_cv = stratified_sample(train, CV_ROWS, TARGET, SEED)
    X_cv = train_cv.drop(columns=[TARGET, "id"])
    y_cv = train_cv[TARGET]
    if len(train_cv) != len(train):
        print(f"CV using stratified sample of {len(train_cv):,} rows out of {len(train):,}.")
    else:
        print(f"CV using full training data of {len(train):,} rows.")

    # Log weights fitted on full train for reference (not used in CV fitting).
    weights, _ = compute_objective_weights(train, num_cols, alpha=ALPHA)

    # 2-fold stratified CV to estimate ROC-AUC; single-threaded for stability.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        pipeline,
        X_cv,
        y_cv,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
    )
    print("Top blended objective weights (first 10):")
    print(weights.sort_values(ascending=False).head(10))
    print(f"CV ROC-AUC ({CV_FOLDS}-fold): mean {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit on full (or sampled) training data.
    train_fit = stratified_sample(train, TRAIN_ROWS, TARGET, SEED)
    X_fit = train_fit.drop(columns=[TARGET, "id"])
    y_fit = train_fit[TARGET]
    if len(train_fit) != len(train):
        print(f"Training on stratified sample of {len(train_fit):,} rows out of {len(train):,}.")
    else:
        print(f"Training on full dataset of {len(train):,} rows.")

    pipeline.fit(X_fit, y_fit)

    # Predict probabilities for the positive class.
    test_pred = pipeline.predict_proba(test.drop(columns=["id"]))[:, 1]
    submission = pd.DataFrame({"id": test["id"], TARGET: test_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
