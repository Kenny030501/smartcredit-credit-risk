"""Shared SmartCredit model training, evaluation, and risk-tier functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from smartcredit.xgboost_runtime import get_xgboost_classifier


def build_logistic_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """Build a logistic pipeline whose imputation and scaling fit training data only."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        # Dense output avoids sparse-index overhead at the current feature cardinality.
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        C=1.0,
        class_weight=None,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_hist_gradient_boosting_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """Build a histogram-gradient-boosting pipeline with native missing and category handling."""
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            # HistGradientBoosting learns split directions for missing numeric values directly.
            ("numeric", "passthrough", numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    # ColumnTransformer emits numeric fields first, followed by ordinal-encoded categories.
    categorical_mask = ([False] * len(numeric_columns)) + ([True] * len(categorical_columns))
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        early_stopping=True,
        scoring="roc_auc",
        validation_fraction=0.1,
        n_iter_no_change=25,
        tol=1e-4,
        random_state=42,
        class_weight=None,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_xgboost_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """Build an XGBoost pipeline with native numeric-missing and categorical splits."""
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            # XGBoost handles numeric missing values natively, without imputation or scaling.
            ("numeric", "passthrough", numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    feature_types = (["q"] * len(numeric_columns)) + (["c"] * len(categorical_columns))
    xgboost_classifier = get_xgboost_classifier()
    model = xgboost_classifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        feature_types=feature_types,
        learning_rate=0.03,
        n_estimators=2_000,
        max_depth=4,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=10.0,
        max_bin=255,
        max_cat_to_onehot=4,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def calculate_binary_metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, object]:
    """Compute probability metrics and classification results at a fixed 0.5 threshold."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probability)
    ks_values = true_positive_rate - false_positive_rate
    ks_index = int(np.argmax(ks_values))
    predicted_class = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted_class).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability)),
        "ks": float(ks_values[ks_index]),
        "ks_threshold": float(thresholds[ks_index]),
        "threshold_0_5": {
            "precision": float(precision_score(y_true, predicted_class, zero_division=0)),
            "recall": float(recall_score(y_true, predicted_class, zero_division=0)),
            "f1": float(f1_score(y_true, predicted_class, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted_class)),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        },
    }


def build_risk_deciles(y_true: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    """Split predictions into descending risk deciles and summarize bad rates and lift."""
    frame = pd.DataFrame(
        {
            "target": np.asarray(y_true, dtype=int),
            "predicted_probability": probability,
        }
    )
    # Create unique ranks so tied probabilities do not prevent qcut from forming ten groups.
    descending_rank = frame["predicted_probability"].rank(method="first", ascending=False)
    frame["risk_decile"] = pd.qcut(descending_rank, q=10, labels=range(1, 11)).astype(int)
    overall_bad_rate = frame["target"].mean()

    result = (
        frame.groupby("risk_decile", observed=True)
        .agg(
            applicant_count=("target", "size"),
            bad_count=("target", "sum"),
            observed_bad_rate=("target", "mean"),
            average_predicted_probability=("predicted_probability", "mean"),
            minimum_predicted_probability=("predicted_probability", "min"),
            maximum_predicted_probability=("predicted_probability", "max"),
        )
        .reset_index()
        .sort_values("risk_decile")
    )
    result["bad_rate_lift"] = result["observed_bad_rate"] / overall_bad_rate
    result["population_share"] = result["applicant_count"] / result["applicant_count"].sum()
    return result


def expected_calibration_error(deciles: pd.DataFrame) -> float:
    """Compute the weighted absolute calibration gap across risk deciles."""
    gap = (deciles["observed_bad_rate"] - deciles["average_predicted_probability"]).abs()
    return float((gap * deciles["population_share"]).sum())
