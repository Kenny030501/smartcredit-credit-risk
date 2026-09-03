"""Validate key properties of model evaluation and risk grouping."""

import numpy as np
import pandas as pd

from smartcredit.modeling import (
    build_hist_gradient_boosting_pipeline,
    build_risk_deciles,
    build_xgboost_pipeline,
    calculate_binary_metrics,
)


def test_binary_metrics_for_perfect_ranking() -> None:
    """A perfect probability ranking should produce ROC-AUC and KS of 1.0."""
    target = pd.Series([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_binary_metrics(target, probability)

    assert metrics["roc_auc"] == 1.0
    assert metrics["ks"] == 1.0


def test_risk_deciles_are_ordered_high_to_low() -> None:
    """Risk decile 1 must have a higher mean probability than decile 10."""
    target = pd.Series(([0] * 90) + ([1] * 10))
    probability = np.linspace(0.01, 0.99, 100)

    deciles = build_risk_deciles(target, probability)

    assert deciles["applicant_count"].sum() == 100
    assert deciles.iloc[0]["risk_decile"] == 1
    assert (
        deciles.iloc[0]["average_predicted_probability"]
        > deciles.iloc[-1]["average_predicted_probability"]
    )


def test_hist_gradient_boosting_pipeline_handles_missing_and_unknown_categories() -> None:
    """The boosting model should handle numeric missingness and unseen categories."""
    features = pd.DataFrame(
        {
            "amount": [100.0, np.nan, 300.0, 400.0, 500.0, 600.0],
            "segment": ["A", "B", "A", "B", "A", "B"],
        }
    )
    target = pd.Series([0, 1, 0, 1, 0, 1])
    pipeline = build_hist_gradient_boosting_pipeline(["amount"], ["segment"])
    pipeline.set_params(model__early_stopping=False, model__max_iter=2, model__min_samples_leaf=1)

    pipeline.fit(features, target)
    probability = pipeline.predict_proba(pd.DataFrame({"amount": [np.nan], "segment": ["UNSEEN"]}))[
        :, 1
    ]

    assert probability.shape == (1,)
    assert 0 <= probability[0] <= 1


def test_xgboost_pipeline_handles_missing_and_unknown_categories() -> None:
    """XGBoost should handle numeric missingness and new categories encoded as missing."""
    features = pd.DataFrame(
        {
            "amount": [100.0, np.nan, 300.0, 400.0, 500.0, 600.0],
            "segment": ["A", "B", "A", "B", "A", "B"],
        }
    )
    target = pd.Series([0, 1, 0, 1, 0, 1])
    pipeline = build_xgboost_pipeline(["amount"], ["segment"])
    pipeline.set_params(
        model__n_estimators=2,
        model__early_stopping_rounds=None,
        model__min_child_weight=1,
    )

    pipeline.fit(features, target)
    probability = pipeline.predict_proba(pd.DataFrame({"amount": [np.nan], "segment": ["UNSEEN"]}))[
        :, 1
    ]

    assert probability.shape == (1,)
    assert 0 <= probability[0] <= 1
