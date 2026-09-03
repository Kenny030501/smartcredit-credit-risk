"""Validate key probability-calibration and credit-policy properties."""

import numpy as np
import pandas as pd

from smartcredit.calibration import (
    apply_calibrator,
    fit_calibrators,
    probability_to_logit,
    select_calibration_method,
)
from smartcredit.policy import (
    build_rejection_capacity_table,
    build_three_band_table,
    build_threshold_policy_table,
)


def test_calibrators_return_valid_probabilities() -> None:
    """Platt and isotonic outputs must remain between zero and one."""
    target = pd.Series([0, 0, 0, 1, 1, 1])
    probability = np.array([0.05, 0.20, 0.35, 0.55, 0.70, 0.90])
    calibrators = fit_calibrators(target, probability)

    assert probability_to_logit(np.array([0.0, 1.0])).shape == (2, 1)
    for method in ("raw", "platt", "isotonic"):
        calibrated = apply_calibrator(method, calibrators, probability)
        assert calibrated.shape == probability.shape
        assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_threshold_table_trades_approval_for_bad_capture() -> None:
    """A higher rejection threshold should raise approval and lower high-risk capture."""
    target = pd.Series([0, 0, 0, 1, 1, 1])
    probability = np.array([0.01, 0.05, 0.10, 0.20, 0.40, 0.80])
    table = build_threshold_policy_table(target, probability, [0.1, 0.5], 0.08, 0.5)

    assert table["approval_rate"].is_monotonic_increasing
    assert table["bad_capture_rate"].is_monotonic_decreasing
    assert table.iloc[0]["approved_bad_rate"] <= table.iloc[-1]["approved_bad_rate"]


def test_capacity_and_three_band_tables_preserve_population() -> None:
    """Capacity policies and three-tier assignments must preserve every applicant."""
    target = pd.Series(([0] * 90) + ([1] * 10))
    probability = np.linspace(0.01, 0.99, 100)
    capacity = build_rejection_capacity_table(target, probability, [0.0, 0.1], 0.08, 0.5)
    bands = build_three_band_table(target, probability)

    assert capacity.iloc[0]["rejection_count"] == 0
    assert capacity.iloc[1]["rejection_count"] == 10
    assert bands["applicant_count"].sum() == 100
    assert bands["population_share"].sum() == 1.0


def test_calibration_selection_requires_material_brier_improvement() -> None:
    """Negligible Brier gains or worse log loss should not replace raw probabilities."""
    metrics = {
        "raw": {"brier_score": 0.066400, "log_loss": 0.239300},
        "platt": {"brier_score": 0.0663995, "log_loss": 0.239299},
        "isotonic": {"brier_score": 0.066390, "log_loss": 0.239500},
    }

    assert select_calibration_method(metrics) == "raw"
