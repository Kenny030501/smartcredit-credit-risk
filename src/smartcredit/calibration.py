"""Probability-calibrator fitting, application, and method selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    """Convert probabilities safely to log odds for Platt scaling."""
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_calibrators(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
) -> dict[str, object]:
    """Fit Platt and isotonic post-processing calibrators on out-of-sample predictions."""
    target_array = np.asarray(target, dtype=int)
    platt = LogisticRegression(C=1_000_000, solver="lbfgs", max_iter=1_000)
    platt.fit(probability_to_logit(probability), target_array)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(np.asarray(probability, dtype=float), target_array)
    return {"platt": platt, "isotonic": isotonic}


def apply_calibrator(
    method: str, calibrators: dict[str, object], probability: np.ndarray
) -> np.ndarray:
    """Return raw, Platt-calibrated, or isotonic-calibrated probabilities by name."""
    raw = np.asarray(probability, dtype=float)
    if method == "raw":
        return raw.copy()
    if method == "platt":
        return calibrators["platt"].predict_proba(probability_to_logit(raw))[:, 1]
    if method == "isotonic":
        return np.asarray(calibrators["isotonic"].predict(raw), dtype=float)
    raise ValueError(f"未知校准方法：{method}")


def select_calibration_method(
    metrics: dict[str, dict[str, object]],
    minimum_brier_improvement: float = 1e-5,
) -> str:
    """Replace raw probabilities only when Brier improves materially without worse log loss."""
    raw_brier = float(metrics["raw"]["brier_score"])
    raw_log_loss = float(metrics["raw"]["log_loss"])
    candidates = [
        method
        for method in metrics
        if method != "raw"
        and raw_brier - float(metrics[method]["brier_score"]) >= minimum_brier_improvement
        and float(metrics[method]["log_loss"]) <= raw_log_loss
    ]
    if not candidates:
        return "raw"
    return min(
        candidates,
        key=lambda method: (
            float(metrics[method]["brier_score"]),
            float(metrics[method]["log_loss"]),
        ),
    )
