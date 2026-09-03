"""Convert risk probabilities into threshold, rejection-capacity, and tiered policies."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return NaN for a zero denominator instead of misreporting an undefined ratio."""
    return float(numerator / denominator) if denominator else float("nan")


def summarize_policy_mask(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    rejected: np.ndarray,
    performing_margin: float,
    loss_given_default: float,
) -> dict[str, float | int]:
    """Summarize approval, risk capture, and normalized economics for a rejection set."""
    target_array = np.asarray(target, dtype=int)
    probability_array = np.asarray(probability, dtype=float)
    rejected_array = np.asarray(rejected, dtype=bool)
    approved = ~rejected_array

    total_count = len(target_array)
    total_bad = int(target_array.sum())
    total_good = total_count - total_bad
    approved_count = int(approved.sum())
    rejected_count = int(rejected_array.sum())
    approved_bad = int(target_array[approved].sum())
    rejected_bad = int(target_array[rejected_array].sum())
    approved_good = approved_count - approved_bad
    rejected_good = rejected_count - rejected_bad
    normalized_value = (
        approved_good * performing_margin - approved_bad * loss_given_default
    ) / total_count

    return {
        "applicant_count": total_count,
        "approval_count": approved_count,
        "approval_rate": _safe_ratio(approved_count, total_count),
        "rejection_count": rejected_count,
        "rejection_rate": _safe_ratio(rejected_count, total_count),
        "approved_bad_count": approved_bad,
        "approved_bad_rate": _safe_ratio(approved_bad, approved_count),
        "rejected_bad_count": rejected_bad,
        "rejected_bad_rate": _safe_ratio(rejected_bad, rejected_count),
        "bad_capture_rate": _safe_ratio(rejected_bad, total_bad),
        "good_rejection_count": rejected_good,
        "good_rejection_rate": _safe_ratio(rejected_good, total_good),
        "rejection_precision": _safe_ratio(rejected_bad, rejected_count),
        "average_approved_probability": float(np.mean(probability_array[approved]))
        if approved_count
        else float("nan"),
        "normalized_value_per_applicant": float(normalized_value),
        "normalized_value_per_approved": _safe_ratio(
            approved_good * performing_margin - approved_bad * loss_given_default,
            approved_count,
        ),
    }


def build_threshold_policy_table(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    thresholds: list[float],
    performing_margin: float,
    loss_given_default: float,
) -> pd.DataFrame:
    """Compute approval, rejection, risk capture, and scenario value by threshold."""
    rows = []
    probability_array = np.asarray(probability, dtype=float)
    for threshold in sorted(set(thresholds)):
        row = summarize_policy_mask(
            target,
            probability_array,
            probability_array >= threshold,
            performing_margin,
            loss_given_default,
        )
        rows.append({"threshold": float(threshold), **row})
    return pd.DataFrame(rows)


def build_rejection_capacity_table(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    rejection_shares: list[float],
    performing_margin: float,
    loss_given_default: float,
) -> pd.DataFrame:
    """Reject a fixed highest-risk share so calibration shifts do not change capacity."""
    probability_array = np.asarray(probability, dtype=float)
    descending_rank = pd.Series(probability_array).rank(method="first", ascending=False).to_numpy()
    rows = []
    for share in sorted(set(rejection_shares)):
        rejection_count = math.ceil(len(probability_array) * share)
        rejected = descending_rank <= rejection_count
        threshold = float(np.min(probability_array[rejected])) if rejection_count else float("inf")
        row = summarize_policy_mask(
            target,
            probability_array,
            rejected,
            performing_margin,
            loss_given_default,
        )
        rows.append({"target_rejection_share": float(share), "threshold": threshold, **row})
    return pd.DataFrame(rows)


def build_three_band_table(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    high_risk_share: float = 0.10,
    review_share: float = 0.20,
) -> pd.DataFrame:
    """Assign the top 10% to high risk, the next 20% to review, and the rest to low risk."""
    target_array = np.asarray(target, dtype=int)
    probability_array = np.asarray(probability, dtype=float)
    percentile_rank = pd.Series(probability_array).rank(method="first", ascending=False, pct=True)
    band = np.select(
        [
            percentile_rank <= high_risk_share,
            percentile_rank <= high_risk_share + review_share,
        ],
        ["高风险｜拒绝候选", "中风险｜人工审核"],
        default="低风险｜自动批准",
    )
    frame = pd.DataFrame(
        {"risk_band": band, "target": target_array, "probability": probability_array}
    )
    order = ["高风险｜拒绝候选", "中风险｜人工审核", "低风险｜自动批准"]
    result = (
        frame.groupby("risk_band", observed=True)
        .agg(
            applicant_count=("target", "size"),
            bad_count=("target", "sum"),
            observed_bad_rate=("target", "mean"),
            average_predicted_probability=("probability", "mean"),
            minimum_probability=("probability", "min"),
            maximum_probability=("probability", "max"),
        )
        .reindex(order)
        .reset_index()
    )
    result["population_share"] = result["applicant_count"] / result["applicant_count"].sum()
    return result
