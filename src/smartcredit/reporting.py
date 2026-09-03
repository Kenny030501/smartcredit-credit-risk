"""Shared generators for SmartCredit model-diagnostic figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# Use a headless backend so terminals and CI can generate figures.
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


def save_diagnostic_figure(
    y_true: pd.Series,
    probability: np.ndarray,
    deciles: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate colorblind-friendly diagnostics distinguished by color, line, and marker."""
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probability)
    precision, recall, _ = precision_recall_curve(y_true, probability)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    blue = "#4477AA"
    purple = "#AA3377"
    orange = "#EE7733"
    gray = "#666666"

    axes[0, 0].plot(false_positive_rate, true_positive_rate, color=blue, linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], color=gray, linestyle="--", linewidth=1)
    axes[0, 0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")

    axes[0, 1].plot(recall, precision, color=purple, linewidth=2)
    axes[0, 1].axhline(float(np.mean(y_true)), color=gray, linestyle=":", linewidth=1)
    axes[0, 1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")

    axes[1, 0].plot(
        deciles["average_predicted_probability"],
        deciles["observed_bad_rate"],
        color=orange,
        marker="o",
        linewidth=2,
    )
    calibration_limit = max(
        0.35,
        float(deciles[["average_predicted_probability", "observed_bad_rate"]].max().max()) * 1.1,
    )
    axes[1, 0].plot(
        [0, calibration_limit],
        [0, calibration_limit],
        color=gray,
        linestyle="--",
        linewidth=1,
    )
    axes[1, 0].set_xlim(0, calibration_limit)
    axes[1, 0].set_ylim(0, calibration_limit)
    axes[1, 0].set(
        title="Calibration by risk decile",
        xlabel="Average predicted probability",
        ylabel="Observed bad rate",
    )

    bars = axes[1, 1].bar(
        deciles["risk_decile"].astype(str),
        deciles["observed_bad_rate"],
        color=blue,
        edgecolor="#222222",
        linewidth=0.6,
    )
    axes[1, 1].bar_label(
        bars,
        labels=[f"{value:.1%}" for value in deciles["observed_bad_rate"]],
        fontsize=8,
        padding=2,
    )
    axes[1, 1].set(
        title="Observed bad rate by risk decile",
        xlabel="Risk decile (1 = highest predicted risk)",
        ylabel="Observed bad rate",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_model_comparison_figure(
    logistic_target: pd.Series,
    logistic_probability: np.ndarray,
    hist_probability: np.ndarray,
    logistic_deciles: pd.DataFrame,
    hist_deciles: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare two models' ranking, calibration, and risk tiers on the same holdout set."""
    logistic_fpr, logistic_tpr, _ = roc_curve(logistic_target, logistic_probability)
    hist_fpr, hist_tpr, _ = roc_curve(logistic_target, hist_probability)
    logistic_precision, logistic_recall, _ = precision_recall_curve(
        logistic_target, logistic_probability
    )
    hist_precision, hist_recall, _ = precision_recall_curve(logistic_target, hist_probability)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    blue = "#4477AA"
    orange = "#EE7733"
    gray = "#666666"

    axes[0, 0].plot(logistic_fpr, logistic_tpr, color=blue, linewidth=2, label="Logistic")
    axes[0, 0].plot(
        hist_fpr,
        hist_tpr,
        color=orange,
        linestyle="--",
        linewidth=2,
        label="HistGradientBoosting",
    )
    axes[0, 0].plot([0, 1], [0, 1], color=gray, linestyle=":", linewidth=1)
    axes[0, 0].set(
        title="ROC comparison", xlabel="False positive rate", ylabel="True positive rate"
    )
    axes[0, 0].legend()

    axes[0, 1].plot(
        logistic_recall,
        logistic_precision,
        color=blue,
        linewidth=2,
        label="Logistic",
    )
    axes[0, 1].plot(
        hist_recall,
        hist_precision,
        color=orange,
        linestyle="--",
        linewidth=2,
        label="HistGradientBoosting",
    )
    axes[0, 1].axhline(float(np.mean(logistic_target)), color=gray, linestyle=":", linewidth=1)
    axes[0, 1].set(title="PR comparison", xlabel="Recall", ylabel="Precision")
    axes[0, 1].legend()

    calibration_limit = max(
        0.35,
        float(
            max(
                logistic_deciles[["average_predicted_probability", "observed_bad_rate"]]
                .max()
                .max(),
                hist_deciles[["average_predicted_probability", "observed_bad_rate"]].max().max(),
            )
        )
        * 1.1,
    )
    axes[1, 0].plot(
        logistic_deciles["average_predicted_probability"],
        logistic_deciles["observed_bad_rate"],
        color=blue,
        marker="o",
        linewidth=2,
        label="Logistic",
    )
    axes[1, 0].plot(
        hist_deciles["average_predicted_probability"],
        hist_deciles["observed_bad_rate"],
        color=orange,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="HistGradientBoosting",
    )
    axes[1, 0].plot(
        [0, calibration_limit],
        [0, calibration_limit],
        color=gray,
        linestyle=":",
        linewidth=1,
    )
    axes[1, 0].set_xlim(0, calibration_limit)
    axes[1, 0].set_ylim(0, calibration_limit)
    axes[1, 0].set(
        title="Calibration comparison",
        xlabel="Average predicted probability",
        ylabel="Observed bad rate",
    )
    axes[1, 0].legend()

    axes[1, 1].plot(
        logistic_deciles["risk_decile"],
        logistic_deciles["observed_bad_rate"],
        color=blue,
        marker="o",
        linewidth=2,
        label="Logistic",
    )
    axes[1, 1].plot(
        hist_deciles["risk_decile"],
        hist_deciles["observed_bad_rate"],
        color=orange,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="HistGradientBoosting",
    )
    axes[1, 1].set(
        title="Risk-decile comparison",
        xlabel="Risk decile (1 = highest predicted risk)",
        ylabel="Observed bad rate",
        xticks=range(1, 11),
    )
    axes[1, 1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_three_model_comparison_figure(
    target: pd.Series,
    logistic_probability: np.ndarray,
    hist_probability: np.ndarray,
    xgboost_probability: np.ndarray,
    logistic_deciles: pd.DataFrame,
    hist_deciles: pd.DataFrame,
    xgboost_deciles: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare three models using color, line style, and markers as redundant encodings."""
    probabilities = {
        "Logistic": logistic_probability,
        "HistGradientBoosting": hist_probability,
        "XGBoost": xgboost_probability,
    }
    decile_tables = {
        "Logistic": logistic_deciles,
        "HistGradientBoosting": hist_deciles,
        "XGBoost": xgboost_deciles,
    }
    styles = {
        "Logistic": {"color": "#4477AA", "linestyle": "-", "marker": "o"},
        "HistGradientBoosting": {
            "color": "#EE7733",
            "linestyle": "--",
            "marker": "s",
        },
        "XGBoost": {"color": "#AA3377", "linestyle": ":", "marker": "^"},
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    for name, probability in probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(target, probability)
        precision, recall, _ = precision_recall_curve(target, probability)
        style = styles[name]
        axes[0, 0].plot(
            false_positive_rate,
            true_positive_rate,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            label=name,
        )
        axes[0, 1].plot(
            recall,
            precision,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            label=name,
        )
    axes[0, 0].plot([0, 1], [0, 1], color="#666666", linestyle="-.", linewidth=1)
    axes[0, 0].set(
        title="ROC comparison", xlabel="False positive rate", ylabel="True positive rate"
    )
    axes[0, 0].legend()
    axes[0, 1].axhline(float(np.mean(target)), color="#666666", linestyle="-.", linewidth=1)
    axes[0, 1].set(title="PR comparison", xlabel="Recall", ylabel="Precision")
    axes[0, 1].legend()

    calibration_limit = max(
        0.35,
        max(
            float(table[["average_predicted_probability", "observed_bad_rate"]].max().max())
            for table in decile_tables.values()
        )
        * 1.1,
    )
    for name, table in decile_tables.items():
        style = styles[name]
        axes[1, 0].plot(
            table["average_predicted_probability"],
            table["observed_bad_rate"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2,
            label=name,
        )
        axes[1, 1].plot(
            table["risk_decile"],
            table["observed_bad_rate"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2,
            label=name,
        )
    axes[1, 0].plot(
        [0, calibration_limit],
        [0, calibration_limit],
        color="#666666",
        linestyle="-.",
        linewidth=1,
    )
    axes[1, 0].set_xlim(0, calibration_limit)
    axes[1, 0].set_ylim(0, calibration_limit)
    axes[1, 0].set(
        title="Calibration comparison",
        xlabel="Average predicted probability",
        ylabel="Observed bad rate",
    )
    axes[1, 0].legend()
    axes[1, 1].set(
        title="Risk-decile comparison",
        xlabel="Risk decile (1 = highest predicted risk)",
        ylabel="Observed bad rate",
        xticks=range(1, 11),
    )
    axes[1, 1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_calibration_comparison_figure(
    decile_tables: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Compare decile calibration curves for raw, Platt, and isotonic probabilities."""
    styles = {
        "raw": {"label": "Raw", "color": "#4477AA", "linestyle": "-", "marker": "o"},
        "platt": {
            "label": "Platt",
            "color": "#EE7733",
            "linestyle": "--",
            "marker": "s",
        },
        "isotonic": {
            "label": "Isotonic",
            "color": "#AA3377",
            "linestyle": ":",
            "marker": "^",
        },
    }
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    limit = max(
        0.35,
        max(
            float(table[["average_predicted_probability", "observed_bad_rate"]].max().max())
            for table in decile_tables.values()
        )
        * 1.1,
    )
    for method, table in decile_tables.items():
        style = styles[method]
        axis.plot(
            table["average_predicted_probability"],
            table["observed_bad_rate"],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2,
        )
    axis.plot([0, limit], [0, limit], color="#666666", linestyle="-.", linewidth=1)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set(
        title="Probability calibration comparison",
        xlabel="Average predicted probability",
        ylabel="Observed bad rate",
    )
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_policy_tradeoff_figure(capacity_table: pd.DataFrame, output_path: Path) -> None:
    """Plot rejection capacity against approved risk, risk capture, and scenario value."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    rejection_share = capacity_table["rejection_rate"]

    axes[0].plot(
        rejection_share,
        capacity_table["approved_bad_rate"],
        color="#4477AA",
        marker="o",
        linewidth=2,
    )
    axes[0].set(
        title="Approved bad rate",
        xlabel="Rejection rate",
        ylabel="Observed bad rate among approvals",
    )

    axes[1].plot(
        rejection_share,
        capacity_table["bad_capture_rate"],
        color="#AA3377",
        marker="s",
        linestyle="--",
        linewidth=2,
    )
    axes[1].set(
        title="Bad-sample capture",
        xlabel="Rejection rate",
        ylabel="Share of all bad samples rejected",
    )

    axes[2].plot(
        rejection_share,
        capacity_table["normalized_value_per_applicant"],
        color="#EE7733",
        marker="^",
        linestyle=":",
        linewidth=2,
    )
    axes[2].set(
        title="Normalized scenario value",
        xlabel="Rejection rate",
        ylabel="Value per applicant",
    )
    for axis in axes:
        axis.grid(axis="y", linestyle=":", alpha=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
