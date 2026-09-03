"""Train HistGradientBoosting and compare it with logistic on the same holdout design."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from smartcredit.data import clean_features, load_model_frames
from smartcredit.modeling import (
    build_hist_gradient_boosting_pipeline,
    build_risk_deciles,
    calculate_binary_metrics,
    expected_calibration_error,
)
from smartcredit.reporting import save_diagnostic_figure, save_model_comparison_figure

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "interim" / "smartcredit.duckdb"
MODEL_DIR = ROOT / "models" / "hist_gradient_boosting"
PROCESSED_DIR = ROOT / "data" / "processed" / "hist_gradient_boosting"
REPORT_DIR = ROOT / "reports" / "modeling"
TABLE_DIR = ROOT / "reports" / "tables" / "hist_gradient_boosting"
FIGURE_DIR = ROOT / "reports" / "figures" / "hist_gradient_boosting"
LOGISTIC_PREDICTIONS = (
    ROOT / "data" / "processed" / "logistic_baseline" / "validation_predictions.parquet"
)
LOGISTIC_METRICS = ROOT / "reports" / "modeling" / "logistic_baseline_metrics.json"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
PERMUTATION_SAMPLE_SIZE = 5_000
PERMUTATION_REPEATS = 2


def parse_args() -> argparse.Namespace:
    """Parse runtime arguments; preserve existing boosting artifacts by default."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有的HistGradientBoosting模型和预测产物",
    )
    return parser.parse_args()


def prepare_output_directories(overwrite: bool) -> None:
    """Create output directories and prevent accidental replacement of a trained model."""
    model_path = MODEL_DIR / "model.joblib"
    if model_path.exists() and not overwrite:
        raise FileExistsError(f"{model_path} 已存在；如需重跑请使用 --overwrite")
    for path in (MODEL_DIR, PROCESSED_DIR, REPORT_DIR, TABLE_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def verify_shared_validation_ids(validation_ids: pd.Series) -> None:
    """Confirm that boosting and logistic use exactly the same outer holdout applicants."""
    if not LOGISTIC_PREDICTIONS.exists():
        raise FileNotFoundError(f"缺少Logistic验证集预测：{LOGISTIC_PREDICTIONS}")
    logistic_ids = pd.read_parquet(LOGISTIC_PREDICTIONS, columns=["SK_ID_CURR"])[
        "SK_ID_CURR"
    ].to_numpy()
    if not np.array_equal(np.sort(validation_ids.to_numpy()), np.sort(logistic_ids)):
        raise ValueError("提升模型与Logistic基线的验证集申请人不一致")


def calculate_permutation_importance(
    model_pipeline: object,
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Compute raw-feature permutation importance on a fixed holdout subsample."""
    sample_count = min(PERMUTATION_SAMPLE_SIZE, len(features))
    sample_features = features.sample(n=sample_count, random_state=RANDOM_STATE)
    sample_target = target.loc[sample_features.index]
    result = permutation_importance(
        model_pipeline,
        sample_features,
        sample_target,
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def load_logistic_results() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Load frozen logistic metrics, holdout probabilities, and risk deciles."""
    if not LOGISTIC_METRICS.exists():
        raise FileNotFoundError(f"缺少Logistic指标：{LOGISTIC_METRICS}")
    logistic_manifest = json.loads(LOGISTIC_METRICS.read_text(encoding="utf-8"))
    logistic_predictions = pd.read_parquet(LOGISTIC_PREDICTIONS).sort_values("SK_ID_CURR")
    logistic_deciles = build_risk_deciles(
        logistic_predictions["TARGET"],
        logistic_predictions["PREDICTED_PROBABILITY"].to_numpy(),
    )
    return logistic_manifest["metrics"], logistic_predictions, logistic_deciles


def write_hist_report(
    metrics: dict[str, object],
    deciles: pd.DataFrame,
    importance: pd.DataFrame,
    best_iterations: int,
    fit_seconds: float,
    train_count: int,
    validation_count: int,
) -> None:
    """Generate the boosting validation, model-design, and interpretation-limit report."""
    confusion = metrics["threshold_0_5"]["confusion_matrix"]
    top_importance_rows = "\n".join(
        f"| `{row.feature}` | {row.importance_mean:.5f} | {row.importance_std:.5f} |"
        for row in importance.head(20).itertuples()
    )
    decile_rows = "\n".join(
        "| {risk_decile} | {applicant_count:,} | {observed_bad_rate:.2%} | "
        "{average_predicted_probability:.2%} | {bad_rate_lift:.2f} |".format(**row._asdict())
        for row in deciles.itertuples(index=False)
    )
    report = f"""# HistGradientBoosting提升模型报告

## 结论

- 验证集ROC-AUC：{metrics["roc_auc"]:.4f}
- 验证集PR-AUC：{metrics["pr_auc"]:.4f}
- KS：{metrics["ks"]:.4f}
- Brier Score：{metrics["brier_score"]:.4f}，越低越好
- 十分位校准误差：{metrics["decile_calibration_error"]:.4f}，越低越好
- 最高风险十分位真实坏样本率：{deciles.iloc[0]["observed_bad_rate"]:.2%}，为总体的{deciles.iloc[0]["bad_rate_lift"]:.2f}倍

## 验证设计

- 开发训练集：{train_count:,}行；独立验证集：{validation_count:,}行
- 与Logistic基线使用完全相同的80/20分层随机划分和`random_state=42`
- 外层验证集不参与早停、字段编码或模型拟合
- 开发训练耗时：{fit_seconds:.2f}秒；早停选择{best_iterations}轮
- 最终模型关闭早停，使用{best_iterations}轮在全部307,511条已标注样本上重训
- 数据没有绝对申请日期，因此无法进行严格时间外验证；当前结论仅代表同分布随机验证

## 模型设计

- 数值缺失值由HistGradientBoosting原生学习分支方向，不做中位数填补和标准化
- 类别字段使用OrdinalEncoder形成类别编码，并由模型按类别字段处理，不把编码当连续数值解释
- `learning_rate=0.05`，最多400轮，31个叶节点，`min_samples_leaf=50`
- 使用L2正则化和开发集内部早停，降低过拟合风险
- 不使用类别权重，以保留概率含义；类别不平衡通过PR-AUC、KS和风险分层评价

## 固定0.5阈值结果

| 指标 | 数值 |
|---|---:|
| Precision | {metrics["threshold_0_5"]["precision"]:.4f} |
| Recall | {metrics["threshold_0_5"]["recall"]:.4f} |
| F1 | {metrics["threshold_0_5"]["f1"]:.4f} |
| Balanced Accuracy | {metrics["threshold_0_5"]["balanced_accuracy"]:.4f} |

| 真实\\预测 | 预测0 | 预测1 |
|---|---:|---:|
| 真实0 | {confusion["true_negative"]:,} | {confusion["false_positive"]:,} |
| 真实1 | {confusion["false_negative"]:,} | {confusion["true_positive"]:,} |

`0.5`仍然只是展示阈值，不能直接作为授信拒绝阈值。

## 风险十分位

| 风险组 | 人数 | 真实坏样本率 | 平均预测概率 | 相对总体提升倍数 |
|---:|---:|---:|---:|---:|
{decile_rows}

## 置换重要性前20

置换重要性表示随机打乱某个原始字段后ROC-AUC下降多少；下降越多，模型越依赖该字段。

| 原始字段 | 平均AUC下降 | 重复间标准差 |
|---|---:|---:|
{top_importance_rows}

重要性说明模型依赖程度，不代表因果关系；高度相关字段还可能互相替代并分散重要性。
"""
    (REPORT_DIR / "hist_gradient_boosting_report.md").write_text(report, encoding="utf-8")


def write_comparison_report(
    logistic_metrics: dict[str, object],
    hist_metrics: dict[str, object],
    logistic_deciles: pd.DataFrame,
    hist_deciles: pd.DataFrame,
) -> None:
    """Compare two models on the same holdout applicants and report the selection result."""
    rows = []
    for key, label, higher_is_better in [
        ("roc_auc", "ROC-AUC", True),
        ("pr_auc", "PR-AUC", True),
        ("ks", "KS", True),
        ("brier_score", "Brier Score", False),
        ("log_loss", "Log Loss", False),
        ("decile_calibration_error", "十分位校准误差", False),
    ]:
        logistic_value = float(logistic_metrics[key])
        hist_value = float(hist_metrics[key])
        delta = hist_value - logistic_value
        improved = delta > 0 if higher_is_better else delta < 0
        rows.append(
            f"| {label} | {logistic_value:.4f} | {hist_value:.4f} | {delta:+.4f} | "
            f"{'提升' if improved else '下降'} |"
        )
    comparison_rows = "\n".join(rows)
    roc_delta = hist_metrics["roc_auc"] - logistic_metrics["roc_auc"]
    winner = "HistGradientBoosting" if roc_delta > 0 else "Logistic Regression"
    report = f"""# 模型对比报告

## 结论

- 两种模型使用相同验证申请人、相同标签和相同随机种子，结果可以直接比较。
- 按主要排序指标ROC-AUC，当前领先模型是**{winner}**。
- HistGradientBoosting相对Logistic的ROC-AUC变化为{roc_delta:+.4f}。
- Logistic保持线性可解释性；HistGradientBoosting用于捕捉非线性和特征交互。

## 同口径验证指标

| 指标 | Logistic | HistGradientBoosting | HGB-Logistic | HGB表现 |
|---|---:|---:|---:|---|
{comparison_rows}

## 最高风险十分位

| 模型 | 真实坏样本率 | 相对总体提升倍数 |
|---|---:|---:|
| Logistic | {logistic_deciles.iloc[0]["observed_bad_rate"]:.2%} | {logistic_deciles.iloc[0]["bad_rate_lift"]:.2f} |
| HistGradientBoosting | {hist_deciles.iloc[0]["observed_bad_rate"]:.2%} | {hist_deciles.iloc[0]["bad_rate_lift"]:.2f} |

## 使用建议

- 保留Logistic作为可解释、可审计基线。
- 若HistGradientBoosting的排序指标稳定领先，将其作为主要候选模型。
- 如果提升模型校准变差，应先做独立校准，再开展授信阈值和成本模拟。
- 当前数据不能提供严格时间外验证，不能把随机验证成绩直接等同于生产稳定性。
"""
    (REPORT_DIR / "model_comparison.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Train boosting, validate it, refit on all labels, and compare with the baseline."""
    args = parse_args()
    prepare_output_directories(args.overwrite)

    print("读取并清洗DuckDB建模视图", flush=True)
    train, test, numeric_columns, categorical_columns = load_model_frames(DATABASE)
    y = train["TARGET"].astype("int8")
    train_ids = train["SK_ID_CURR"].astype("int64")
    test_ids = test["SK_ID_CURR"].astype("int64")
    features = clean_features(train, numeric_columns, categorical_columns)
    test_features = clean_features(test, numeric_columns, categorical_columns)
    del train, test
    gc.collect()

    split = train_test_split(
        features,
        y,
        train_ids,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_train, x_validation, y_train, y_validation, _, validation_ids = split
    verify_shared_validation_ids(validation_ids)

    print("训练HistGradientBoosting开发集模型", flush=True)
    pipeline = build_hist_gradient_boosting_pipeline(numeric_columns, categorical_columns)
    validation_pipeline = clone(pipeline)
    started = perf_counter()
    validation_pipeline.fit(x_train, y_train)
    fit_seconds = perf_counter() - started
    best_iterations = int(validation_pipeline.named_steps["model"].n_iter_)
    validation_probability = validation_pipeline.predict_proba(x_validation)[:, 1]
    metrics = calculate_binary_metrics(y_validation, validation_probability)
    metrics["validation_positive_rate"] = float(y_validation.mean())
    metrics["train_rows"] = len(x_train)
    metrics["validation_rows"] = len(x_validation)

    validation_predictions = pd.DataFrame(
        {
            "SK_ID_CURR": validation_ids.to_numpy(),
            "TARGET": y_validation.to_numpy(),
            "PREDICTED_PROBABILITY": validation_probability,
        }
    ).sort_values("SK_ID_CURR")
    validation_predictions.to_parquet(
        PROCESSED_DIR / "validation_predictions.parquet",
        index=False,
    )
    deciles = build_risk_deciles(y_validation, validation_probability)
    metrics["decile_calibration_error"] = expected_calibration_error(deciles)
    deciles.to_csv(TABLE_DIR / "validation_risk_deciles.csv", index=False)
    save_diagnostic_figure(
        y_validation,
        validation_probability,
        deciles,
        FIGURE_DIR / "validation_diagnostics.png",
    )

    print("计算验证集置换重要性", flush=True)
    importance = calculate_permutation_importance(
        validation_pipeline,
        x_validation,
        y_validation,
    )
    importance.to_csv(TABLE_DIR / "permutation_importance.csv", index=False)

    logistic_metrics, logistic_predictions, logistic_deciles = load_logistic_results()
    aligned = logistic_predictions.merge(
        validation_predictions,
        on="SK_ID_CURR",
        suffixes=("_LOGISTIC", "_HIST"),
        validate="one_to_one",
    )
    if not np.array_equal(aligned["TARGET_LOGISTIC"], aligned["TARGET_HIST"]):
        raise ValueError("两种模型的验证标签不一致")
    save_model_comparison_figure(
        aligned["TARGET_LOGISTIC"],
        aligned["PREDICTED_PROBABILITY_LOGISTIC"].to_numpy(),
        aligned["PREDICTED_PROBABILITY_HIST"].to_numpy(),
        logistic_deciles,
        deciles,
        ROOT / "reports" / "figures" / "model_comparison.png",
    )

    # Fix the early-stopping iteration count, then refit the final model on all labeled data.
    del validation_pipeline, x_train, x_validation, y_train, y_validation, validation_ids
    gc.collect()
    print(f"使用全部已标注数据重训最终模型，共{best_iterations}轮", flush=True)
    final_pipeline = clone(pipeline).set_params(
        model__early_stopping=False,
        model__max_iter=best_iterations,
    )
    final_pipeline.fit(features, y)
    test_probability = final_pipeline.predict_proba(test_features)[:, 1]
    if not np.isfinite(test_probability).all() or not (
        (test_probability >= 0).all() and (test_probability <= 1).all()
    ):
        raise ValueError("测试集预测概率存在非法值")

    submission = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": test_probability})
    if len(submission) != 48_744 or submission["SK_ID_CURR"].duplicated().any():
        raise ValueError("Kaggle提交文件行数或主键不正确")
    submission.to_csv(PROCESSED_DIR / "submission_hist_gradient_boosting.csv", index=False)
    submission.to_parquet(PROCESSED_DIR / "test_predictions.parquet", index=False)
    joblib.dump(final_pipeline, MODEL_DIR / "model.joblib", compress=3)

    manifest = {
        "model": "HistGradientBoostingClassifier",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "database": str(DATABASE.relative_to(ROOT)),
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "full_training_rows": len(features),
        "test_rows": len(test_features),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "input_feature_count": len(features.columns),
        "best_iterations": best_iterations,
        "validation_fit_seconds": fit_seconds,
        "permutation_sample_size": min(PERMUTATION_SAMPLE_SIZE, len(validation_predictions)),
        "permutation_repeats": PERMUTATION_REPEATS,
        "sklearn_version": sklearn.__version__,
        "metrics": metrics,
    }
    (REPORT_DIR / "hist_gradient_boosting_metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_hist_report(
        metrics,
        deciles,
        importance,
        best_iterations,
        fit_seconds,
        metrics["train_rows"],
        metrics["validation_rows"],
    )
    write_comparison_report(logistic_metrics, metrics, logistic_deciles, deciles)
    print(
        f"完成：ROC-AUC={metrics['roc_auc']:.4f}，PR-AUC={metrics['pr_auc']:.4f}，"
        f"相对Logistic AUC变化={metrics['roc_auc'] - logistic_metrics['roc_auc']:+.4f}",
        flush=True,
    )
    print(
        f"Kaggle提交文件：{PROCESSED_DIR / 'submission_hist_gradient_boosting.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
