"""Train an interpretable logistic baseline and generate validation and Kaggle outputs."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from smartcredit.data import clean_features, load_model_frames
from smartcredit.modeling import (
    build_logistic_pipeline,
    build_risk_deciles,
    calculate_binary_metrics,
    expected_calibration_error,
)
from smartcredit.reporting import save_diagnostic_figure

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "interim" / "smartcredit.duckdb"
MODEL_DIR = ROOT / "models" / "logistic_baseline"
PROCESSED_DIR = ROOT / "data" / "processed" / "logistic_baseline"
REPORT_DIR = ROOT / "reports" / "modeling"
TABLE_DIR = ROOT / "reports" / "tables" / "logistic_baseline"
FIGURE_DIR = ROOT / "reports" / "figures" / "logistic_baseline"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def parse_args() -> argparse.Namespace:
    """Parse runtime arguments; existing artifacts are preserved by default."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有的Logistic Regression基线产物",
    )
    return parser.parse_args()


def prepare_output_directories(overwrite: bool) -> None:
    """Create output directories and prevent accidental replacement of a trained model."""
    model_path = MODEL_DIR / "model.joblib"
    if model_path.exists() and not overwrite:
        raise FileExistsError(f"{model_path} 已存在；如需重跑请使用 --overwrite")
    for path in (MODEL_DIR, PROCESSED_DIR, REPORT_DIR, TABLE_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def coefficient_table(model_pipeline: object) -> pd.DataFrame:
    """Extract transformed feature names and coefficients for interpretation and audit."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]
    if len(feature_names) != len(coefficients):
        raise ValueError("变换后字段名与模型系数数量不一致")
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        .assign(absolute_coefficient=lambda frame: frame["coefficient"].abs())
        .sort_values("absolute_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def write_report(
    metrics: dict[str, object],
    deciles: pd.DataFrame,
    coefficients: pd.DataFrame,
    train_count: int,
    validation_count: int,
    numeric_count: int,
    categorical_count: int,
) -> None:
    """Write the training design, metrics, risk tiers, and interpretation limits in Chinese."""
    confusion = metrics["threshold_0_5"]["confusion_matrix"]
    top_positive = coefficients.sort_values("coefficient", ascending=False).head(10)
    top_negative = coefficients.sort_values("coefficient", ascending=True).head(10)
    positive_rows = "\n".join(
        f"| `{row.feature}` | {row.coefficient:.4f} |" for row in top_positive.itertuples()
    )
    negative_rows = "\n".join(
        f"| `{row.feature}` | {row.coefficient:.4f} |" for row in top_negative.itertuples()
    )
    decile_rows = "\n".join(
        "| {risk_decile} | {applicant_count:,} | {observed_bad_rate:.2%} | "
        "{average_predicted_probability:.2%} | {bad_rate_lift:.2f} |".format(**row._asdict())
        for row in deciles.itertuples(index=False)
    )

    report = f"""# Logistic Regression基线报告

## 结论

- 验证集ROC-AUC：{metrics["roc_auc"]:.4f}
- 验证集PR-AUC：{metrics["pr_auc"]:.4f}；验证集正例率为{metrics["validation_positive_rate"]:.2%}
- KS：{metrics["ks"]:.4f}
- Brier Score：{metrics["brier_score"]:.4f}，越低越好
- 十分位校准误差：{metrics["decile_calibration_error"]:.4f}，越低越好
- 最高风险十分位的真实坏样本率为{deciles.iloc[0]["observed_bad_rate"]:.2%}，是总体的{deciles.iloc[0]["bad_rate_lift"]:.2f}倍

## 验证设计

- 数据源：`mart.feature_matrix_train`
- 开发训练集：{train_count:,}行
- 独立验证集：{validation_count:,}行
- 划分方式：80/20分层随机划分，`random_state=42`
- 数值字段：{numeric_count}个；类别字段：{categorical_count}个
- `SK_ID_CURR`只用于关联和提交，不进入模型
- 数值缺失值、中位数、标准化参数和类别编码均只在开发训练集拟合，再应用到验证集
- 完成验证后，另用全部307,511条已标注样本重训最终模型，并预测Kaggle测试集

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

`0.5`只是固定演示阈值，不是授信阈值。后续策略模拟应根据风险成本、审批率和业务约束选择阈值。

## 风险十分位

风险组1代表预测风险最高的10%申请人。

| 风险组 | 人数 | 真实坏样本率 | 平均预测概率 | 相对总体提升倍数 |
|---:|---:|---:|---:|---:|
{decile_rows}

## 系数绝对值较大的特征

正系数与更高预测风险相关，负系数与更低预测风险相关；系数是模型关联，不代表因果关系。

### 正系数前10

| 变换后特征 | 系数 |
|---|---:|
{positive_rows}

### 负系数前10

| 变换后特征 | 系数 |
|---|---:|
{negative_rows}

## 概率解释边界

- 本基线不使用`class_weight='balanced'`，因为人为改变类别权重会使输出概率偏离样本中的真实坏样本率。
- 类别不平衡通过ROC-AUC、PR-AUC、KS和风险分层评价，而不是只看Accuracy。
- `TARGET`是Home Credit比赛定义的还款困难标签，不等同于监管违约口径。
- 随机验证只能评估同分布泛化；真实生产风控还需要时间外验证、稳定性监控和公平性审查。
"""
    (REPORT_DIR / "logistic_baseline_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Load data, run holdout validation, refit on all labels, and validate artifacts."""
    args = parse_args()
    prepare_output_directories(args.overwrite)

    print("读取DuckDB建模视图", flush=True)
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

    print("训练开发集模型并评估独立验证集", flush=True)
    pipeline = build_logistic_pipeline(numeric_columns, categorical_columns)
    validation_pipeline = clone(pipeline)
    validation_pipeline.fit(x_train, y_train)
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

    # Release development objects before fitting the submission model on all labeled data.
    del validation_pipeline, x_train, x_validation, y_train, y_validation, validation_ids
    gc.collect()

    print("使用全部已标注数据重训最终模型", flush=True)
    pipeline.fit(features, y)
    test_probability = pipeline.predict_proba(test_features)[:, 1]
    if not np.isfinite(test_probability).all() or not (
        (test_probability >= 0).all() and (test_probability <= 1).all()
    ):
        raise ValueError("测试集预测概率存在非法值")

    submission = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": test_probability})
    if len(submission) != 48_744 or submission["SK_ID_CURR"].duplicated().any():
        raise ValueError("Kaggle提交文件行数或主键不正确")
    submission.to_csv(PROCESSED_DIR / "submission_logistic_baseline.csv", index=False)
    submission.to_parquet(PROCESSED_DIR / "test_predictions.parquet", index=False)

    coefficients = coefficient_table(pipeline)
    coefficients.to_csv(TABLE_DIR / "logistic_coefficients.csv", index=False)
    joblib.dump(pipeline, MODEL_DIR / "model.joblib", compress=3)

    manifest = {
        "model": "LogisticRegression",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "database": str(DATABASE.relative_to(ROOT)),
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "full_training_rows": len(features),
        "test_rows": len(test_features),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "transformed_feature_count": len(coefficients),
        "sklearn_version": sklearn.__version__,
        "metrics": metrics,
    }
    (REPORT_DIR / "logistic_baseline_metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        metrics=metrics,
        deciles=deciles,
        coefficients=coefficients,
        train_count=metrics["train_rows"],
        validation_count=metrics["validation_rows"],
        numeric_count=len(numeric_columns),
        categorical_count=len(categorical_columns),
    )
    print(f"完成：ROC-AUC={metrics['roc_auc']:.4f}，PR-AUC={metrics['pr_auc']:.4f}", flush=True)
    print(f"Kaggle提交文件：{PROCESSED_DIR / 'submission_logistic_baseline.csv'}", flush=True)


if __name__ == "__main__":
    main()
