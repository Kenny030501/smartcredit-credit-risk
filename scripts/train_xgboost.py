"""Train XGBoost and compare it with logistic and HistGradientBoosting consistently."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from smartcredit.data import clean_features, load_model_frames
from smartcredit.modeling import (
    build_risk_deciles,
    build_xgboost_pipeline,
    calculate_binary_metrics,
    expected_calibration_error,
)
from smartcredit.reporting import save_diagnostic_figure, save_three_model_comparison_figure
from smartcredit.xgboost_runtime import ensure_openmp_runtime

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "interim" / "smartcredit.duckdb"
MODEL_DIR = ROOT / "models" / "xgboost"
PROCESSED_DIR = ROOT / "data" / "processed" / "xgboost"
REPORT_DIR = ROOT / "reports" / "modeling"
TABLE_DIR = ROOT / "reports" / "tables" / "xgboost"
FIGURE_DIR = ROOT / "reports" / "figures" / "xgboost"
COMPARISON_FIGURE = ROOT / "reports" / "figures" / "model_comparison.png"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
EARLY_STOPPING_SIZE = 0.10
PERMUTATION_SAMPLE_SIZE = 5_000
PERMUTATION_REPEATS = 2

EXISTING_MODELS = {
    "Logistic": {
        "metrics": REPORT_DIR / "logistic_baseline_metrics.json",
        "predictions": ROOT
        / "data"
        / "processed"
        / "logistic_baseline"
        / "validation_predictions.parquet",
    },
    "HistGradientBoosting": {
        "metrics": REPORT_DIR / "hist_gradient_boosting_metrics.json",
        "predictions": ROOT
        / "data"
        / "processed"
        / "hist_gradient_boosting"
        / "validation_predictions.parquet",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse runtime arguments; preserve existing XGBoost artifacts by default."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有的XGBoost模型、报告和预测产物",
    )
    return parser.parse_args()


def prepare_output_directories(overwrite: bool) -> None:
    """Create output directories and prevent accidental replacement of a trained model."""
    model_path = MODEL_DIR / "model.joblib"
    if model_path.exists() and not overwrite:
        raise FileExistsError(f"{model_path} 已存在；如需重跑请使用 --overwrite")
    for path in (MODEL_DIR, PROCESSED_DIR, REPORT_DIR, TABLE_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_existing_model_results(
    model_name: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Load frozen validation metrics, applicant probabilities, and risk deciles."""
    paths = EXISTING_MODELS[model_name]
    if not paths["metrics"].exists() or not paths["predictions"].exists():
        raise FileNotFoundError(f"缺少{model_name}的指标或验证集预测")
    manifest = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    predictions = pd.read_parquet(paths["predictions"]).sort_values("SK_ID_CURR")
    deciles = build_risk_deciles(
        predictions["TARGET"],
        predictions["PREDICTED_PROBABILITY"].to_numpy(),
    )
    return manifest["metrics"], predictions, deciles


def verify_shared_validation_ids(
    validation_ids: pd.Series,
    existing_predictions: dict[str, pd.DataFrame],
) -> None:
    """Confirm that all three models use exactly the same outer holdout applicants."""
    expected_ids = np.sort(validation_ids.to_numpy())
    for model_name, predictions in existing_predictions.items():
        actual_ids = np.sort(predictions["SK_ID_CURR"].to_numpy())
        if not np.array_equal(expected_ids, actual_ids):
            raise ValueError(f"XGBoost与{model_name}的验证申请人不一致")


def calculate_feature_importance(
    model_pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Compute XGBoost raw-feature permutation importance on a fixed holdout subsample."""
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


def write_xgboost_report(
    metrics: dict[str, object],
    deciles: pd.DataFrame,
    importance: pd.DataFrame,
    best_iterations: int,
    best_internal_auc: float,
    fit_seconds: float,
    train_count: int,
    early_stopping_count: int,
    validation_count: int,
) -> None:
    """Generate the XGBoost validation, training-design, and interpretation-limit report."""
    confusion = metrics["threshold_0_5"]["confusion_matrix"]
    importance_rows = "\n".join(
        f"| `{row.feature}` | {row.importance_mean:.5f} | {row.importance_std:.5f} |"
        for row in importance.head(20).itertuples()
    )
    decile_rows = "\n".join(
        "| {risk_decile} | {applicant_count:,} | {observed_bad_rate:.2%} | "
        "{average_predicted_probability:.2%} | {bad_rate_lift:.2f} |".format(**row._asdict())
        for row in deciles.itertuples(index=False)
    )
    report = f"""# XGBoost模型报告

## 结论

- 验证集ROC-AUC：{metrics["roc_auc"]:.4f}
- 验证集PR-AUC：{metrics["pr_auc"]:.4f}
- KS：{metrics["ks"]:.4f}
- Brier Score：{metrics["brier_score"]:.4f}，越低越好
- 十分位校准误差：{metrics["decile_calibration_error"]:.4f}，越低越好
- 最高风险十分位真实坏样本率：{deciles.iloc[0]["observed_bad_rate"]:.2%}，为总体的{deciles.iloc[0]["bad_rate_lift"]:.2f}倍

## 验证设计

- XGBoost与Logistic、HistGradientBoosting使用相同的61,503条外层验证申请人
- 80%开发训练集内部再划分90%拟合数据和10%早停数据
- 实际树训练样本：{train_count:,}条；内部早停样本：{early_stopping_count:,}条
- 外层验证样本：{validation_count:,}条，不参与编码、拟合或早停
- 早停最多允许2,000轮，最终选择{best_iterations}轮，内部早停ROC-AUC为{best_internal_auc:.4f}
- 开发模型训练耗时：{fit_seconds:.2f}秒
- 最终模型固定{best_iterations}轮并使用全部307,511条已标注样本重训

## 模型设计

- 数值缺失由XGBoost原生处理，不做中位数填补或标准化
- 类别字段经OrdinalEncoder编码，并通过`feature_types`标记为原生类别字段
- `tree_method='hist'`，`learning_rate=0.03`，`max_depth=4`
- 使用行采样和字段采样各80%，并设置L1和L2正则化
- 不设置`scale_pos_weight`，避免为了分类召回率改变原始概率含义
- 当前参数是规则化起点，只使用Early Stopping选择轮数，尚未执行系统超参数搜索

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

`0.5`只用于展示分类结果，不是Kaggle提交格式或最终授信阈值。

## 风险十分位

| 风险组 | 人数 | 真实坏样本率 | 平均预测概率 | 相对总体提升倍数 |
|---:|---:|---:|---:|---:|
{decile_rows}

## 置换重要性前20

| 原始字段 | 平均AUC下降 | 重复间标准差 |
|---|---:|---:|
{importance_rows}

置换重要性反映模型依赖程度，不代表变量对违约具有因果作用。
"""
    (REPORT_DIR / "xgboost_report.md").write_text(report, encoding="utf-8")


def write_three_model_comparison_report(
    model_metrics: dict[str, dict[str, object]],
    model_deciles: dict[str, pd.DataFrame],
) -> None:
    """Generate a consistent three-model comparison for logistic, HGB, and XGBoost."""
    metric_specs = [
        ("roc_auc", "ROC-AUC", True),
        ("pr_auc", "PR-AUC", True),
        ("ks", "KS", True),
        ("brier_score", "Brier Score", False),
        ("log_loss", "Log Loss", False),
        ("decile_calibration_error", "十分位校准误差", False),
    ]
    metric_rows = []
    for key, label, higher_is_better in metric_specs:
        values = {name: float(metrics[key]) for name, metrics in model_metrics.items()}
        winner = max(values, key=values.get) if higher_is_better else min(values, key=values.get)
        metric_rows.append(
            f"| {label} | {values['Logistic']:.4f} | {values['HistGradientBoosting']:.4f} | "
            f"{values['XGBoost']:.4f} | {winner} |"
        )
    decile_rows = "\n".join(
        f"| {name} | {table.iloc[0]['observed_bad_rate']:.2%} | "
        f"{table.iloc[0]['bad_rate_lift']:.2f} |"
        for name, table in model_deciles.items()
    )
    auc_values = {name: float(metrics["roc_auc"]) for name, metrics in model_metrics.items()}
    primary_model = max(auc_values, key=auc_values.get)
    xgb_vs_hist = auc_values["XGBoost"] - auc_values["HistGradientBoosting"]
    report = f"""# 三模型比较报告

## 结论

- 三种模型使用相同外层验证申请人、标签和评价指标，可以直接比较。
- 按主要排序指标ROC-AUC，当前领先模型是**{primary_model}**。
- XGBoost相对HistGradientBoosting的ROC-AUC变化为{xgb_vs_hist:+.4f}。
- Logistic保留为可解释基线；树提升模型负责捕捉非线性和变量交互。

## 同口径验证指标

| 指标 | Logistic | HistGradientBoosting | XGBoost | 最优模型 |
|---|---:|---:|---:|---|
{chr(10).join(metric_rows)}

## 最高风险十分位

| 模型 | 真实坏样本率 | 相对总体提升倍数 |
|---|---:|---:|
{decile_rows}

## 使用建议

- 主要模型选择不能只看ROC-AUC，还要同时检查PR-AUC、KS、概率校准和业务阈值表现。
- 若XGBoost仅带来很小提升，应权衡新增依赖、部署复杂度和模型治理成本。
- 选定最终候选模型后停止堆模型，进入概率校准与授信策略模拟。
- 数据缺少绝对申请时间，当前结果不代表严格的时间外稳定性。
"""
    (REPORT_DIR / "model_comparison.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Run XGBoost early stopping, holdout validation, full refit, and model comparison."""
    args = parse_args()
    prepare_output_directories(args.overwrite)
    openmp_path = ensure_openmp_runtime()

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
    x_development, x_validation, y_development, y_validation, _, validation_ids = split

    existing_metrics: dict[str, dict[str, object]] = {}
    existing_predictions: dict[str, pd.DataFrame] = {}
    existing_deciles: dict[str, pd.DataFrame] = {}
    for model_name in EXISTING_MODELS:
        metrics, predictions, deciles = load_existing_model_results(model_name)
        existing_metrics[model_name] = metrics
        existing_predictions[model_name] = predictions
        existing_deciles[model_name] = deciles
    verify_shared_validation_ids(validation_ids, existing_predictions)

    internal_split = train_test_split(
        x_development,
        y_development,
        test_size=EARLY_STOPPING_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )
    x_train, x_early_stopping, y_train, y_early_stopping = internal_split

    print("拟合XGBoost预处理并训练内部早停模型", flush=True)
    template = build_xgboost_pipeline(numeric_columns, categorical_columns)
    evaluation_preprocessor = clone(template.named_steps["preprocessor"])
    transformed_train = evaluation_preprocessor.fit_transform(x_train)
    transformed_early_stopping = evaluation_preprocessor.transform(x_early_stopping)
    transformed_validation = evaluation_preprocessor.transform(x_validation)
    evaluation_model = clone(template.named_steps["model"])
    started = perf_counter()
    evaluation_model.fit(
        transformed_train,
        y_train,
        eval_set=[(transformed_early_stopping, y_early_stopping)],
        verbose=False,
    )
    fit_seconds = perf_counter() - started
    best_iterations = int(evaluation_model.best_iteration) + 1
    best_internal_auc = float(evaluation_model.best_score)
    validation_probability = evaluation_model.predict_proba(transformed_validation)[:, 1]
    evaluation_pipeline = Pipeline(
        steps=[
            ("preprocessor", evaluation_preprocessor),
            ("model", evaluation_model),
        ]
    )

    metrics = calculate_binary_metrics(y_validation, validation_probability)
    metrics["validation_positive_rate"] = float(y_validation.mean())
    metrics["train_rows"] = len(x_train)
    metrics["early_stopping_rows"] = len(x_early_stopping)
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

    print("计算XGBoost验证集置换重要性", flush=True)
    importance = calculate_feature_importance(evaluation_pipeline, x_validation, y_validation)
    importance.to_csv(TABLE_DIR / "permutation_importance.csv", index=False)

    aligned = validation_predictions[["SK_ID_CURR", "TARGET", "PREDICTED_PROBABILITY"]].rename(
        columns={
            "TARGET": "TARGET_XGBOOST",
            "PREDICTED_PROBABILITY": "PROBABILITY_XGBOOST",
        }
    )
    for model_name, predictions in existing_predictions.items():
        model_frame = predictions[["SK_ID_CURR", "TARGET", "PREDICTED_PROBABILITY"]].rename(
            columns={
                "TARGET": f"TARGET_{model_name}",
                "PREDICTED_PROBABILITY": f"PROBABILITY_{model_name}",
            }
        )
        aligned = aligned.merge(model_frame, on="SK_ID_CURR", validate="one_to_one")
        if not np.array_equal(aligned["TARGET_XGBOOST"], aligned[f"TARGET_{model_name}"]):
            raise ValueError(f"XGBoost与{model_name}的验证标签不一致")
    save_three_model_comparison_figure(
        aligned["TARGET_XGBOOST"],
        aligned["PROBABILITY_Logistic"].to_numpy(),
        aligned["PROBABILITY_HistGradientBoosting"].to_numpy(),
        aligned["PROBABILITY_XGBOOST"].to_numpy(),
        existing_deciles["Logistic"],
        existing_deciles["HistGradientBoosting"],
        deciles,
        COMPARISON_FIGURE,
    )

    # Fix the best iteration count, refit preprocessing, and train on all labeled data.
    del (
        evaluation_pipeline,
        evaluation_model,
        evaluation_preprocessor,
        transformed_train,
        transformed_early_stopping,
        transformed_validation,
        x_train,
        x_early_stopping,
        y_train,
        y_early_stopping,
        x_development,
        y_development,
        x_validation,
        y_validation,
        validation_ids,
    )
    gc.collect()

    print(f"使用全部已标注数据重训XGBoost，共{best_iterations}轮", flush=True)
    final_preprocessor = clone(template.named_steps["preprocessor"])
    transformed_full = final_preprocessor.fit_transform(features)
    transformed_test = final_preprocessor.transform(test_features)
    final_model = clone(template.named_steps["model"]).set_params(
        n_estimators=best_iterations,
        early_stopping_rounds=None,
    )
    final_model.fit(transformed_full, y, verbose=False)
    test_probability = final_model.predict_proba(transformed_test)[:, 1]
    if not np.isfinite(test_probability).all() or not (
        (test_probability >= 0).all() and (test_probability <= 1).all()
    ):
        raise ValueError("测试集预测概率存在非法值")

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", final_preprocessor),
            ("model", final_model),
        ]
    )
    internal_predictions = pd.DataFrame(
        {
            "SK_ID_CURR": test_ids,
            "PREDICTED_PROBABILITY": test_probability,
        }
    )
    submission = internal_predictions.rename(columns={"PREDICTED_PROBABILITY": "TARGET"})
    if len(submission) != 48_744 or submission["SK_ID_CURR"].duplicated().any():
        raise ValueError("Kaggle提交文件行数或主键不正确")
    internal_predictions.to_parquet(PROCESSED_DIR / "test_predictions.parquet", index=False)
    submission.to_csv(PROCESSED_DIR / "submission_xgboost.csv", index=False)
    joblib.dump(final_pipeline, MODEL_DIR / "model.joblib", compress=3)
    final_model.save_model(MODEL_DIR / "model.ubj")

    manifest = {
        "model": "XGBClassifier",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "database": str(DATABASE.relative_to(ROOT)),
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "early_stopping_size": EARLY_STOPPING_SIZE,
        "full_training_rows": len(features),
        "test_rows": len(test_features),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "input_feature_count": len(features.columns),
        "best_iterations": best_iterations,
        "best_internal_auc": best_internal_auc,
        "validation_fit_seconds": fit_seconds,
        "permutation_sample_size": min(PERMUTATION_SAMPLE_SIZE, len(validation_predictions)),
        "permutation_repeats": PERMUTATION_REPEATS,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": version("xgboost"),
        "openmp_runtime": str(openmp_path) if openmp_path else None,
        "metrics": metrics,
    }
    (REPORT_DIR / "xgboost_metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_xgboost_report(
        metrics,
        deciles,
        importance,
        best_iterations,
        best_internal_auc,
        fit_seconds,
        metrics["train_rows"],
        metrics["early_stopping_rows"],
        metrics["validation_rows"],
    )
    comparison_metrics = {**existing_metrics, "XGBoost": metrics}
    comparison_deciles = {**existing_deciles, "XGBoost": deciles}
    write_three_model_comparison_report(comparison_metrics, comparison_deciles)
    print(
        f"完成：ROC-AUC={metrics['roc_auc']:.4f}，PR-AUC={metrics['pr_auc']:.4f}，"
        f"相对HGB AUC变化={metrics['roc_auc'] - existing_metrics['HistGradientBoosting']['roc_auc']:+.4f}",
        flush=True,
    )
    print(f"Kaggle提交文件：{PROCESSED_DIR / 'submission_xgboost.csv'}", flush=True)


if __name__ == "__main__":
    main()
