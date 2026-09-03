"""Tune XGBoost within the development set and evaluate one candidate on holdout data."""

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from smartcredit.data import clean_features, load_model_frames
from smartcredit.modeling import (
    build_risk_deciles,
    build_xgboost_pipeline,
    calculate_binary_metrics,
    expected_calibration_error,
)
from smartcredit.reporting import save_diagnostic_figure
from smartcredit.xgboost_runtime import ensure_openmp_runtime

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "interim" / "smartcredit.duckdb"
MODEL_DIR = ROOT / "models" / "xgboost_tuned"
PROCESSED_DIR = ROOT / "data" / "processed" / "xgboost_tuned"
REPORT_DIR = ROOT / "reports" / "modeling"
TABLE_DIR = ROOT / "reports" / "tables" / "xgboost_tuning"
FIGURE_DIR = ROOT / "reports" / "figures" / "xgboost_tuned"
BASELINE_METRICS_PATH = REPORT_DIR / "xgboost_metrics.json"
BASELINE_VALIDATION_PATH = (
    ROOT / "data" / "processed" / "xgboost" / "validation_predictions.parquet"
)
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
EARLY_STOPPING_SIZE = 0.10
TUNING_SAMPLE_SIZE = 120_000
CV_FOLDS = 3
RANDOM_CANDIDATES = 9

BASELINE_PARAMS = {
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 10.0,
    "gamma": 0.0,
}

PARAMETER_SPACE = {
    "learning_rate": [0.02, 0.03, 0.05],
    "max_depth": [3, 4, 5],
    "min_child_weight": [10, 20, 40, 80],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [5.0, 10.0, 20.0, 40.0],
    "gamma": [0.0, 0.1, 0.5],
}


def parse_args() -> argparse.Namespace:
    """Parse runtime arguments; preserve existing tuning artifacts by default."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有的XGBoost调优结果、模型和预测文件",
    )
    return parser.parse_args()


def prepare_output_directories(overwrite: bool) -> None:
    """Create tuning output directories and prevent accidental replacement of results."""
    model_path = MODEL_DIR / "model.joblib"
    if model_path.exists() and not overwrite:
        raise FileExistsError(f"{model_path} 已存在；如需重跑请使用 --overwrite")
    for path in (MODEL_DIR, PROCESSED_DIR, REPORT_DIR, TABLE_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def build_candidates() -> list[dict[str, object]]:
    """Sample nine seeded configurations and include the untuned setup as candidate zero."""
    sampled = list(
        ParameterSampler(
            PARAMETER_SPACE,
            n_iter=RANDOM_CANDIDATES,
            random_state=RANDOM_STATE,
        )
    )
    candidates = [BASELINE_PARAMS.copy()]
    for params in sampled:
        if params not in candidates:
            candidates.append(params)
    return candidates


def prepare_fold_cache(
    tuning_features: pd.DataFrame,
    tuning_target: pd.Series,
    template: Pipeline,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Fit preprocessing once per fold to avoid repeatedly encoding the same data."""
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for fold_index, (train_index, validation_index) in enumerate(
        splitter.split(tuning_features, tuning_target),
        start=1,
    ):
        preprocessor = clone(template.named_steps["preprocessor"])
        fold_train = tuning_features.iloc[train_index]
        fold_validation = tuning_features.iloc[validation_index]
        transformed_train = preprocessor.fit_transform(fold_train)
        transformed_validation = preprocessor.transform(fold_validation)
        folds.append(
            (
                transformed_train,
                tuning_target.iloc[train_index].to_numpy(),
                transformed_validation,
                tuning_target.iloc[validation_index].to_numpy(),
            )
        )
        print(f"已缓存第{fold_index}/{CV_FOLDS}折预处理数据", flush=True)
    return folds


def evaluate_candidates(
    candidates: list[dict[str, object]],
    template: Pipeline,
    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Run three-fold early stopping and select one candidate by mean ROC-AUC."""
    trial_rows = []
    for candidate_index, params in enumerate(candidates):
        fold_auc = []
        fold_iterations = []
        started = perf_counter()
        for transformed_train, y_train, transformed_validation, y_validation in folds:
            model = clone(template.named_steps["model"]).set_params(
                **params,
                n_estimators=1_500,
                early_stopping_rounds=40,
            )
            model.fit(
                transformed_train,
                y_train,
                eval_set=[(transformed_validation, y_validation)],
                verbose=False,
            )
            probability = model.predict_proba(transformed_validation)[:, 1]
            fold_auc.append(float(roc_auc_score(y_validation, probability)))
            fold_iterations.append(int(model.best_iteration) + 1)
        elapsed = perf_counter() - started
        row = {
            "candidate_id": candidate_index,
            "is_baseline": candidate_index == 0,
            **params,
            "cv_auc_mean": float(np.mean(fold_auc)),
            "cv_auc_std": float(np.std(fold_auc, ddof=1)),
            "best_iterations_mean": float(np.mean(fold_iterations)),
            "fit_seconds": elapsed,
        }
        trial_rows.append(row)
        pd.DataFrame(trial_rows).to_csv(TABLE_DIR / "xgboost_tuning_trials.csv", index=False)
        print(
            f"候选{candidate_index + 1}/{len(candidates)}："
            f"CV AUC={row['cv_auc_mean']:.5f}±{row['cv_auc_std']:.5f}，"
            f"平均轮数={row['best_iterations_mean']:.0f}",
            flush=True,
        )
    return pd.DataFrame(trial_rows).sort_values(
        ["cv_auc_mean", "cv_auc_std"],
        ascending=[False, True],
    )


def selected_parameters(best_row: pd.Series) -> dict[str, object]:
    """Extract XGBoost-compatible model parameters from the tuning result."""
    return {
        key: best_row[key].item() if hasattr(best_row[key], "item") else best_row[key]
        for key in BASELINE_PARAMS
    }


def write_tuning_report(
    trials: pd.DataFrame,
    selected_params: dict[str, object],
    tuned_metrics: dict[str, object],
    baseline_metrics: dict[str, object],
    best_iterations: int,
    outer_fit_seconds: float,
) -> None:
    """Record the search budget, candidates, holdout delta, and final selection."""
    trial_rows = "\n".join(
        "| {candidate_id} | {learning_rate:.2f} | {max_depth:.0f} | {min_child_weight:.0f} | "
        "{subsample:.1f} | {colsample_bytree:.1f} | {reg_alpha:.1f} | {reg_lambda:.1f} | "
        "{gamma:.1f} | {cv_auc_mean:.5f} | {cv_auc_std:.5f} |".format(**row._asdict())
        for row in trials.itertuples(index=False)
    )
    parameter_rows = "\n".join(f"| `{key}` | {value} |" for key, value in selected_params.items())
    auc_delta = tuned_metrics["roc_auc"] - baseline_metrics["roc_auc"]
    pr_delta = tuned_metrics["pr_auc"] - baseline_metrics["pr_auc"]
    champion = "调优XGBoost" if auc_delta > 0 else "原始XGBoost"
    report = f"""# XGBoost受控超参数调优报告

## 结论

- 开发集内部三折CV选择的参数在外层验证集ROC-AUC为{tuned_metrics["roc_auc"]:.4f}。
- 相对原始XGBoost，ROC-AUC变化为{auc_delta:+.4f}，PR-AUC变化为{pr_delta:+.4f}。
- 当前保留的主要候选模型是**{champion}**。
- 调优模型的最终Early Stopping轮数为{best_iterations}，外层开发模型训练耗时{outer_fit_seconds:.2f}秒。

## 验证结构

- 外层20%验证集共61,503人，不参与参数搜索。
- 参数搜索只使用80%开发集中的固定120,000人分层样本。
- 使用3折StratifiedKFold，比较1组原始参数和{len(trials) - 1}组固定随机参数。
- 每折最多1,500轮，连续40轮AUC不改善则早停。
- 只选择CV平均AUC最高的一组参数进入外层验证，避免按外层结果挑参数。

## 选中参数

| 参数 | 数值 |
|---|---:|
{parameter_rows}

## 搜索结果

| ID | 学习率 | 深度 | 最小子节点权重 | 行采样 | 列采样 | L1 | L2 | Gamma | CV AUC | 标准差 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{trial_rows}

## 外层验证比较

| 指标 | 原始XGBoost | 调优XGBoost | 变化 |
|---|---:|---:|---:|
| ROC-AUC | {baseline_metrics["roc_auc"]:.4f} | {tuned_metrics["roc_auc"]:.4f} | {auc_delta:+.4f} |
| PR-AUC | {baseline_metrics["pr_auc"]:.4f} | {tuned_metrics["pr_auc"]:.4f} | {pr_delta:+.4f} |
| KS | {baseline_metrics["ks"]:.4f} | {tuned_metrics["ks"]:.4f} | {tuned_metrics["ks"] - baseline_metrics["ks"]:+.4f} |
| Brier Score | {baseline_metrics["brier_score"]:.4f} | {tuned_metrics["brier_score"]:.4f} | {tuned_metrics["brier_score"] - baseline_metrics["brier_score"]:+.4f} |
| 校准误差 | {baseline_metrics["decile_calibration_error"]:.4f} | {tuned_metrics["decile_calibration_error"]:.4f} | {tuned_metrics["decile_calibration_error"] - baseline_metrics["decile_calibration_error"]:+.4f} |

## 解释边界

- 这是有限预算调优，不是对全部参数组合的穷举搜索。
- 调优样本来自同分布随机划分，无法替代时间外稳定性检验。
- 如果AUC增量很小，应优先考虑特征工程、概率校准和授信策略，而不是继续扩大搜索预算。
"""
    (REPORT_DIR / "xgboost_tuning_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Tune within development data, validate once, refit fully, and save artifacts."""
    args = parse_args()
    prepare_output_directories(args.overwrite)
    openmp_path = ensure_openmp_runtime()

    if not BASELINE_METRICS_PATH.exists() or not BASELINE_VALIDATION_PATH.exists():
        raise FileNotFoundError("缺少原始XGBoost指标或验证集预测")
    baseline_manifest = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    baseline_metrics = baseline_manifest["metrics"]
    baseline_validation = pd.read_parquet(BASELINE_VALIDATION_PATH)

    print("读取并清洗建模数据", flush=True)
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
    if not np.array_equal(
        np.sort(validation_ids.to_numpy()),
        np.sort(baseline_validation["SK_ID_CURR"].to_numpy()),
    ):
        raise ValueError("调优模型与原始XGBoost的外层验证申请人不一致")

    tuning_features, _, tuning_target, _ = train_test_split(
        x_development,
        y_development,
        train_size=TUNING_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )
    template = build_xgboost_pipeline(numeric_columns, categorical_columns)
    print("缓存三折预处理结果", flush=True)
    folds = prepare_fold_cache(tuning_features, tuning_target, template)
    candidates = build_candidates()
    print(f"开始评估{len(candidates)}组参数", flush=True)
    trials = evaluate_candidates(candidates, template, folds)
    trials.to_csv(TABLE_DIR / "xgboost_tuning_trials.csv", index=False)
    best_row = trials.iloc[0]
    params = selected_parameters(best_row)
    del folds, tuning_features, tuning_target
    gc.collect()

    print(f"使用CV最优候选进行唯一一次外层验证：{params}", flush=True)
    x_train, x_early, y_train, y_early = train_test_split(
        x_development,
        y_development,
        test_size=EARLY_STOPPING_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )
    evaluation_preprocessor = clone(template.named_steps["preprocessor"])
    transformed_train = evaluation_preprocessor.fit_transform(x_train)
    transformed_early = evaluation_preprocessor.transform(x_early)
    transformed_validation = evaluation_preprocessor.transform(x_validation)
    evaluation_model = clone(template.named_steps["model"]).set_params(**params)
    started = perf_counter()
    evaluation_model.fit(
        transformed_train,
        y_train,
        eval_set=[(transformed_early, y_early)],
        verbose=False,
    )
    outer_fit_seconds = perf_counter() - started
    best_iterations = int(evaluation_model.best_iteration) + 1
    validation_probability = evaluation_model.predict_proba(transformed_validation)[:, 1]
    metrics = calculate_binary_metrics(y_validation, validation_probability)
    metrics["validation_positive_rate"] = float(y_validation.mean())
    metrics["train_rows"] = len(x_train)
    metrics["early_stopping_rows"] = len(x_early)
    metrics["validation_rows"] = len(x_validation)
    deciles = build_risk_deciles(y_validation, validation_probability)
    metrics["decile_calibration_error"] = expected_calibration_error(deciles)

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
    deciles.to_csv(TABLE_DIR / "validation_risk_deciles.csv", index=False)
    save_diagnostic_figure(
        y_validation,
        validation_probability,
        deciles,
        FIGURE_DIR / "validation_diagnostics.png",
    )

    del (
        evaluation_model,
        evaluation_preprocessor,
        transformed_train,
        transformed_early,
        transformed_validation,
        x_train,
        x_early,
        y_train,
        y_early,
        x_development,
        y_development,
        x_validation,
        y_validation,
        validation_ids,
    )
    gc.collect()

    print(f"使用全部已标注数据重训调优XGBoost，共{best_iterations}轮", flush=True)
    final_preprocessor = clone(template.named_steps["preprocessor"])
    transformed_full = final_preprocessor.fit_transform(features)
    transformed_test = final_preprocessor.transform(test_features)
    final_model = clone(template.named_steps["model"]).set_params(
        **params,
        n_estimators=best_iterations,
        early_stopping_rounds=None,
    )
    final_model.fit(transformed_full, y, verbose=False)
    test_probability = final_model.predict_proba(transformed_test)[:, 1]
    if not np.isfinite(test_probability).all() or not (
        (test_probability >= 0).all() and (test_probability <= 1).all()
    ):
        raise ValueError("测试集预测概率存在非法值")

    final_pipeline = Pipeline(steps=[("preprocessor", final_preprocessor), ("model", final_model)])
    test_predictions = pd.DataFrame(
        {"SK_ID_CURR": test_ids, "PREDICTED_PROBABILITY": test_probability}
    )
    submission = test_predictions.rename(columns={"PREDICTED_PROBABILITY": "TARGET"})
    if len(submission) != 48_744 or submission["SK_ID_CURR"].duplicated().any():
        raise ValueError("Kaggle提交文件行数或主键不正确")
    test_predictions.to_parquet(PROCESSED_DIR / "test_predictions.parquet", index=False)
    submission.to_csv(PROCESSED_DIR / "submission_xgboost_tuned.csv", index=False)
    joblib.dump(final_pipeline, MODEL_DIR / "model.joblib", compress=3)
    final_model.save_model(MODEL_DIR / "model.ubj")

    manifest = {
        "model": "XGBClassifierTuned",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "database": str(DATABASE.relative_to(ROOT)),
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "tuning_sample_size": TUNING_SAMPLE_SIZE,
        "cv_folds": CV_FOLDS,
        "candidate_count": len(candidates),
        "selected_params": params,
        "selected_cv_auc_mean": float(best_row["cv_auc_mean"]),
        "selected_cv_auc_std": float(best_row["cv_auc_std"]),
        "best_iterations": best_iterations,
        "outer_fit_seconds": outer_fit_seconds,
        "full_training_rows": len(features),
        "test_rows": len(test_features),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": version("xgboost"),
        "openmp_runtime": str(openmp_path) if openmp_path else None,
        "metrics": metrics,
    }
    (REPORT_DIR / "xgboost_tuned_metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_tuning_report(
        trials,
        params,
        metrics,
        baseline_metrics,
        best_iterations,
        outer_fit_seconds,
    )
    print(
        f"完成：调优XGBoost ROC-AUC={metrics['roc_auc']:.4f}，"
        f"相对原始XGBoost变化={metrics['roc_auc'] - baseline_metrics['roc_auc']:+.4f}",
        flush=True,
    )
    print(f"Kaggle提交文件：{PROCESSED_DIR / 'submission_xgboost_tuned.csv'}", flush=True)


if __name__ == "__main__":
    main()
