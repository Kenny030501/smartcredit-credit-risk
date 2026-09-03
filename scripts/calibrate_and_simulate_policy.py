"""Calibrate XGBoost with development OOF predictions and simulate credit policies."""

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
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from smartcredit.calibration import (
    apply_calibrator,
    fit_calibrators,
    select_calibration_method,
)
from smartcredit.data import clean_features, load_model_frames
from smartcredit.modeling import (
    build_risk_deciles,
    build_xgboost_pipeline,
    calculate_binary_metrics,
    expected_calibration_error,
)
from smartcredit.policy import (
    build_rejection_capacity_table,
    build_three_band_table,
    build_threshold_policy_table,
)
from smartcredit.reporting import (
    save_calibration_comparison_figure,
    save_diagnostic_figure,
    save_policy_tradeoff_figure,
)
from smartcredit.xgboost_runtime import ensure_openmp_runtime

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "interim" / "smartcredit.duckdb"
TUNED_METRICS = ROOT / "reports" / "modeling" / "xgboost_tuned_metrics.json"
TUNED_VALIDATION = ROOT / "data" / "processed" / "xgboost_tuned" / "validation_predictions.parquet"
TUNED_TEST = ROOT / "data" / "processed" / "xgboost_tuned" / "test_predictions.parquet"
MODEL_DIR = ROOT / "models" / "calibration"
PROCESSED_DIR = ROOT / "data" / "processed" / "calibration_policy"
REPORT_DIR = ROOT / "reports" / "modeling"
TABLE_DIR = ROOT / "reports" / "tables" / "calibration_policy"
FIGURE_DIR = ROOT / "reports" / "figures" / "calibration_policy"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
OOF_FOLDS = 3
EARLY_STOPPING_SIZE = 0.10
CALIBRATION_SELECTION_FOLDS = 5


def parse_args() -> argparse.Namespace:
    """Parse the overwrite flag and explicitly labeled economic assumptions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有校准和策略产物")
    parser.add_argument(
        "--reuse-oof",
        action="store_true",
        help="复用已生成的开发集OOF概率和折结果，跳过三次XGBoost训练",
    )
    parser.add_argument(
        "--performing-margin",
        type=float,
        default=0.08,
        help="正常贷款单位风险暴露的净贡献率假设，默认8%",
    )
    parser.add_argument(
        "--loss-given-default",
        type=float,
        default=0.50,
        help="坏样本单位风险暴露损失率假设，默认50%",
    )
    return parser.parse_args()


def prepare_output_directories(overwrite: bool) -> None:
    """Create output directories and prevent accidental calibrator replacement."""
    calibrator_path = MODEL_DIR / "calibrators.joblib"
    if calibrator_path.exists() and not overwrite:
        raise FileExistsError(f"{calibrator_path} 已存在；如需重跑请使用 --overwrite")
    for path in (MODEL_DIR, PROCESSED_DIR, REPORT_DIR, TABLE_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def generate_development_oof_probability(
    features: pd.DataFrame,
    target: pd.Series,
    template: object,
    selected_params: dict[str, object],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate three-fold out-of-fold development probabilities for calibration."""
    splitter = StratifiedKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probability = np.full(len(features), np.nan, dtype=float)
    fold_rows = []

    for fold, (fold_train_index, fold_validation_index) in enumerate(
        splitter.split(features, target),
        start=1,
    ):
        fold_train = features.iloc[fold_train_index]
        fold_target = target.iloc[fold_train_index]
        fold_validation = features.iloc[fold_validation_index]
        fold_validation_target = target.iloc[fold_validation_index]
        x_train, x_early, y_train, y_early = train_test_split(
            fold_train,
            fold_target,
            test_size=EARLY_STOPPING_SIZE,
            random_state=RANDOM_STATE + fold,
            stratify=fold_target,
        )

        preprocessor = clone(template.named_steps["preprocessor"])
        transformed_train = preprocessor.fit_transform(x_train)
        transformed_early = preprocessor.transform(x_early)
        transformed_validation = preprocessor.transform(fold_validation)
        model = clone(template.named_steps["model"]).set_params(**selected_params)
        started = perf_counter()
        model.fit(
            transformed_train,
            y_train,
            eval_set=[(transformed_early, y_early)],
            verbose=False,
        )
        probability = model.predict_proba(transformed_validation)[:, 1]
        elapsed = perf_counter() - started
        oof_probability[fold_validation_index] = probability
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(x_train),
                "early_stopping_rows": len(x_early),
                "validation_rows": len(fold_validation),
                "best_iterations": int(model.best_iteration) + 1,
                "fold_roc_auc": float(roc_auc_score(fold_validation_target, probability)),
                "fit_seconds": elapsed,
            }
        )
        print(
            f"OOF第{fold}/{OOF_FOLDS}折：AUC={fold_rows[-1]['fold_roc_auc']:.4f}，"
            f"轮数={fold_rows[-1]['best_iterations']}",
            flush=True,
        )
        del (
            preprocessor,
            model,
            transformed_train,
            transformed_early,
            transformed_validation,
            x_train,
            x_early,
            y_train,
            y_early,
            fold_train,
            fold_validation,
        )
        gc.collect()

    if not np.isfinite(oof_probability).all():
        raise ValueError("开发集OOF概率未覆盖全部样本")
    return oof_probability, pd.DataFrame(fold_rows)


def evaluate_calibration_methods(
    target: pd.Series,
    raw_probability: np.ndarray,
    calibrators: dict[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """Compare raw, Platt, and isotonic probabilities on the outer holdout set."""
    metrics = {}
    probabilities = {}
    deciles = {}
    for method in ("raw", "platt", "isotonic"):
        probability = apply_calibrator(method, calibrators, raw_probability)
        method_metrics = calculate_binary_metrics(target, probability)
        method_deciles = build_risk_deciles(target, probability)
        method_metrics["decile_calibration_error"] = expected_calibration_error(method_deciles)
        probabilities[method] = probability
        metrics[method] = method_metrics
        deciles[method] = method_deciles
    return metrics, probabilities, deciles


def generate_cross_validated_calibration_probabilities(
    target: pd.Series,
    raw_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    """Cross-fit calibrators on development OOF predictions to reduce split noise."""
    target_array = np.asarray(target, dtype=int)
    raw_array = np.asarray(raw_probability, dtype=float)
    probabilities = {
        "raw": raw_array.copy(),
        "platt": np.full(len(raw_array), np.nan, dtype=float),
        "isotonic": np.full(len(raw_array), np.nan, dtype=float),
    }
    splitter = StratifiedKFold(
        n_splits=CALIBRATION_SELECTION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    for fit_index, validation_index in splitter.split(raw_array, target_array):
        calibrators = fit_calibrators(target_array[fit_index], raw_array[fit_index])
        for method in ("platt", "isotonic"):
            probabilities[method][validation_index] = apply_calibrator(
                method,
                calibrators,
                raw_array[validation_index],
            )
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise ValueError("交叉校准概率未覆盖全部开发样本")
    return probabilities


def calculate_calibration_metrics(
    target: pd.Series,
    probabilities: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    """Compute consistent ranking and calibration metrics for all probability variants."""
    metrics = {}
    for method, probability in probabilities.items():
        method_metrics = calculate_binary_metrics(target, probability)
        method_deciles = build_risk_deciles(target, probability)
        method_metrics["decile_calibration_error"] = expected_calibration_error(method_deciles)
        metrics[method] = method_metrics
    return metrics


def write_report(
    selection_metrics: dict[str, dict[str, object]],
    calibration_metrics: dict[str, dict[str, object]],
    selected_method: str,
    fold_results: pd.DataFrame,
    capacity_table: pd.DataFrame,
    band_table: pd.DataFrame,
    performing_margin: float,
    loss_given_default: float,
) -> None:
    """Write leakage-safe calibration results, policy tradeoffs, and assumptions in Chinese."""
    method_names = {"raw": "原始概率", "platt": "Platt", "isotonic": "Isotonic"}
    selection_rows = "\n".join(
        f"| {method_names[method]} | {values['roc_auc']:.4f} | {values['brier_score']:.5f} | "
        f"{values['log_loss']:.5f} | {values['decile_calibration_error']:.5f} |"
        for method, values in selection_metrics.items()
    )
    outer_rows = "\n".join(
        f"| {method_names[method]} | {values['roc_auc']:.4f} | {values['brier_score']:.5f} | "
        f"{values['log_loss']:.5f} | {values['decile_calibration_error']:.5f} |"
        for method, values in calibration_metrics.items()
    )
    fold_rows = "\n".join(
        f"| {int(row.fold)} | {int(row.train_rows):,} | {int(row.early_stopping_rows):,} | "
        f"{int(row.validation_rows):,} | {int(row.best_iterations):,} | "
        f"{row.fold_roc_auc:.4f} | {row.fit_seconds:.2f}秒 |"
        for row in fold_results.itertuples()
    )
    capacity_rows = "\n".join(
        f"| {row.rejection_rate:.0%} | {row.threshold:.4f} | {row.approval_rate:.0%} | "
        f"{row.approved_bad_rate:.2%} | {row.bad_capture_rate:.2%} | "
        f"{int(row.good_rejection_count):,} | {row.normalized_value_per_applicant:.4f} |"
        for row in capacity_table.itertuples()
    )
    band_rows = "\n".join(
        f"| {row.risk_band} | {int(row.applicant_count):,} | {row.population_share:.0%} | "
        f"{row.observed_bad_rate:.2%} | {row.average_predicted_probability:.2%} |"
        for row in band_table.itertuples()
    )
    best_capacity = capacity_table.loc[capacity_table["normalized_value_per_applicant"].idxmax()]
    report = f"""# 概率校准与授信策略模拟报告

## 结论

- 校准器和方法选择都只使用开发集三折OOF概率，外层61,503人不参与拟合或选择。
- 开发集内部要求Brier至少改善0.00001且Log Loss不恶化，否则保留原始概率；当前方法为**{method_names[selected_method]}**。
- 风险最高10%客群仍然集中约30%的真实坏样本，模型具备策略分层价值。
- 在“正常贷款净贡献{performing_margin:.0%}、坏样本损失{loss_given_default:.0%}、单位EAD=1”的假设下，表内最高归一化价值对应拒绝约{best_capacity.rejection_rate:.0%}；这不是银行真实最优阈值。

## 一、无泄漏校准设计

1. 固定原来的80/20外层划分。
2. 在80%开发集内部做3折交叉拟合；每个申请人的OOF概率都来自未见过该申请人的模型。
3. 在开发集OOF概率上再做{CALIBRATION_SELECTION_FOLDS}折交叉校准，比较样本外Brier和Log Loss并确定方法。
4. 用全部开发集OOF概率重拟合校准器，再到外层验证集评价预先选定的方法。

### 开发集内部方法选择

| 方法 | ROC-AUC | Brier | Log Loss | 十分位校准误差 |
|---|---:|---:|---:|---:|
{selection_rows}

### 外层验证结果

| 方法 | ROC-AUC | Brier | Log Loss | 十分位校准误差 |
|---|---:|---:|---:|---:|
{outer_rows}

ROC-AUC主要评价排序，校准选择以Brier和Log Loss为主；Isotonic可能因产生相同概率而轻微改变排序。

### OOF训练记录

| 折 | 训练 | 早停 | OOF验证 | 最佳轮数 | AUC | 耗时 |
|---:|---:|---:|---:|---:|---:|---:|
{fold_rows}

## 二、固定拒绝容量策略

按预测风险从高到低拒绝固定比例，比固定概率阈值更容易对应人工审核或拒绝容量。

| 拒绝率 | 对应阈值 | 审批率 | 审批客群坏样本率 | 坏样本捕获率 | 误拒正常客户 | 单位申请价值 |
|---:|---:|---:|---:|---:|---:|---:|
{capacity_rows}

## 三、三段式授信示例

本示例仅用于展示：风险最高10%为拒绝候选，随后20%进入人工审核，其余70%自动批准。

| 风险带 | 人数 | 占比 | 真实坏样本率 | 平均预测概率 |
|---|---:|---:|---:|---:|
{band_rows}

## 四、经济情景假设与边界

归一化价值计算：

```text
单位申请价值 = 批准正常客户占比 × {performing_margin:.0%}
             - 批准坏样本占比 × {loss_given_default:.0%}
```

- 假设每笔贷款EAD均为1，拒绝后不产生收入、损失或人工审核成本。
- 数据没有真实利率、期限、资金成本、LGD、回收率和运营成本。
- “误拒正常客户”只按`TARGET=0`统计，不代表客户一定具有正利润。
- 因此经济结果只用于阈值敏感性展示，不能用于真实授信决策。

## 五、下一步

1. 在Streamlit中提供拒绝率、概率阈值、正常贷款贡献率和LGD滑块。
2. 同时显示审批率、审批客群坏样本率、风险捕获率和误拒数量。
3. 保留原始概率与校准概率切换，并展示模型和策略的证据边界。
4. Kaggle仍提交原始调优XGBoost连续概率，因为比赛评价ROC-AUC而不是业务阈值。
"""
    (REPORT_DIR / "calibration_policy_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Generate development OOF predictions, select calibration, and simulate holdout policy."""
    args = parse_args()
    if not 0 <= args.performing_margin <= 1 or not 0 <= args.loss_given_default <= 1:
        raise ValueError("经济情景参数必须位于0到1之间")
    prepare_output_directories(args.overwrite)
    openmp_path = ensure_openmp_runtime()

    if not TUNED_METRICS.exists() or not TUNED_VALIDATION.exists() or not TUNED_TEST.exists():
        raise FileNotFoundError("缺少调优XGBoost指标或预测文件")
    tuned_manifest = json.loads(TUNED_METRICS.read_text(encoding="utf-8"))
    selected_params = tuned_manifest["selected_params"]
    outer_predictions = pd.read_parquet(TUNED_VALIDATION).sort_values("SK_ID_CURR")
    tuned_test = pd.read_parquet(TUNED_TEST).sort_values("SK_ID_CURR")

    print("读取建模数据并恢复固定外层划分", flush=True)
    train, test, numeric_columns, categorical_columns = load_model_frames(DATABASE)
    y = train["TARGET"].astype("int8")
    train_ids = train["SK_ID_CURR"].astype("int64")
    features = clean_features(train, numeric_columns, categorical_columns)
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
    x_development, _, y_development, _, development_ids, validation_ids = split
    if not np.array_equal(
        np.sort(validation_ids.to_numpy()),
        outer_predictions["SK_ID_CURR"].to_numpy(),
    ):
        raise ValueError("校准阶段外层验证申请人与调优阶段不一致")

    template = build_xgboost_pipeline(numeric_columns, categorical_columns)
    oof_path = PROCESSED_DIR / "development_oof_predictions.parquet"
    fold_path = TABLE_DIR / "calibration_oof_folds.csv"
    if args.reuse_oof:
        if not oof_path.exists() or not fold_path.exists():
            raise FileNotFoundError("--reuse-oof需要已有OOF概率和折结果")
        print("复用已有开发集三折OOF概率", flush=True)
        oof_predictions = pd.read_parquet(oof_path).sort_values("SK_ID_CURR")
        fold_results = pd.read_csv(fold_path)
        if not np.array_equal(
            oof_predictions["SK_ID_CURR"].to_numpy(),
            np.sort(development_ids.to_numpy()),
        ):
            raise ValueError("已有OOF概率与当前开发集申请人不一致")
        oof_probability_by_id = oof_predictions.set_index("SK_ID_CURR")["OOF_PROBABILITY"]
        oof_probability = development_ids.map(oof_probability_by_id).to_numpy()
    else:
        print("生成开发集三折OOF概率", flush=True)
        oof_probability, fold_results = generate_development_oof_probability(
            x_development,
            y_development,
            template,
            selected_params,
        )
        oof_predictions = pd.DataFrame(
            {
                "SK_ID_CURR": development_ids.to_numpy(),
                "TARGET": y_development.to_numpy(),
                "OOF_PROBABILITY": oof_probability,
            }
        ).sort_values("SK_ID_CURR")
        oof_predictions.to_parquet(oof_path, index=False)
        fold_results.to_csv(fold_path, index=False)

    print("拟合并比较概率校准方法", flush=True)
    selection_probabilities = generate_cross_validated_calibration_probabilities(
        y_development,
        oof_probability,
    )
    selection_metrics = calculate_calibration_metrics(y_development, selection_probabilities)
    selected_method = select_calibration_method(selection_metrics)
    calibrators = fit_calibrators(y_development, oof_probability)
    outer_target = outer_predictions["TARGET"].astype("int8")
    raw_outer_probability = outer_predictions["PREDICTED_PROBABILITY"].to_numpy()
    calibration_metrics, probabilities, decile_tables = evaluate_calibration_methods(
        outer_target,
        raw_outer_probability,
        calibrators,
    )
    selected_probability = probabilities[selected_method]
    joblib.dump(
        {"selected_method": selected_method, "calibrators": calibrators},
        MODEL_DIR / "calibrators.joblib",
        compress=3,
    )

    comparison = outer_predictions.rename(
        columns={"PREDICTED_PROBABILITY": "RAW_PROBABILITY"}
    ).copy()
    comparison["PLATT_PROBABILITY"] = probabilities["platt"]
    comparison["ISOTONIC_PROBABILITY"] = probabilities["isotonic"]
    comparison["SELECTED_PROBABILITY"] = selected_probability
    comparison.to_parquet(PROCESSED_DIR / "validation_calibration_comparison.parquet", index=False)
    selected_deciles = decile_tables[selected_method]
    selected_deciles.to_csv(TABLE_DIR / "selected_probability_deciles.csv", index=False)

    calibrated_test_probability = apply_calibrator(
        selected_method,
        calibrators,
        tuned_test["PREDICTED_PROBABILITY"].to_numpy(),
    )
    pd.DataFrame(
        {
            "SK_ID_CURR": tuned_test["SK_ID_CURR"],
            "RAW_PROBABILITY": tuned_test["PREDICTED_PROBABILITY"],
            "CALIBRATED_PROBABILITY": calibrated_test_probability,
            "CALIBRATION_METHOD": selected_method,
        }
    ).to_parquet(PROCESSED_DIR / "calibrated_test_predictions.parquet", index=False)

    ks_threshold = float(calibration_metrics[selected_method]["ks_threshold"])
    thresholds = [
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.10,
        0.12,
        0.15,
        0.20,
        0.25,
        0.30,
        0.50,
        ks_threshold,
    ]
    threshold_table = build_threshold_policy_table(
        outer_target,
        selected_probability,
        thresholds,
        args.performing_margin,
        args.loss_given_default,
    )
    capacity_table = build_rejection_capacity_table(
        outer_target,
        selected_probability,
        [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        args.performing_margin,
        args.loss_given_default,
    )
    band_table = build_three_band_table(outer_target, selected_probability)
    threshold_table.to_csv(TABLE_DIR / "threshold_policy.csv", index=False)
    capacity_table.to_csv(TABLE_DIR / "rejection_capacity_policy.csv", index=False)
    band_table.to_csv(TABLE_DIR / "three_band_policy.csv", index=False)

    save_calibration_comparison_figure(
        decile_tables,
        FIGURE_DIR / "calibration_comparison.png",
    )
    save_diagnostic_figure(
        outer_target,
        selected_probability,
        selected_deciles,
        FIGURE_DIR / "selected_probability_diagnostics.png",
    )
    save_policy_tradeoff_figure(
        capacity_table,
        FIGURE_DIR / "policy_tradeoffs.png",
    )
    write_report(
        selection_metrics,
        calibration_metrics,
        selected_method,
        fold_results,
        capacity_table,
        band_table,
        args.performing_margin,
        args.loss_given_default,
    )

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_model": "XGBClassifierTuned",
        "random_state": RANDOM_STATE,
        "oof_folds": OOF_FOLDS,
        "calibration_selection_folds": CALIBRATION_SELECTION_FOLDS,
        "development_rows": len(x_development),
        "outer_validation_rows": len(outer_predictions),
        "selected_method": selected_method,
        "selection_metrics": selection_metrics,
        "calibration_metrics": calibration_metrics,
        "economic_scenario": {
            "performing_margin": args.performing_margin,
            "loss_given_default": args.loss_given_default,
            "normalized_ead": 1.0,
        },
        "openmp_runtime": str(openmp_path) if openmp_path else None,
    }
    (REPORT_DIR / "calibration_policy_metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"完成：选择{selected_method}，Brier="
        f"{calibration_metrics[selected_method]['brier_score']:.5f}",
        flush=True,
    )
    print(f"策略报告：{REPORT_DIR / 'calibration_policy_report.md'}", flush=True)


if __name__ == "__main__":
    main()
