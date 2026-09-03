"""SmartCredit data loading, schema validation, and deterministic cleaning."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def load_model_frames(
    database: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Load modeling views from DuckDB and identify numeric and categorical fields."""
    if not database.exists():
        raise FileNotFoundError(f"缺少数据库：{database}")

    with duckdb.connect(str(database), read_only=True) as con:
        train = con.execute("SELECT * FROM mart.feature_matrix_train ORDER BY SK_ID_CURR").fetchdf()
        test = con.execute("SELECT * FROM mart.feature_matrix_test ORDER BY SK_ID_CURR").fetchdf()
        schema = con.execute("DESCRIBE mart.feature_matrix_train").fetchdf()

    feature_columns = [column for column in train.columns if column not in {"SK_ID_CURR", "TARGET"}]
    if feature_columns != [column for column in test.columns if column != "SK_ID_CURR"]:
        raise ValueError("训练集和测试集的特征列不一致")
    if train["SK_ID_CURR"].duplicated().any() or test["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR不是唯一键")
    if train["TARGET"].isna().any() or not set(train["TARGET"].unique()).issubset({0, 1}):
        raise ValueError("训练标签必须完整且只能取0或1")

    type_by_column = dict(zip(schema["column_name"], schema["column_type"], strict=True))
    categorical_columns = [
        column for column in feature_columns if type_by_column[column] in {"VARCHAR", "BOOLEAN"}
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    return train, test, numeric_columns, categorical_columns


def clean_features(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Correct known sentinel values and normalize train, validation, and test dtypes."""
    features = frame[numeric_columns + categorical_columns].copy()

    # The source uses 365243 for missing employment history; treat it as missing, not days.
    if "DAYS_EMPLOYED" in features.columns:
        features.loc[features["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    # Cast numeric fields to float32 for memory efficiency; pipelines handle infinities later.
    for column in numeric_columns:
        features[column] = (
            pd.to_numeric(features[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )

    # A fixed category marker learns no full-dataset statistics and cannot leak holdout data.
    for column in categorical_columns:
        series = features[column].astype("string")
        features[column] = series.fillna("__MISSING__").astype("object")
    return features
