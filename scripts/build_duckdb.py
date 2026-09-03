"""Build a layered DuckDB database from Home Credit CSV files and validate it."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

# Project paths and build artifact locations.
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "home-credit-default-risk"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports" / "data_quality"
FINAL_DB = INTERIM / "smartcredit.duckdb"
MART_SQL = ROOT / "sql" / "01_marts.sql"

RAW_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
    "sample_submission": "sample_submission.csv",
}

# Expected competition row counts used to reject truncated or incomplete inputs.
EXPECTED_ROWS = {
    "application_train": 307_511,
    "application_test": 48_744,
    "bureau": 1_716_428,
    "bureau_balance": 27_299_925,
    "previous_application": 1_670_214,
    "pos_cash_balance": 10_001_358,
    "credit_card_balance": 3_840_312,
    "installments_payments": 13_605_401,
    "sample_submission": 48_744,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments; replacing an existing database requires --overwrite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sql_path(path: Path) -> str:
    """Escape single quotes so a file path can be embedded safely in DuckDB SQL."""
    return str(path).replace("'", "''")


def load_raw_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Materialize the nine competition tables and column descriptions in raw."""
    con.execute("CREATE SCHEMA raw")
    for table, filename in RAW_FILES.items():
        path = RAW / filename
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Loading raw.{table} from {filename}", flush=True)
        # Let read_csv_auto infer types and materialize tables to avoid repeated CSV scans.
        con.execute(
            f"""
            CREATE TABLE raw.{table} AS
            SELECT *
            FROM read_csv_auto(
                '{sql_path(path)}',
                header = true,
                sample_size = 100000,
                parallel = true
            )
            """
        )
        actual = con.execute(f"SELECT COUNT(*) FROM raw.{table}").fetchone()[0]
        if actual != EXPECTED_ROWS[table]:
            raise ValueError(f"raw.{table}: expected {EXPECTED_ROWS[table]}, got {actual}")

    # Convert the Windows-1252 column-description file through pandas before loading it.
    description_path = RAW / "HomeCredit_columns_description.csv"
    descriptions = pd.read_csv(description_path, encoding="cp1252")
    descriptions = descriptions.rename(
        columns={
            "Unnamed: 0": "source_row",
            "Table": "source_table_pattern",
            "Row": "column_name",
            "Description": "description",
            "Special": "special",
        }
    )
    con.register("column_descriptions_df", descriptions)
    con.execute("CREATE TABLE raw.column_descriptions AS SELECT * FROM column_descriptions_df")
    con.unregister("column_descriptions_df")


def populate_inventory(con: duckdb.DuckDBPyConnection) -> None:
    """Record dimensions and build time for each materialized raw and mart table."""
    con.execute(
        """
        CREATE TABLE meta.table_inventory (
            schema_name VARCHAR,
            table_name VARCHAR,
            row_count BIGINT,
            column_count BIGINT,
            built_at TIMESTAMPTZ
        )
        """
    )
    tables = con.execute(
        """
        SELECT schema_name, table_name
        FROM duckdb_tables()
        WHERE schema_name IN ('raw', 'mart')
        ORDER BY schema_name, table_name
        """
    ).fetchall()
    built_at = datetime.now(UTC)
    for schema, table in tables:
        row_count = con.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"').fetchone()[0]
        column_count = con.execute(
            """
            SELECT COUNT(*)
            FROM duckdb_columns()
            WHERE schema_name = ? AND table_name = ?
            """,
            [schema, table],
        ).fetchone()[0]
        con.execute(
            "INSERT INTO meta.table_inventory VALUES (?, ?, ?, ?, ?)",
            [schema, table, row_count, column_count, built_at],
        )


def verify_database(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    """Validate applicant grain, training labels, and primary-foreign key checks."""
    application_count, unique_application_count = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT SK_ID_CURR) FROM mart.applications"
    ).fetchone()
    matrix_count, unique_matrix_count = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT SK_ID_CURR) FROM mart.feature_matrix"
    ).fetchone()
    train_count, positive_count, positive_rate = con.execute(
        """
        SELECT COUNT(*), COUNT_IF(TARGET = 1), AVG(TARGET)
        FROM mart.feature_matrix_train
        """
    ).fetchone()
    matrix_columns = con.execute(
        """
        SELECT COUNT(*)
        FROM duckdb_columns()
        WHERE schema_name = 'mart' AND table_name = 'feature_matrix'
        """
    ).fetchone()[0]
    relationship_violations = con.execute(
        "SELECT COALESCE(SUM(violation_rows), 0) FROM meta.relationship_checks"
    ).fetchone()[0]

    # Promote the temporary database only after every assertion passes.
    if application_count != 356_255 or unique_application_count != application_count:
        raise ValueError("mart.applications failed grain verification")
    if matrix_count != application_count or unique_matrix_count != matrix_count:
        raise ValueError("mart.feature_matrix failed grain verification")
    if train_count != EXPECTED_ROWS["application_train"] or positive_count != 24_825:
        raise ValueError("training target verification failed")
    if relationship_violations != 0:
        raise ValueError("relationship verification failed")

    return {
        "application_count": application_count,
        "matrix_count": matrix_count,
        "matrix_columns": matrix_columns,
        "train_count": train_count,
        "positive_count": positive_count,
        "positive_rate": positive_rate,
    }


def write_report(database_path: Path, metrics: dict[str, object]) -> None:
    """Read metadata from the completed database and generate a Chinese build report."""
    with duckdb.connect(str(database_path), read_only=True) as con:
        inventory = con.execute(
            """
            SELECT schema_name, table_name, row_count, column_count
            FROM meta.table_inventory
            ORDER BY schema_name, table_name
            """
        ).fetchall()
        checks = con.execute(
            "SELECT check_name, violation_rows FROM meta.relationship_checks ORDER BY check_name"
        ).fetchall()

    inventory_rows = "\n".join(
        f"| {schema} | {table} | {rows:,} | {columns} |"
        for schema, table, rows, columns in inventory
    )
    check_rows = "\n".join(f"| {name} | {violations:,} |" for name, violations in checks)
    size_gb = database_path.stat().st_size / 1024**3
    report = f"""# DuckDB数据库构建报告

## 结论

- 数据库：`data/interim/smartcredit.duckdb`
- 文件大小：{size_gb:.2f} GB
- 申请记录：{metrics["application_count"]:,}条
- 训练记录：{metrics["train_count"]:,}条
- 还款困难样本：{metrics["positive_count"]:,}条，占{metrics["positive_rate"] * 100:.2f}%
- 特征矩阵：{metrics["matrix_count"]:,}行 × {metrics["matrix_columns"]}列
- `mart.feature_matrix`保持每个`SK_ID_CURR`一行，所有申请均保留。

## 表清单

| Schema | 表 | 行数 | 列数 |
|---|---|---:|---:|
{inventory_rows}

## 关系检查

| 检查 | 违规行数 |
|---|---:|
{check_rows}

## 常用入口

- `mart.feature_matrix_train`：建模训练数据
- `mart.feature_matrix_test`：Kaggle测试数据
- `mart.applications`：合并后的申请主表
- `meta.table_inventory`：表规模清单
- `meta.column_catalog`：字段和类型目录
- `meta.relationship_checks`：主外键及粒度检查
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "duckdb_build_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Build and validate a temporary database before atomically replacing the target."""
    args = parse_args()
    INTERIM.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if FINAL_DB.exists() and not args.overwrite:
        raise FileExistsError(f"{FINAL_DB} already exists; use --overwrite")

    # Isolate temporary files by process ID so failures cannot damage the existing database.
    build_path = INTERIM / f"smartcredit.build.{os.getpid()}.duckdb"
    con = duckdb.connect(str(build_path))
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '8GB'")
    con.execute(f"SET temp_directory = '{sql_path(INTERIM / 'duckdb_temp')}'")

    try:
        # The raw layer preserves source data, mart aggregates it, and meta supports audits.
        load_raw_tables(con)
        print("Building mart tables", flush=True)
        con.execute(MART_SQL.read_text(encoding="utf-8"))
        populate_inventory(con)
        metrics = verify_database(con)
        con.execute("CHECKPOINT")
    finally:
        con.close()

        # When overwrite is explicit, preserve a timestamped backup of the old database.
    if FINAL_DB.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        FINAL_DB.replace(INTERIM / f"smartcredit.backup.{timestamp}.duckdb")
    build_path.replace(FINAL_DB)
    write_report(FINAL_DB, metrics)
    print(f"Created {FINAL_DB}", flush=True)


if __name__ == "__main__":
    # Start the database build only when this file is executed directly.
    main()
