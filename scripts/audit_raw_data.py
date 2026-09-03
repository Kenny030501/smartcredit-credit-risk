"""Audit raw Home Credit CSV files for size, missingness, and key integrity."""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb

# Resolve paths from the project root so execution does not depend on the current directory.
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "home-credit-default-risk"
OUTPUT = ROOT / "reports" / "data_quality"

FILES = {
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


def csv_view(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    """Register a CSV as a DuckDB view without loading the entire file into memory."""
    escaped = str(path).replace("'", "''")
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_csv_auto('{escaped}', header=true, sample_size=100000)"
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write structured audit results as UTF-8 CSV for review or dashboards."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scalar(con: duckdb.DuckDBPyConnection, query: str) -> object:
    """Execute a SQL query that returns one scalar value."""
    return con.execute(query).fetchone()[0]


def main() -> None:
    """Run the full raw-data audit and generate Markdown and CSV reports."""
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Validate all inputs before scanning large files.
    missing = [filename for filename in FILES.values() if not (RAW / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}")

    con = duckdb.connect()

    # Create views only; do not copy the 2.5 GB raw dataset.
    for name, filename in FILES.items():
        csv_view(con, name, RAW / filename)

        # Inventory row counts, column counts, and disk size for every table.
    summaries = []
    for name, filename in FILES.items():
        columns = con.execute(f"DESCRIBE {name}").fetchall()
        summaries.append(
            {
                "table": name,
                "file": filename,
                "rows": scalar(con, f"SELECT COUNT(*) FROM {name}"),
                "columns": len(columns),
                "size_mb": round((RAW / filename).stat().st_size / 1024**2, 2),
            }
        )

        # Generate missing-value statistics for every training column dynamically.
    train_columns = [row[0] for row in con.execute("DESCRIBE application_train").fetchall()]
    missing_select = ", ".join(
        f'SUM(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END) AS "{column}"'
        for column in train_columns
    )
    missing_counts = con.execute(f"SELECT {missing_select} FROM application_train").fetchone()
    train_rows = next(row["rows"] for row in summaries if row["table"] == "application_train")
    missingness = sorted(
        [
            {
                "column": column,
                "missing_count": count,
                "missing_pct": round(count * 100 / train_rows, 4),
            }
            for column, count in zip(train_columns, missing_counts)
        ],
        key=lambda row: row["missing_pct"],
        reverse=True,
    )

    # Check unique keys, train-test overlap, and parent-table coverage.
    key_checks = [
        {
            "check": "application_train duplicate SK_ID_CURR",
            "violations": scalar(
                con,
                "SELECT COUNT(*) - COUNT(DISTINCT SK_ID_CURR) FROM application_train",
            ),
        },
        {
            "check": "application_test duplicate SK_ID_CURR",
            "violations": scalar(
                con,
                "SELECT COUNT(*) - COUNT(DISTINCT SK_ID_CURR) FROM application_test",
            ),
        },
        {
            "check": "train-test overlapping SK_ID_CURR",
            "violations": scalar(
                con,
                "SELECT COUNT(*) FROM application_train t INNER JOIN application_test s USING (SK_ID_CURR)",
            ),
        },
        {
            "check": "bureau orphan SK_ID_CURR",
            "violations": scalar(
                con,
                """
                SELECT COUNT(*)
                FROM bureau b
                WHERE NOT EXISTS (
                    SELECT 1 FROM application_train a WHERE a.SK_ID_CURR = b.SK_ID_CURR
                ) AND NOT EXISTS (
                    SELECT 1 FROM application_test a WHERE a.SK_ID_CURR = b.SK_ID_CURR
                )
                """,
            ),
        },
        {
            "check": "bureau_balance rows without bureau parent",
            "violations": scalar(
                con,
                """
                SELECT COUNT(*)
                FROM bureau_balance bb
                WHERE NOT EXISTS (
                    SELECT 1 FROM bureau b WHERE b.SK_ID_BUREAU = bb.SK_ID_BUREAU
                )
                """,
            ),
        },
        {
            "check": "previous_application orphan SK_ID_CURR",
            "violations": scalar(
                con,
                """
                SELECT COUNT(*)
                FROM previous_application p
                WHERE NOT EXISTS (
                    SELECT 1 FROM application_train a WHERE a.SK_ID_CURR = p.SK_ID_CURR
                ) AND NOT EXISTS (
                    SELECT 1 FROM application_test a WHERE a.SK_ID_CURR = p.SK_ID_CURR
                )
                """,
            ),
        },
        {
            "check": "POS_CASH rows without previous_application parent",
            "violations": scalar(
                con,
                """
                SELECT COUNT(*)
                FROM pos_cash_balance c
                WHERE NOT EXISTS (
                    SELECT 1 FROM previous_application p WHERE p.SK_ID_PREV = c.SK_ID_PREV
                )
                """,
            ),
        },
        {
            "check": "POS_CASH rows without application parent",
            "violations": scalar(
                con,
                """
                SELECT COUNT(*)
                FROM pos_cash_balance c
                WHERE NOT EXISTS (
                    SELECT 1 FROM application_train a WHERE a.SK_ID_CURR = c.SK_ID_CURR
                ) AND NOT EXISTS (
                    SELECT 1 FROM application_test a WHERE a.SK_ID_CURR = c.SK_ID_CURR
                )
                """,
            ),
        },
        {
            "check": "credit_card rows without previous_application parent",
            "violations": scalar(
                con,
                """
                SELECT COUNT(*)
                FROM credit_card_balance c
                WHERE NOT EXISTS (
                    SELECT 1 FROM previous_application p WHERE p.SK_ID_PREV = c.SK_ID_PREV
                )
                """,
            ),
        },
        {
            "check": "credit_card rows without application parent",
            "violations": scalar(
                con,
                """
                SELECT COUNT(*)
                FROM credit_card_balance c
                WHERE NOT EXISTS (
                    SELECT 1 FROM application_train a WHERE a.SK_ID_CURR = c.SK_ID_CURR
                ) AND NOT EXISTS (
                    SELECT 1 FROM application_test a WHERE a.SK_ID_CURR = c.SK_ID_CURR
                )
                """,
            ),
        },
        {
            "check": "installments rows without previous_application parent",
            "violations": scalar(
                con,
                """
                SELECT COUNT(*)
                FROM installments_payments i
                WHERE NOT EXISTS (
                    SELECT 1 FROM previous_application p WHERE p.SK_ID_PREV = i.SK_ID_PREV
                )
                """,
            ),
        },
        {
            "check": "installments rows without application parent",
            "violations": scalar(
                con,
                """
                SELECT COUNT(*)
                FROM installments_payments i
                WHERE NOT EXISTS (
                    SELECT 1 FROM application_train a WHERE a.SK_ID_CURR = i.SK_ID_CURR
                ) AND NOT EXISTS (
                    SELECT 1 FROM application_test a WHERE a.SK_ID_CURR = i.SK_ID_CURR
                )
                """,
            ),
        },
    ]

    # TARGET=1 denotes a repayment-difficulty case under the competition definition.
    target = con.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN TARGET = 1 THEN 1 ELSE 0 END) AS difficulties,
            AVG(TARGET) AS difficulty_rate
        FROM application_train
        """
    ).fetchone()

    # Produce both machine-readable CSV files and a human-readable Markdown report.
    write_csv(OUTPUT / "table_summary.csv", summaries)
    write_csv(OUTPUT / "application_train_missingness.csv", missingness)
    write_csv(OUTPUT / "key_integrity.csv", key_checks)

    top_missing = "\n".join(
        f"| {row['column']} | {row['missing_count']:,} | {row['missing_pct']:.2f}% |"
        for row in missingness[:15]
    )
    table_rows = "\n".join(
        f"| {row['table']} | {row['rows']:,} | {row['columns']} | {row['size_mb']:.2f} |"
        for row in summaries
    )
    key_rows = "\n".join(f"| {row['check']} | {row['violations']:,} |" for row in key_checks)
    report = f"""# 原始数据审计

## 结论

- 数据文件：{len(summaries)}张业务表，均可读取。
- 训练集：{target[0]:,}条申请。
- 还款困难样本：{target[1]:,}条，占{target[2] * 100:.2f}%。
- 类别明显不平衡，后续不能仅使用准确率评价模型。
- POS、信用卡和分期明细虽有部分`SK_ID_PREV`不在历史申请表中，但全部仍可通过`SK_ID_CURR`连接到主申请。
- `bureau_balance`中有部分账户不在`bureau`中，无法映射到申请人；申请人级建模时需要排除这些不可关联记录。

## 表规模

| 表 | 行数 | 列数 | 文件大小MB |
|---|---:|---:|---:|
{table_rows}

## 主外键检查

| 检查 | 违规行数 |
|---|---:|
{key_rows}

## 训练集缺失率最高的字段

| 字段 | 缺失数 | 缺失率 |
|---|---:|---:|
{top_missing}

## 下一步

1. 核对字段定义及时间语义。
2. 把各明细表聚合到`SK_ID_CURR`粒度，严禁直接产生多对多行数膨胀。
3. 建立只使用申请时可见信息的基线特征集。
4. 使用分层训练/验证切分，先完成Logistic Regression基线。
"""
    (OUTPUT / "raw_data_audit.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    # Run the audit only when this file is executed directly.
    main()
