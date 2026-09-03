-- List all materialized tables in the database.
SHOW ALL TABLES;

-- Preview 20 rows of applicant-level POS and cash-loan features.
SELECT *
FROM mart.pos_cash_features
LIMIT 20;

-- Inspect a compact set of interpretable POS and cash-loan fields.
SELECT
    SK_ID_CURR,
    pos_contract_count,
    pos_month_count,
    pos_oldest_month,
    pos_latest_month,
    pos_max_dpd,
    pos_delinquent_month_share
FROM mart.pos_cash_features
ORDER BY pos_max_dpd DESC
LIMIT 50;

-- Preview applications, labels, and selected historical risk features in training data.
SELECT
    SK_ID_CURR,
    TARGET,
    AMT_INCOME_TOTAL,
    AMT_CREDIT,
    bureau_credit_count,
    bureau_total_debt_amount,
    card_avg_utilization,
    installment_late_payment_share
FROM mart.feature_matrix_train
LIMIT 50;

-- Inspect all modeling fields for one applicant; replace the SK_ID_CURR value as needed.
SELECT *
FROM mart.feature_matrix_train
WHERE SK_ID_CURR = 100002;

-- Cross-check the applicant's records in the raw monthly POS table.
SELECT *
FROM raw.pos_cash_balance
WHERE SK_ID_CURR = 100002
ORDER BY MONTHS_BALANCE;
