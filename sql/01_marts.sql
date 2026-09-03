-- Aggregate raw detail into applicant-level marts with one row per SK_ID_CURR.
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS meta;

-- Combine train and test while retaining source; TARGET is always null for test rows.
CREATE TABLE mart.applications AS
SELECT *, 'train'::VARCHAR AS dataset_split
FROM raw.application_train
UNION ALL BY NAME
SELECT *, NULL::BIGINT AS TARGET, 'test'::VARCHAR AS dataset_split
FROM raw.application_test;

CREATE UNIQUE INDEX applications_sk_id_curr_idx
ON mart.applications (SK_ID_CURR);

-- Aggregate monthly bureau status to account and then applicant to prevent join inflation.
CREATE TABLE mart.bureau_features AS
WITH balance_by_bureau AS (
    SELECT
        SK_ID_BUREAU,
        COUNT(*) AS bureau_balance_month_count,
        MIN(MONTHS_BALANCE) AS bureau_oldest_balance_month,
        MAX(MONTHS_BALANCE) AS bureau_latest_balance_month,
        COUNT_IF(TRY_CAST(STATUS AS INTEGER) >= 1) AS bureau_delinquent_month_count,
        COUNT_IF(TRY_CAST(STATUS AS INTEGER) >= 3) AS bureau_severe_delinquent_month_count,
        -- STATUS values 0-5 are delinquency levels; C and X are nonnumeric and map to zero.
        MAX(COALESCE(TRY_CAST(STATUS AS INTEGER), 0)) AS bureau_max_monthly_status
    FROM raw.bureau_balance
    GROUP BY SK_ID_BUREAU
),
enriched AS (
    SELECT b.*, bb.* EXCLUDE (SK_ID_BUREAU)
    FROM raw.bureau b
    LEFT JOIN balance_by_bureau bb USING (SK_ID_BUREAU)
)
SELECT
    SK_ID_CURR,
    COUNT(*) AS bureau_credit_count,
    COUNT_IF(CREDIT_ACTIVE = 'Active') AS bureau_active_credit_count,
    COUNT_IF(CREDIT_ACTIVE = 'Closed') AS bureau_closed_credit_count,
    COUNT(DISTINCT CREDIT_TYPE) AS bureau_credit_type_count,
    MIN(DAYS_CREDIT) AS bureau_oldest_credit_days,
    MAX(DAYS_CREDIT) AS bureau_most_recent_credit_days,
    AVG(DAYS_CREDIT) AS bureau_avg_credit_days,
    MAX(CREDIT_DAY_OVERDUE) AS bureau_max_credit_day_overdue,
    SUM(COALESCE(CNT_CREDIT_PROLONG, 0)) AS bureau_total_credit_prolong,
    SUM(COALESCE(AMT_CREDIT_SUM, 0)) AS bureau_total_credit_amount,
    SUM(COALESCE(AMT_CREDIT_SUM_DEBT, 0)) AS bureau_total_debt_amount,
    SUM(COALESCE(AMT_CREDIT_SUM_OVERDUE, 0)) AS bureau_total_overdue_amount,
    MAX(COALESCE(AMT_CREDIT_SUM_OVERDUE, 0)) AS bureau_max_overdue_amount,
    SUM(COALESCE(AMT_ANNUITY, 0)) AS bureau_total_annuity,
    SUM(COALESCE(bureau_balance_month_count, 0)) AS bureau_balance_month_count,
    SUM(COALESCE(bureau_delinquent_month_count, 0)) AS bureau_delinquent_month_count,
    SUM(COALESCE(bureau_severe_delinquent_month_count, 0)) AS bureau_severe_delinquent_month_count,
    MAX(COALESCE(bureau_max_monthly_status, 0)) AS bureau_max_monthly_status
FROM enriched
GROUP BY SK_ID_CURR;

CREATE UNIQUE INDEX bureau_features_sk_id_curr_idx
ON mart.bureau_features (SK_ID_CURR);

-- Summarize decisions, amounts, products, and installment terms from prior applications.
CREATE TABLE mart.previous_application_features AS
SELECT
    SK_ID_CURR,
    COUNT(*) AS previous_application_count,
    COUNT_IF(NAME_CONTRACT_STATUS = 'Approved') AS previous_approved_count,
    COUNT_IF(NAME_CONTRACT_STATUS = 'Refused') AS previous_refused_count,
    COUNT_IF(NAME_CONTRACT_STATUS = 'Canceled') AS previous_canceled_count,
    COUNT(DISTINCT NAME_CONTRACT_TYPE) AS previous_contract_type_count,
    COUNT(DISTINCT NAME_PORTFOLIO) AS previous_portfolio_count,
    AVG(DAYS_DECISION) AS previous_avg_decision_days,
    MAX(DAYS_DECISION) AS previous_most_recent_decision_days,
    AVG(AMT_APPLICATION) AS previous_avg_application_amount,
    SUM(COALESCE(AMT_APPLICATION, 0)) AS previous_total_application_amount,
    AVG(AMT_CREDIT) AS previous_avg_credit_amount,
    SUM(COALESCE(AMT_CREDIT, 0)) AS previous_total_credit_amount,
    AVG(AMT_ANNUITY) AS previous_avg_annuity,
    AVG(CNT_PAYMENT) AS previous_avg_payment_count,
    AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0)) AS previous_avg_credit_to_application
FROM raw.previous_application
GROUP BY SK_ID_CURR;

CREATE UNIQUE INDEX previous_features_sk_id_curr_idx
ON mart.previous_application_features (SK_ID_CURR);

-- Summarize monthly contract status and delinquency days for POS and cash loans.
CREATE TABLE mart.pos_cash_features AS
SELECT
    SK_ID_CURR,
    COUNT(*) AS pos_month_count,
    COUNT(DISTINCT SK_ID_PREV) AS pos_contract_count,
    MIN(MONTHS_BALANCE) AS pos_oldest_month,
    MAX(MONTHS_BALANCE) AS pos_latest_month,
    AVG(CNT_INSTALMENT) AS pos_avg_instalment_count,
    AVG(CNT_INSTALMENT_FUTURE) AS pos_avg_future_instalment_count,
    AVG(SK_DPD) AS pos_avg_dpd,
    MAX(SK_DPD) AS pos_max_dpd,
    AVG(SK_DPD_DEF) AS pos_avg_dpd_def,
    MAX(SK_DPD_DEF) AS pos_max_dpd_def,
    AVG(CASE WHEN SK_DPD > 0 THEN 1.0 ELSE 0.0 END) AS pos_delinquent_month_share,
    COUNT_IF(NAME_CONTRACT_STATUS = 'Completed') AS pos_completed_month_count
FROM raw.pos_cash_balance
GROUP BY SK_ID_CURR;

CREATE UNIQUE INDEX pos_features_sk_id_curr_idx
ON mart.pos_cash_features (SK_ID_CURR);

-- Summarize credit-card balances, utilization, payments, and delinquency.
CREATE TABLE mart.credit_card_features AS
SELECT
    SK_ID_CURR,
    COUNT(*) AS card_month_count,
    COUNT(DISTINCT SK_ID_PREV) AS card_contract_count,
    MIN(MONTHS_BALANCE) AS card_oldest_month,
    MAX(MONTHS_BALANCE) AS card_latest_month,
    AVG(AMT_BALANCE) AS card_avg_balance,
    MAX(AMT_BALANCE) AS card_max_balance,
    AVG(AMT_CREDIT_LIMIT_ACTUAL) AS card_avg_credit_limit,
    MAX(AMT_CREDIT_LIMIT_ACTUAL) AS card_max_credit_limit,
    AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) AS card_avg_utilization,
    MAX(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) AS card_max_utilization,
    SUM(COALESCE(AMT_DRAWINGS_CURRENT, 0)) AS card_total_drawings,
    SUM(COALESCE(AMT_PAYMENT_TOTAL_CURRENT, 0)) AS card_total_payments,
    AVG(AMT_TOTAL_RECEIVABLE) AS card_avg_total_receivable,
    MAX(AMT_TOTAL_RECEIVABLE) AS card_max_total_receivable,
    AVG(SK_DPD) AS card_avg_dpd,
    MAX(SK_DPD) AS card_max_dpd,
    AVG(SK_DPD_DEF) AS card_avg_dpd_def,
    MAX(SK_DPD_DEF) AS card_max_dpd_def,
    AVG(CASE WHEN SK_DPD > 0 THEN 1.0 ELSE 0.0 END) AS card_delinquent_month_share
FROM raw.credit_card_balance
GROUP BY SK_ID_CURR;

CREATE UNIQUE INDEX card_features_sk_id_curr_idx
ON mart.credit_card_features (SK_ID_CURR);

-- Compare scheduled and actual installments to derive shortfall, ratio, and delay features.
CREATE TABLE mart.installment_features AS
SELECT
    SK_ID_CURR,
    COUNT(*) AS installment_record_count,
    COUNT(DISTINCT SK_ID_PREV) AS installment_contract_count,
    COUNT(DISTINCT NUM_INSTALMENT_NUMBER) AS installment_number_count,
    SUM(COALESCE(AMT_INSTALMENT, 0)) AS installment_total_scheduled_amount,
    SUM(COALESCE(AMT_PAYMENT, 0)) AS installment_total_paid_amount,
    SUM(GREATEST(COALESCE(AMT_INSTALMENT, 0) - COALESCE(AMT_PAYMENT, 0), 0))
        AS installment_total_shortfall,
    SUM(COALESCE(AMT_PAYMENT, 0)) / NULLIF(SUM(COALESCE(AMT_INSTALMENT, 0)), 0)
        AS installment_payment_ratio,
    AVG(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS installment_avg_days_late,
    MAX(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS installment_max_days_late,
    AVG(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1.0 ELSE 0.0 END)
        AS installment_late_payment_share,
    AVG(
        CASE
            WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1.0
            ELSE 0.0
        END
    ) AS installment_short_payment_share
FROM raw.installments_payments
GROUP BY SK_ID_CURR;

CREATE UNIQUE INDEX installment_features_sk_id_curr_idx
ON mart.installment_features (SK_ID_CURR);

-- Preserve every current application and left-join five applicant-level feature groups.
CREATE TABLE mart.feature_matrix AS
SELECT
    a.*,
    b.* EXCLUDE (SK_ID_CURR),
    p.* EXCLUDE (SK_ID_CURR),
    pos.* EXCLUDE (SK_ID_CURR),
    cc.* EXCLUDE (SK_ID_CURR),
    i.* EXCLUDE (SK_ID_CURR)
FROM mart.applications a
LEFT JOIN mart.bureau_features b USING (SK_ID_CURR)
LEFT JOIN mart.previous_application_features p USING (SK_ID_CURR)
LEFT JOIN mart.pos_cash_features pos USING (SK_ID_CURR)
LEFT JOIN mart.credit_card_features cc USING (SK_ID_CURR)
LEFT JOIN mart.installment_features i USING (SK_ID_CURR);

CREATE UNIQUE INDEX feature_matrix_sk_id_curr_idx
ON mart.feature_matrix (SK_ID_CURR);

-- Retain TARGET in the training view and remove it from the Kaggle test view.
CREATE VIEW mart.feature_matrix_train AS
SELECT * EXCLUDE (dataset_split)
FROM mart.feature_matrix
WHERE dataset_split = 'train';

CREATE VIEW mart.feature_matrix_test AS
SELECT * EXCLUDE (TARGET, dataset_split)
FROM mart.feature_matrix
WHERE dataset_split = 'test';

-- Persist uniqueness and parent-coverage checks for automatic build validation.
CREATE TABLE meta.relationship_checks AS
SELECT 'application_train duplicate SK_ID_CURR' AS check_name,
       COUNT(*) - COUNT(DISTINCT SK_ID_CURR) AS violation_rows
FROM raw.application_train
UNION ALL
SELECT 'application_test duplicate SK_ID_CURR',
       COUNT(*) - COUNT(DISTINCT SK_ID_CURR)
FROM raw.application_test
UNION ALL
SELECT 'feature_matrix duplicate SK_ID_CURR',
       COUNT(*) - COUNT(DISTINCT SK_ID_CURR)
FROM mart.feature_matrix
UNION ALL
SELECT 'bureau rows without application parent', COUNT(*)
FROM raw.bureau b
WHERE NOT EXISTS (
    SELECT 1 FROM mart.applications a WHERE a.SK_ID_CURR = b.SK_ID_CURR
)
UNION ALL
SELECT 'previous_application rows without application parent', COUNT(*)
FROM raw.previous_application p
WHERE NOT EXISTS (
    SELECT 1 FROM mart.applications a WHERE a.SK_ID_CURR = p.SK_ID_CURR
)
UNION ALL
SELECT 'POS_CASH rows without application parent', COUNT(*)
FROM raw.pos_cash_balance p
WHERE NOT EXISTS (
    SELECT 1 FROM mart.applications a WHERE a.SK_ID_CURR = p.SK_ID_CURR
)
UNION ALL
SELECT 'credit_card rows without application parent', COUNT(*)
FROM raw.credit_card_balance c
WHERE NOT EXISTS (
    SELECT 1 FROM mart.applications a WHERE a.SK_ID_CURR = c.SK_ID_CURR
)
UNION ALL
SELECT 'installments rows without application parent', COUNT(*)
FROM raw.installments_payments i
WHERE NOT EXISTS (
    SELECT 1 FROM mart.applications a WHERE a.SK_ID_CURR = i.SK_ID_CURR
);

-- Expose raw and mart field names, types, and nullability as a column catalog.
CREATE VIEW meta.column_catalog AS
SELECT
    schema_name,
    table_name,
    column_index,
    column_name,
    data_type,
    is_nullable
FROM duckdb_columns()
WHERE schema_name IN ('raw', 'mart')
ORDER BY schema_name, table_name, column_index;
