# DuckDB数据库构建报告

## 结论

- 数据库：`data/interim/smartcredit.duckdb`
- 文件大小：1.01 GB
- 申请记录：356,255条
- 训练记录：307,511条
- 还款困难样本：24,825条，占8.07%
- 特征矩阵：356,255行 × 198列
- `mart.feature_matrix`保持每个`SK_ID_CURR`一行，所有申请均保留。

## 表清单

| Schema | 表 | 行数 | 列数 |
|---|---|---:|---:|
| mart | applications | 356,255 | 123 |
| mart | bureau_features | 305,811 | 19 |
| mart | credit_card_features | 103,558 | 20 |
| mart | feature_matrix | 356,255 | 198 |
| mart | installment_features | 339,587 | 12 |
| mart | pos_cash_features | 337,252 | 13 |
| mart | previous_application_features | 338,857 | 16 |
| raw | application_test | 48,744 | 121 |
| raw | application_train | 307,511 | 122 |
| raw | bureau | 1,716,428 | 17 |
| raw | bureau_balance | 27,299,925 | 3 |
| raw | column_descriptions | 219 | 5 |
| raw | credit_card_balance | 3,840,312 | 23 |
| raw | installments_payments | 13,605,401 | 8 |
| raw | pos_cash_balance | 10,001,358 | 8 |
| raw | previous_application | 1,670,214 | 37 |
| raw | sample_submission | 48,744 | 2 |

## 关系检查

| 检查 | 违规行数 |
|---|---:|
| POS_CASH rows without application parent | 0 |
| application_test duplicate SK_ID_CURR | 0 |
| application_train duplicate SK_ID_CURR | 0 |
| bureau rows without application parent | 0 |
| credit_card rows without application parent | 0 |
| feature_matrix duplicate SK_ID_CURR | 0 |
| installments rows without application parent | 0 |
| previous_application rows without application parent | 0 |

## 常用入口

- `mart.feature_matrix_train`：建模训练数据
- `mart.feature_matrix_test`：Kaggle测试数据
- `mart.applications`：合并后的申请主表
- `meta.table_inventory`：表规模清单
- `meta.column_catalog`：字段和类型目录
- `meta.relationship_checks`：主外键及粒度检查
