# 原始数据审计

## 结论

- 数据文件：9张业务表，均可读取。
- 训练集：307,511条申请。
- 还款困难样本：24,825条，占8.07%。
- 类别明显不平衡，后续不能仅使用准确率评价模型。
- POS、信用卡和分期明细虽有部分`SK_ID_PREV`不在历史申请表中，但全部仍可通过`SK_ID_CURR`连接到主申请。
- `bureau_balance`中有部分账户不在`bureau`中，无法映射到申请人；申请人级建模时需要排除这些不可关联记录。

## 表规模

| 表 | 行数 | 列数 | 文件大小MB |
|---|---:|---:|---:|
| application_train | 307,511 | 122 | 158.44 |
| application_test | 48,744 | 121 | 25.34 |
| bureau | 1,716,428 | 17 | 162.14 |
| bureau_balance | 27,299,925 | 3 | 358.19 |
| previous_application | 1,670,214 | 37 | 386.21 |
| pos_cash_balance | 10,001,358 | 8 | 374.51 |
| credit_card_balance | 3,840,312 | 23 | 404.91 |
| installments_payments | 13,605,401 | 8 | 689.62 |
| sample_submission | 48,744 | 2 | 0.51 |

## 主外键检查

| 检查 | 违规行数 |
|---|---:|
| application_train duplicate SK_ID_CURR | 0 |
| application_test duplicate SK_ID_CURR | 0 |
| train-test overlapping SK_ID_CURR | 0 |
| bureau orphan SK_ID_CURR | 0 |
| bureau_balance rows without bureau parent | 3,120,184 |
| previous_application orphan SK_ID_CURR | 0 |
| POS_CASH rows without previous_application parent | 340,561 |
| POS_CASH rows without application parent | 0 |
| credit_card rows without previous_application parent | 1,082,816 |
| credit_card rows without application parent | 0 |
| installments rows without previous_application parent | 1,250,826 |
| installments rows without application parent | 0 |

## 训练集缺失率最高的字段

| 字段 | 缺失数 | 缺失率 |
|---|---:|---:|
| COMMONAREA_AVG | 214,865 | 69.87% |
| COMMONAREA_MODE | 214,865 | 69.87% |
| COMMONAREA_MEDI | 214,865 | 69.87% |
| NONLIVINGAPARTMENTS_AVG | 213,514 | 69.43% |
| NONLIVINGAPARTMENTS_MODE | 213,514 | 69.43% |
| NONLIVINGAPARTMENTS_MEDI | 213,514 | 69.43% |
| FONDKAPREMONT_MODE | 210,295 | 68.39% |
| LIVINGAPARTMENTS_AVG | 210,199 | 68.36% |
| LIVINGAPARTMENTS_MODE | 210,199 | 68.36% |
| LIVINGAPARTMENTS_MEDI | 210,199 | 68.36% |
| FLOORSMIN_AVG | 208,642 | 67.85% |
| FLOORSMIN_MODE | 208,642 | 67.85% |
| FLOORSMIN_MEDI | 208,642 | 67.85% |
| YEARS_BUILD_AVG | 204,488 | 66.50% |
| YEARS_BUILD_MODE | 204,488 | 66.50% |

## 下一步

1. 核对字段定义及时间语义。
2. 把各明细表聚合到`SK_ID_CURR`粒度，严禁直接产生多对多行数膨胀。
3. 建立只使用申请时可见信息的基线特征集。
4. 使用分层训练/验证切分，先完成Logistic Regression基线。
