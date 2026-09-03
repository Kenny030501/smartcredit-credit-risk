# Kaggle Late Submission Report

## Result

| Evaluation | ROC-AUC |
|---|---:|
| Local stratified holdout | 0.78545 |
| Kaggle Public | 0.77905 |
| Kaggle Private | 0.77994 |
| Local minus Private | 0.00551 |

The submission used `submission_xgboost_tuned.csv`, containing 48,744 unique
`SK_ID_CURR` values and continuous `TARGET` probabilities without missing
values.

## Interpretation

- Public and Private scores differ by only 0.00089, indicating similar
  performance across the two hidden test partitions.
- The local holdout exceeded the Private score by 0.00551, so the random
  stratified validation was mildly optimistic for the competition test set.
- A score of 0.77994 matches historical Private leaderboard positions roughly
  3,930-3,935 among 7,180 displayed rows. This is a historical comparison, not
  an official rank, because the submission was made after the 2018 deadline.
- The gap suggests that future competition improvements should prioritize
  richer time-window and product-level features, stronger validation, and
  leakage-safe ensembling rather than more tuning of the same feature matrix.

## Evidence boundary

`TARGET` is the competition's repayment-difficulty label, not a regulatory
default definition. The repository does not contain the competition data, and
the Late Submission result does not confer an official placement, medal, or
award.

Official competition: https://www.kaggle.com/competitions/home-credit-default-risk

