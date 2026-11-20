# 70 Percent Precision Across About 100 Matches

The shared linear goal-scoring model, selected using earlier validation seasons, got **71 of 100 predictions correct** in the retrospective evaluation. Its previously frozen 65% confidence cutoff produced **72 correct predictions out of 102**, or **70.6%**. Both results meet the requested target on the historical sample of 619 eligible matches.

This study interprets “about 100” as 90–110 predictions from that sample. The evaluation seasons have already been examined during the project, so these observations require independent future validation before they can establish reliable performance.

## Results at the requested prediction count

Precision here is the share of issued predictions that were correct. Coverage is the share of the 619 eligible fixtures receiving a prediction.

| Selection rule for the chosen model | Correct / predicted | Precision | Coverage |
| --- | ---: | ---: | ---: |
| Select its 100 most confident forecasts from the complete evaluation pool | 71 / 100 | 71.0% | 16.2% |
| Apply the previous 65% confidence cutoff to each match | 72 / 102 | 70.6% | 16.5% |
| Apply the new validation confidence cutoff of approximately 67.51% | 51 / 69 | 73.9% | 11.1% |
| Update the cutoff from the preceding 350 confidence scores | 53 / 73 | 72.6% | 11.8% |

Selecting exactly 100 requires ranking the complete pool. This is a retrospective comparison across matches whose features became available on different dates. The fixed and rolling cutoffs can make decisions as matches arrive. Their eventual prediction counts vary with the available fixtures and their confidence scores. [Full metrics](metrics.csv)

The 71/100 result has an illustrative 95% Wilson interval of 61.5%–79.0%. Its season results were 53/74 in 2024–2025 and 18/26 in the available 2025–2026 matches. The 72/102 result has an interval of 61.1%–78.6%, with season results of 54/75 and 18/27. The intervals assume independent calls and do not adjust for the sequence of project experiments.

## How the model was chosen

The four existing shared scoring models were reused. Two equal-weight ensembles were also evaluated, one combining the two actual-goal models and one combining all four models. Their weights and feature definitions follow the previous shared-goal study. This round aligns model selection with precision at the requested prediction count.

Each candidate generated forecasts for 2019–2020 through 2023–2024 using training matches from earlier seasons. Approximately 16.2% of each validation season was selected by confidence, giving 283 validation predictions per candidate. Eligibility required at least 70% precision overall and in each of the two most recent validation seasons. The shared linear model trained on actual goals led with 216/283, or 76.3%, and was selected before evaluating the later sample in this run. [Validation comparison](validation_summary.csv) · [Validation by season](validation_by_season.csv)

The chosen model uses both teams' earlier form, Elo strength, shots, expected-goal histories, opponent statistics, and home advantage. One scoring function learns from both home and away team records, and its predicted goal rates become home-win, draw, and away-win probabilities. Its target is actual goals. Expected-goal history remains an input. [Frozen model and selection settings](frozen_policy.json)

The 65% confidence cutoff was chosen in the earlier shared-goal experiment under its conservative validation requirements. It was already recorded before the earlier evaluation and is included as an existing operating rule. The new fixed cutoff comes from the validation confidence quantile for the requested coverage. The rolling rule uses the same quantile of preceding confidence scores and updates only after every match on a calendar date has been evaluated. [Earlier cutoff selection](../shared_goals/frozen_policy.json)

## Comparison at exactly 100 predictions

| Forecast | Correct / predicted |
| --- | ---: |
| Selected shared linear model trained on goals | 71 / 100 |
| Shared linear model trained on goals and xG | 70 / 100 |
| Shared CatBoost model trained on goals | 69 / 100 |
| Shared CatBoost model trained on goals and xG | 68 / 100 |
| Ensemble of the two actual-goal models | 71 / 100 |
| Ensemble of all four shared models | 71 / 100 |
| Previous CatBoost outcome classifier | 65 / 100 |
| Archived market benchmark | 72 / 100 |

The selected model meets the requested historical precision and prediction count. The market benchmark is one correct prediction ahead at the same count. The earlier 80.6% CatBoost result covered 62 matches, and its result at 100 predictions is 69%. [Metrics](metrics.csv)

## Verification and reproduction

Four new targeted tests passed. They check exact selection counts, deterministic tie handling, exclusion of future confidence scores from earlier decisions, shared cutoffs within each date, and rejection of unordered dates. The saved forecast bundle reproduced the selected model's probability exports after reloading. The experiment uses the retained expected-goals data and the same eligible fixtures as the previous study. [Data coverage](../xg/coverage.csv)

Run the following from the repository root in the environment used for the preceding studies.

```sh
python3 scripts/coverage_target_study.py
python3 -m unittest discover -s tests -p test_coverage_target_study.py
```

The chosen model bundle is saved locally as `models/coverage_target/selected_forecaster.joblib`. Its `previous_conservative_policy` field stores the earlier 65% confidence cutoff, while `fixed_cutoff` and the confidence history support the newly tested rules. The exact retrospective list is saved locally as `reports/coverage_target/selected_100_matches.csv`, with predictions, confidence scores, outcomes, and correctness. Full prediction exports are in `reports/coverage_target/predictions.csv`.

These experiments extend the numerical portion of Milestone 1. The original preview-feature comparison remains the evidence for the LLM's contribution. [Milestone 1 results](../full/results.md)
