# Continuing the precision experiments

The new experiments increased the number of predictions, but precision fell below 65% on the common 619-match evaluation sample. The previous goal-count policy remains the strongest observed result among the limited-coverage policies, with 44/58 correct predictions. Evidence that the project can reliably maintain 65% across larger or future samples remains insufficient.

## Same-match comparison

All rows below use the same 619 eligible fixtures from 2024–2025 and part of 2025–2026. Precision here means accuracy among the predictions issued. Coverage measures the fraction of eligible matches receiving a prediction.

| Model and selection policy | Correct / predicted | Precision | Coverage |
| --- | ---: | ---: | ---: |
| Previous goal-count model, conservative cutoff | 44 / 58 | 75.9% | 9.4% |
| Previous CatBoost with expected goals, conservative cutoff | 54 / 81 | 66.7% | 13.1% |
| Monthly team-strength model, conservative cutoff | 60 / 93 | 64.5% | 15.0% |
| Goal-count model with expected goals, broader cutoff | 100 / 158 | 63.3% | 25.5% |
| Equal blend of goal-count and CatBoost probabilities | 106 / 177 | 59.9% | 28.6% |
| Blend with probability calibration | 110 / 183 | 60.1% | 29.6% |
| Blend with a learned confidence model | 106 / 179 | 59.2% | 28.9% |
| CatBoost with expected goals, broader cutoff | 122 / 206 | 59.2% | 33.3% |

The broader policy selected CatBoost using its validation coverage before evaluating these later matches. Its subsequent precision missed the target. The other rows show the complete comparison. [Confidence experiment metrics](metrics.csv) · [Matched monthly-model comparison](paired_comparison.csv) · [Earlier conservative-policy results](../goal_score/metrics.csv)

## What was trained

The first experiment produced forecasts for seven earlier seasons using models fitted only on preceding seasons. It then trained two adjustments on those earlier forecasting errors. One adjustment recalibrated the blend's home/draw/away probabilities. The other estimated whether the blend's chosen winner would be correct, using confidence, the gap between the leading outcomes, draw probability, agreement between models, and whether the call favored the home team. Their evaluation for each later validation season used adjustments fitted on preceding seasons only. [Probability calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)

Base-model settings were retained from the earlier project experiments. The two logistic adjustments use regularization C=0.1. Selection searched confidence cutoffs using 2020–2021 through 2023–2024. A feasible cutoff required a pooled 95% Wilson lower bound of at least 65%, at least 100 calls and 15% coverage, plus 65% precision and 10% coverage in each of the final two validation seasons. The bounds serve as a selection rule. Searching models and cutoffs, correlated fixtures, and later changes in performance limit their interpretation as reliability guarantees. [Frozen policy](frozen_policy.json) · [Validation results](forward_validation.csv) · [Cutoff search](validation_cutoffs.csv)

The second experiment estimated an attacking strength and a defensive strength for every club, together with home advantage. It fitted these effects to earlier goal counts and updated them at the start of each calendar month. Validation selected a 365-day weighting half-life, regularization 0.001, and a zero low-score correlation adjustment. The model's cutoff was 64.5%, chosen under the previous conservative policy's development requirements. Predictions use team identities and venue, and can be generated for fixtures whose final scores are unknown. [Monthly model settings](../team_strength/frozen_policy.json)

## Full seasons and the newer snapshot

The monthly team-strength model can evaluate all 700 fixtures in the original numerical cohort, including the 81 excluded from the expected-goals comparison.

| Evaluation period | Correct / selected | Precision | Coverage |
| --- | ---: | ---: | ---: |
| 2024–2025 | 39 / 61 | 63.9% | 17.4% |
| 2025–2026 | 29 / 42 | 69.0% | 12.0% |
| Both seasons | 68 / 103 | 66.0% | 14.7% |
| Retained 2026–2027 snapshot | 4 / 7 | 57.1% | 10.0% |

The 68/103 result has an illustrative 95% Wilson interval of 56.4%–74.4%. The 4/7 snapshot interval is 25.0%–84.2%. Monthly forecasts use only matches completed before that month's first day. The snapshot contains 70 eligible matches and uses the retained November 19, 2026 source. Its outcomes had already been examined in earlier project experiments. These remain retrospective results. Across all fixtures, the monthly model scored 49.3% over the two full seasons and 50.0% in the snapshot. [Season and class metrics](../team_strength/metrics.csv)

The same-match comparison is important here. The monthly model's 66.0% uses 700 eligible fixtures, while the earlier expected-goals models use 619. Within the common 619 fixtures, monthly precision is 64.5%. The additional eligible fixtures account for eight correct calls among ten extra predictions.

## Contribution of expected goals

Using the same goal-regression architecture and the same 619 fixtures, adding expected-goals features increased overall accuracy from 316/619 to 323/619, or 51.1% to 52.2%. Log loss changed from 1.0066 to 1.0046. The exploratory weekly bootstrap interval for the accuracy difference is 0.00 to +2.38 percentage points, and the log-loss difference interval is −0.0061 to +0.0022. The measured gain is small, with uncertainty that includes no improvement. [Paired comparison](frozen_policy.json)

## Data, checks, and reproduction

The live Understat season pages returned an availability message on November 20, 2026. The expected-goals studies therefore use the retained archive with the previously recorded hash. The monthly model uses the existing Football-Data season files. [Understat 2026 page](https://understat.com/league/EPL/2026) · [Expected-goals source](../../data/xg_source.json) · [Snapshot source](../../data/market_snapshot_source.json)

Nine focused tests passed across the two study test files. They cover earlier-date feature construction, score probability calculations, empty selection policies, extreme probability inputs, exclusion of future results from monthly training, and predictions for unseen clubs without final scores. Saved forecast bundles reproduce their probabilities and confidence scores after reloading. Full-project regression tests were outside this change's scope.

From the repository root, use the same Python environment and dependencies as the expected-goals study.

```sh
python3 scripts/confidence_study.py
python3 scripts/team_strength_study.py
python3 scripts/compare_confidence.py
python3 -m unittest discover -s tests -p 'test_*study.py'
```

The trained bundles are saved under `models/confidence/`. The final team-strength model includes completed matches available before November 20, 2026. Historical metrics come from the monthly refits described above. Aggregate reports are under `reports/confidence/` and `reports/team_strength/`, with fixture-level exports retained locally. [Precision and coverage chart](precision_and_coverage.png)

These experiments extend the numerical component of Milestone 1. The original comparison of LLM-extracted previews with recent-match inputs remains the evidence for the LLM's contribution. [Milestone 1 results](../full/results.md)
