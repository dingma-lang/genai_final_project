# Sharing the goal-scoring model across teams

The model selected by earlier validation log loss correctly predicted **72 of 107 selected matches, or 67.3%**, covering **17.3%** of the common 619-match sample. A CatBoost goal-count variant scored **50 of 62, or 80.6%**, at **10.0% coverage**. The latter is a secondary result from the same experiment. All candidate settings and cutoffs were chosen using seasons through 2023–2024 before scoring the later seasons in this run.

The later seasons have already been examined repeatedly during the project. These results remain retrospective and exploratory. They require independent evaluation before making a claim of reliable future precision above 65%.

## Results on the same 619 fixtures

Precision in this report means accuracy among issued predictions. Coverage is the percentage of eligible fixtures receiving a prediction.

| Model | Training target | Correct / selected | Precision | Coverage | Accuracy on all 619 |
| --- | --- | ---: | ---: | ---: | ---: |
| Previous separate home/away goal models | Goals | 44 / 58 | 75.9% | 9.4% | 52.2% |
| Shared linear goal model | Goals | 72 / 102 | 70.6% | 16.5% | 52.3% |
| **Shared linear goal model, validation selection** | Mean of goals and xG | **72 / 107** | **67.3%** | **17.3%** | **52.5%** |
| Shared CatBoost goal model | Goals | 50 / 62 | 80.6% | 10.0% | 52.5% |
| Shared CatBoost goal model | Mean of goals and xG | 60 / 90 | 66.7% | 14.5% | 52.3% |

The selected model's cutoff is 63.5%. The CatBoost goals variant's cutoff is 69%. Each cutoff maximizes validation coverage subject to at least 75% precision and 10% coverage overall, plus 70% precision and 5% coverage in each of the last two validation seasons. These thresholds are predicted probabilities. Actual success rates appear in the table. [Frozen settings](frozen_policy.json) · [Class and season metrics](metrics.csv)

## Season stability and uncertainty

| Model | 2024–2025 | Available 2025–2026 matches |
| --- | ---: | ---: |
| Shared linear, goals | 54 / 75, or 72.0% | 18 / 27, or 66.7% |
| Shared linear, goals and xG | 54 / 79, or 68.4% | 18 / 28, or 64.3% |
| Shared CatBoost, goals | 34 / 43, or 79.1% | 16 / 19, or 84.2% |

The selected model's pooled illustrative 95% Wilson interval is 57.9%–75.4%. The CatBoost goals variant's interval is 69.1%–88.6%. Those intervals assume independent calls and do not adjust for repeated experiments or model comparison. The selected model falls below 65% in the smaller second-season sample. All four variants still issue zero draw predictions.

Across all 619 fixtures, the selected model improves accuracy by only two correct predictions over the previous separate goal models. The paired weekly bootstrap accuracy interval is −0.37 to +1.08 percentage points. The log-loss difference is +0.00009, with an interval from −0.00329 to +0.00341. The overall prediction gain is small and uncertain. [Paired comparison](frozen_policy.json)

## What changed

The previous approach fitted separate regressions for home and away goals. This experiment creates one row for each team in a match and fits one shared scoring function. Each row describes that team's prior form, strength, shots, expected goals, its opponent's corresponding history, and whether it is playing at home. The strength gap changes sign for the away team's row. Both teams' predicted goal rates are then converted into home-win, draw, and away-win probabilities.

Sharing the scoring function gives 6,842 team rows from 3,421 development matches. The two rows from a match remain linked during splitting and evaluation. The independent evaluation unit is a match. The comparison also removes 26 development fixtures with missing target-match xG, so every new variant uses identical training fixtures. The previous separate-model reference used all 3,447 development fixtures.

The linear model uses regularized Poisson regression. CatBoost uses boosted trees with a Poisson objective to learn interactions between own-team and opponent features. For example, it can learn how a team's chance creation relates to the opposing defence. The alternative training target is a fixed equal-weight average of goals and expected goals. Current-match goals and xG enter training targets only. Inputs contain prior-match summaries. CatBoost predictions explicitly use exponentiated outputs to obtain goal rates. [CatBoost prediction documentation](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_predict)

Development tested two linear regularization levels and four CatBoost configurations for each target. Validation selected regularization 0.1 for both linear models, and depth 3 with 300 trees for both CatBoost variants. The shared linear model using goals and xG had the best validation log loss. [Complete validation search](forward_validation.csv)

## Selection versus improved outcome prediction

Ranking each model by its own confidence gives the following retrospective comparison at a fixed budget of 62 predictions.

| Model | Correct among its 62 most confident predictions |
| --- | ---: |
| Shared CatBoost goal model, goals target | 50 / 62, or 80.6% |
| Previous separate goal models | 47 / 62, or 75.8% |
| Archived market benchmark | 46 / 62, or 74.2% |
| Previous CatBoost outcome classifier | 44 / 62, or 71.0% |

On the exact 62 fixtures selected by the shared CatBoost goal model, every reference also got 50 right. The measured difference concerns which matches receive predictions. At a budget of 107 predictions, the validation-selected model got 72 right, compared with 74 for the previous goal models and 75 for the market benchmark. The fixed-budget comparisons use the evaluation batch to choose the count and serve as diagnostics. [Matched fixtures and prediction counts](matched_comparisons.csv)

Broader confidence policies produced precision between 60.5% and 62.0%, with coverage between 26.0% and 36.8%. The benefit remains concentrated in a limited selection of fixtures. [All policies](metrics.csv) · [Cutoff search](validation_cutoffs.csv)

## Data, verification, and reproduction

The experiment retains the same 619 eligible fixtures and expected-goals archive used in the previous studies. The archive ends in April 2026, so evaluating these variants on the later 2026–2027 snapshot requires updated expected-goals histories. [Source manifest](../../data/xg_source.json) · [Coverage](../xg/coverage.csv)

Three new focused tests passed. They check the home/away row mapping, reversal of the strength gap, exclusion of current scores and xG from predictors, and alignment of scoring targets. Each saved model reproduced its predicted goal rates after reloading. Existing chronology checks reject overlapping training and validation dates. Full-project regression tests were outside this change's scope.

Run from the repository root using the same environment as the preceding studies.

```sh
python3 scripts/shared_goal_study.py
python3 -m unittest discover -s tests -p test_shared_goal_study.py
```

Saved model bundles are under `models/shared_goals/`. Aggregate outputs are under `reports/shared_goals/`, and fixture-level predictions are retained locally in that directory. The source script also produces the matched comparisons. These experiments extend the numerical portion of Milestone 1. The original preview-feature comparison remains the evidence for the LLM's contribution. [Original Milestone 1 results](../full/results.md)
