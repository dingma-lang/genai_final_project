# Expected goals and selective match prediction

The goal-count model correctly predicted **44 of 58 selected matches, or 75.9%**, at **9.4% coverage**. CatBoost with expected-goals features and a stricter confidence cutoff reached **54 of 81, or 66.7%**, at **13.1% coverage**. Both cutoffs were chosen using earlier validation seasons. Across all 619 eligible matches, the goal-count model reached 52.2% accuracy and the previous CatBoost model reached 51.4%.

These are retrospective follow-up experiments on seasons already examined during the project. They support further testing of selective predictions. An independent future evaluation is needed to establish how reliably either policy clears 65%.

## Results on the same eligible matches

The evaluation contains 350 matches in 2024–2025 and 269 matches in the available portion of 2025–2026. Accuracy among issued predictions is the percentage of those predictions that were correct. Coverage is the percentage of the 619 eligible matches receiving a prediction. The class-specific precision and recall values are in the metrics files.

| Model and policy | Correct / predicted | Accuracy among predictions | Coverage |
| --- | ---: | ---: | ---: |
| Previous CatBoost, every eligible match | 318 / 619 | 51.4% | 100% |
| CatBoost with expected goals, every eligible match | 315 / 619 | 50.9% | 100% |
| Goal-count model, every eligible match | 323 / 619 | 52.2% | 100% |
| CatBoost with expected goals, confidence at least 66.5% | 54 / 81 | 66.7% | 13.1% |
| Goal-count model, confidence at least 70% | 44 / 58 | 75.9% | 9.4% |

The goal-count policy scored 34/44 in 2024–2025 and 10/14 in 2025–2026. Its illustrative 95% Wilson interval is 63.5%–85.0%, assuming independent calls. The CatBoost policy's interval is 55.9%–76.0%. Team and calendar dependence can affect uncertainty. Both models selected zero draws, so draw identification remains unresolved. [Metrics by season and outcome](metrics.csv)

## What changed

Expected goals measure the quality of chances created and conceded. The new features summarize each team's previous eight available league matches, weighting recent matches more heavily. They include expected goals for and against, actual goals minus expected goals, and home/away expected-goal averages. The original Elo, recent results, shots, and rest features also remain inputs. History can carry across seasons.

The first experiment compared logistic regression and CatBoost, with and without expected-goal features, and tested extra pressure measures from the archive. It also tested giving recent training matches greater weight. Configuration selection used expanding-season validation through 2023–2024. CatBoost with expected goals had the best validation log loss, but its initial confidence policy achieved only 122/206 correct calls in the later sample. Extra pressure measures and recency weighting did not win validation. [All six feature/model comparisons](../xg/metrics.csv) · [Configuration search](../xg/forward_validation.csv)

The second experiment trained two regularized Poisson regressions to estimate home and away goal counts. It converted a grid of possible scores into home-win, draw, and away-win probabilities. Development compared three regularization levels, two training recency policies, and an optional Dixon–Coles adjustment for low-scoring results. Validation selected regularization 0.1, uniform training weights, and a zero low-score correction. [Frozen settings](frozen_policy.json) · [Dixon and Coles, 1997](https://doi.org/10.1111/1467-9876.00065)

For the stricter policy, the cutoff maximized validation coverage while requiring at least 75% accuracy among predictions and 10% coverage overall, plus 70% accuracy and 5% coverage in each of the last two validation seasons. The resulting cutoffs were 70% for the goal-count model and 66.5% for CatBoost. These are model probability thresholds. The measured success rates appear in the table above. [All cutoff candidates](validation_cutoffs.csv)

## How much improvement comes from selection

At an identical budget of 58 predictions, ranking each model by its own confidence yielded the following results. This ranking is a retrospective comparison that uses the evaluation batch to set the prediction count.

| Model | Correct among its 58 most confident predictions |
| --- | ---: |
| Previous CatBoost | 41 / 58, or 70.7% |
| Previous Elo logistic regression | 41 / 58, or 70.7% |
| Archived market benchmark | 43 / 58, or 74.1% |
| CatBoost with expected goals | 44 / 58, or 75.9% |
| Goal-count model | 44 / 58, or 75.9% |

On the exact 58 fixtures selected by the goal-count policy, every reference also got 44 right. Much of the higher precision comes from selecting strong favorites. Evidence of an advantage over the market remains limited. The trained models use football statistics and earlier expected goals. Market odds provide a comparison. [Matched-fixture and matched-count comparisons](metrics.csv)

Across all 619 matches, the goal-count model improved accuracy by 0.81 percentage points over the previous CatBoost model. An exploratory calendar-week bootstrap interval runs from −0.97 to +2.40 points. Log loss improved from 1.0190 to 1.0046, with a paired difference interval from −0.0232 to −0.0049. The accuracy gain remains uncertain, and the interval does not account for the full sequence of project experiments. [Paired comparisons](frozen_policy.json)

## Data and timing checks

The source is the locally retained Understat metadata archive from pannadata. Its SHA-256 matches the published release digest. The loader removes 189 identical duplicate EPL records and rejects conflicting records. It joins on date and both teams, validates one-to-one matches, and requires agreement on matched scores. Exact-date coverage is incomplete in two early seasons, and the archive ends on April 13, 2026. [Source and hash](../../data/xg_source.json) · [Season coverage](../xg/coverage.csv) · [Public release](https://github.com/peteowen1/pannadata/releases/tag/understat-latest)

Eligibility requires at least three known expected-goal matches among each team's preceding eight league matches, including its most recent league match. It also retains the existing requirement for three earlier in-season results. Availability of the target match's expected goals never controls whether that match receives a forecast. All rolling inputs use earlier calendar dates. This leaves 619 of the previous 700 evaluation fixtures eligible. The analysis uses archived historical xG estimates whose original publication timestamps are unavailable.

Six targeted tests check timing, missing-history behavior, probability ordering, normalization, and the low-score correction. Saved classifier and goal-count models reproduced their exported probabilities after reloading. Full-project regression tests were outside this change's scope.

## Milestone 1 and reproduction

Milestone 1 asks whether LLM-extracted preview features add value beyond recent results. These experiments extend the numerical comparison with chance-quality inputs and selective predictions. The LLM's contribution remains measured by the original preview-feature comparison. [Original Milestone 1 results](../full/results.md)

Run the following from the repository root in an environment with the project dependencies, CatBoost, and PyArrow installed. This run used CatBoost 1.2.10 and PyArrow 26.0.0. The retained raw season files, xG snapshot, and previous precision-study predictions are required. The source manifests identify the raw files and their hashes.

```sh
python3 scripts/xg_study.py
python3 scripts/goal_score_study.py
python3 -m unittest discover -s tests -p test_xg_study.py
```

The scripts save fitted models under `models/xg/` and fixture-level exports as `predictions.csv` in their respective report directories. The aggregate outputs are under `reports/xg/` and `reports/goal_score/`. [Comparison chart](accuracy_and_coverage.png)
