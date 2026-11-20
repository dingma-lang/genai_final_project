# Training toward 70 percent across 150 matches

The newly trained football-statistics model reached **96/150 correct, or 64.0%**. A separately trained model using closing betting odds reached **103/150, or 68.7%**, two correct predictions short of the target. The untrained closing-odds benchmark reached **105/150, or 70.0%**. The benchmark's result does not establish that our training achieved the target or improved on the market.

These are retrospective follow-ups on repeatedly examined seasons. Settings were selected using earlier seasons, but further design choices were informed by earlier experiments. Independent future evaluation is still needed.

## Comparison at the requested coverage

Here precision means correct predicted outcomes divided by selected matches. It is selected-match accuracy, not class-specific precision or accuracy across every fixture. Each method selected its own 150 highest-confidence predictions from the same 619 eligible matches, giving 24.2% coverage. Selection used confidence without inspecting individual match outcomes.

| Method | Correct among 150 | Precision |
| --- | ---: | ---: |
| Previous selected linear goal model | 92 | 61.3% |
| Previous CatBoost goal model | 95 | 63.3% |
| Newly selected CatBoost goal model | 96 | 64.0% |
| Newly trained closing-odds model | 103 | 68.7% |
| Closing-odds benchmark | 105 | 70.0% |

The football model gained four correct predictions over the previously selected linear model and one over the previous CatBoost goal model. The market model incorporates additional information available close to kickoff. On the same 150 fixtures chosen by that model, the untrained odds benchmark also got 103 correct. There is no measured improvement over the benchmark on that selection.

The trained market model's descriptive 95% Wilson interval is 60.9% to 75.5%. The benchmark's is 62.2% to 76.8%. These intervals do not account for correlated fixtures or repeated experimentation.

## What was trained

The football-statistics experiment tested **54 configurations** across five expanding validation seasons from 2019–20 through 2023–24. It compared shared linear Poisson and CatBoost goal models, actual-goal and mixed goal/xG targets, recent-match weighting, and additional opponent-adjusted features. Each validation season selected approximately 150/619 of its fixtures. Selection maximized pooled precision, then the weaker of the final two validation seasons, then minimized log loss.

The winner was a depth-2 CatBoost Poisson model with 300 trees, a target averaging actual goals and xG, and a 730-day training-weight half-life. It used the existing football-history features. It predicted each side's scoring rate, then converted the two rates into home-win, draw, and away-win probabilities. Validation precision was 314/424, or 74.1%.

Additional features tracked attack and defence ratings, opponent-adjusted xG over the previous eight matches, schedule strength, and expected xG from those ratings. A further variant added deep completions and pressing statistics. All updates use earlier calendar dates, with actual goals replacing missing historical xG. These feature additions did not win validation.

The market experiment tested **32 configurations** across expanding validation seasons from 2021–22 through 2023–24. It compared logistic regression and CatBoost using odds alone or odds plus match history, xG, and opponent-adjusted form. Its winner was logistic regression using two closing-odds log ratios, C of 1.0, and a 730-day training-weight half-life. It trained on 1,747 fixtures through 2023–24. Validation precision was 195/255, or 76.5%, compared with 192/255, or 75.3%, for the untrained market benchmark.

Both experiments freeze their chosen settings and confidence thresholds before calculating that run's evaluation metrics. The saved model files were reloaded and reproduced their original prediction probabilities.

## Timing and coverage limits

The common evaluation sample contains 350 eligible fixtures from 2024–25 and 269 from the available 2025–26 data. The retained xG snapshot ends April 13, 2026. Current-match xG and final scores are training targets or evaluation labels, never inputs for predicting that same match. Earlier evaluation-season results can update later fixtures' rolling history.

The market model uses AvgCA, AvgCD, and AvgCH, the archived consensus closing odds. The benchmark normalizes reciprocal odds into probabilities. This requires a forecast close to kickoff after those quotes are available. The archive does not establish an exact usable forecast timestamp for each fixture, so this result cannot be described as a day-ahead forecast. [Football-Data column definitions](https://football-data.co.uk/data.php)

Selecting exactly 150 from an entire historical batch requires all 619 confidence scores. A live policy chooses matches as they arrive and cannot guarantee the same number of predictions. The validation-frozen and rolling policies produced the following results.

| Model | Selection policy | Correct | Calls | Precision |
| --- | --- | ---: | ---: | ---: |
| Football statistics | Fixed cutoff of 0.6153 | 93 | 142 | 65.5% |
| Football statistics | Prior 350 confidence scores | 83 | 125 | 66.4% |
| Trained closing odds | Fixed cutoff of 0.6435 | 105 | 155 | 67.7% |
| Trained closing odds | Prior 350 confidence scores | 90 | 134 | 67.2% |

The fixed market policy's 105 correct predictions are out of 155 calls, so that policy also falls short of 70%.

| Batch selection by season | 2024–25 | Available 2025–26 |
| --- | ---: | ---: |
| Football statistics | 66/106, 62.3% | 30/44, 68.2% |
| Trained closing odds | 68/97, 70.1% | 35/53, 66.0% |
| Closing-odds benchmark | 72/100, 72.0% | 33/50, 66.0% |

## Connection to Milestone 1

Milestone 1 asks whether features extracted from Guardian previews improve predictions beyond recent match results. These experiments extend the numerical prediction work and do not use newly generated LLM features or human labels. They therefore do not demonstrate an LLM contribution. The original combined model's result remains 143/279, or 51.25%, on its separate 2017–18 evaluation sample. Its existing extraction-quality and historical-outcome leakage limitations still apply. [Original model comparison](../full/results.md) · [Extraction findings](../quality/full/findings.md)

## Saved files and reproduction

- [Football training script](../../scripts/opponent_form_study.py), [validation search](validation_search.csv), [frozen settings](frozen_policy.json), and [metrics](metrics.csv).
- [Market training script](../../scripts/market_precision150.py), [validation search](market/validation_search.csv), [frozen settings](market/frozen_policy.json), and [metrics](market/metrics.csv).
- Local fitted models in `models/precision150/scoring_forecaster.joblib` and `models/precision150/market_forecaster.joblib`.
- Local fixture-level predictions in `reports/precision150/predictions.csv` and `reports/precision150/market/predictions.csv`. These files and model binaries are excluded from Git.
- [Three targeted tests](../../tests/test_opponent_form_study.py) cover current/future outcome isolation, historical missing-xG fallback, and home/away feature alignment.

Run from the repository with its retained source snapshots and CatBoost installed in the Python environment.

```sh
python3 scripts/opponent_form_study.py
python3 scripts/market_precision150.py
python3 -m unittest discover -s tests -p 'test_opponent_form_study.py'
```

The football script also reads the prior `reports/shared_goals/predictions.csv` for its reference comparison. Generate it first with `python3 scripts/shared_goal_study.py` if it is absent. Source manifests and hashes remain in the existing data directory.
