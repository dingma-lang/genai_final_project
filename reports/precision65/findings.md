# Testing a 65% target for EPL match calls

The football-only models fell short of 65% on two fresh full-season holdouts. A separate model using football statistics and closing odds reached 87.5% on **7 of 8** high-confidence calls in a later snapshot, covering **11.4%** of its eligible fixtures. Closing odds alone also got 7 of those same 8 matches right. The small snapshot provides preliminary evidence for selective forecasting. It leaves the trained model's incremental value over the market unresolved.

Here, accuracy among issued calls is the share of calls that were correct across the three outcomes. We report coverage as the share of fixtures that received a call. Macro precision gives home wins, draws, and away wins equal weight. The metrics CSVs include class-level precision, recall, and call counts.

## Football-only models

We fitted regularized logistic regression and CatBoost on identical pre-match inputs: Elo, previous results, home and away form, shot statistics, and rest. The actual result is the training target. Features use matches completed on earlier calendar dates. Neither trained model uses text, current-match statistics, or odds.

We chose configurations by expanding-season validation log loss through 2023–2024. For each model, the cutoff maximized validation coverage while requiring at least 69% validation precision and 15% coverage. The two holdout seasons were 2024–2025 and 2025–2026. We kept both the model weights and cutoffs fixed. Earlier holdout matches could update later fixtures' team histories.

| Forecast | Accuracy on all 700 matches | Accuracy on selected matches | Coverage |
| --- | ---: | ---: | ---: |
| Elo logistic regression | 50.6% (354/700) | 60.3% (140/232) | 33.1% |
| CatBoost | 50.6% (354/700) | 58.5% (138/236) | 33.7% |
| Archived Bet365 pre-closing odds | 51.3% (359/700) | 60.6% (206/340) | 48.6% |

Development selected the Elo model for coverage. Its cutoff was 57%, and CatBoost's was 58%. CatBoost used depth 3 and 200 trees. On all holdout fixtures, Elo log loss was 1.0179 against CatBoost's 1.0230. CatBoost did not call a draw. Numeric features are available earlier in the day than some archived odds quotes, so timing matters when using the market benchmark. [Frozen settings](frozen_policy.json) · [Metrics by season and outcome](metrics.csv)

## Market-aware follow-up

The second model combines the same football features with normalized consensus closing-odds information. Its forecast represents a decision just before kickoff. The model and cutoff were selected from completed seasons through 2025–2026, with a stricter requirement: at least 72% precision and 12% coverage across development validation folds, and at least 65% precision and 10% coverage in both recent seasons. The trained model's selected cutoff was 62.5%. The 72% development result is part of cutoff selection.

The evaluation uses a separate 2026–2027 snapshot frozen on November 19, 2026. The original file contains 100 completed games through November 8. Of these, 70 have three prior in-season matches for both teams and usable odds. [Closing odds column definitions](https://football-data.co.uk/data.php) · [Snapshot dates and counts](market_followup/snapshot_coverage.json)

| Forecast | Accuracy on all 70 | Accuracy on selected matches | Coverage |
| --- | ---: | ---: | ---: |
| Football plus closing odds | 45.7% (32/70) | **87.5% (7/8)** | 11.4% |
| Closing odds alone | 48.6% (34/70) | 85.7% (6/7) | 10.0% |

On the **same eight fixtures** selected by the trained model, the closing market also got seven right. The trained model selected seven home wins and one away win, with zero draw calls. An illustrative 95% Wilson interval for 7 of 8 is 52.9%–97.8%, treating calls as independent. The snapshot gives a useful next measurement point, and another independent batch is needed to judge whether the policy consistently clears 65%. The study saves the fixture-level outputs locally as `predictions.csv` in each report directory. [Frozen market-aware settings](market_followup/frozen_policy.json)

**Presentation wording.** “We set a 65% goal and tested it on all games and on higher-confidence calls. CatBoost did not reach it in the two full-season tests. The market-aware model reached 7 of 8 correct calls in a later snapshot, covering 11% of eligible games. Closing odds alone got the same number right on those fixtures.”

The two models are structured-data extensions to the original Milestone 1 question. The original comparison evaluated whether LLM-extracted previews added value. [Milestone 1 results](../full/results.md)

## Reproduce

Run `python3 scripts/train_precision65.py` and then `python3 scripts/train_market_confidence.py` from the project directory. The environment used CatBoost 1.2.10 and the repository's existing Python packages. The studies check source hashes. The current-season script needs the retained snapshot because the live source continues changing. Both saved football-only pipelines and the saved market-aware model reproduced the probability exports within numerical tolerance. [Additional season sources](../../data/confidence_sources.json) · [Snapshot source](../../data/market_snapshot_source.json)
