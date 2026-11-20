# Draw prediction and lineup continuity

The existing Elo and match-statistics model remained the best choice in forward validation for both forecasts. The draw features produced the same accuracy and worse log loss on the 2021–2022 holdout. Keeper continuity slightly improved held-out log loss by 0.00025 on the lineup comparison, with a 95% paired interval from −0.00323 to +0.00251. The lineup model with all continuity features scored worse than its numerical reference.

## Forecast using earlier matches

Each team needed three previous matches in the season. Development used 2,798 fixtures through 2020–2021, with seven expanding-season validation folds. The holdout was 350 matches in 2021–2022, including 80 draws. Predictors use results and stats from calendar dates before each target fixture. This approximates a forecast made at the start of match day. The cutoff is based on dates and can be less than 24 hours before kickoff. [Football-Data.co.uk season files](https://football-data.co.uk/englandm.php)

| Signals | Accuracy | Log loss | Correct among 35 matches with the highest draw probability |
| --- | ---: | ---: | ---: |
| Existing Elo and history | **53.1%** | **0.9894** | 7 |
| Add draw signals | 53.1% | 0.9949 | 5 |
| Add first-half scoring | 52.6% | 0.9918 | 8 |
| Add corners | 51.7% | 0.9907 | 6 |
| Add discipline | 52.9% | 0.9903 | 9 |
| Add all four groups | 52.9% | 0.9999 | 7 |

The draw signals combine the two teams' venue-specific scoring and conceding rates into estimates of scoring total and scoring gap. An interaction emphasizes low expected scoring when the home-adjusted Elo gap is small. A fourth feature is the teams' recent draw rate. First-half scoring, corners, fouls, and cards are averaged over the last six available league fixtures within the season. The four groups were compared individually and together. Discipline's 9 of 35 is descriptive for this holdout. [Probability and class metrics](day_ahead/metrics.csv) · [Forward validation](day_ahead/forward_validation.csv)

The normal Elo model called one draw and got it wrong. We also tried assigning draw whenever its estimated probability was at least 29%. The threshold was selected to maximize draw F1 while keeping the loss in development accuracy within two percentage points. In the held-out season, the rule called 61 draws and got 14 right. Draw precision was **23.0%** and recall was **17.5%**. Overall accuracy fell from **53.1% to 48.9%**. The held-out accuracy loss exceeded the development limit, and the precision of the draw calls was close to the season's 22.9% draw rate. The decision rule leaves the underlying probabilities and log loss unchanged. The script saves `test_predictions.csv` locally for each forecast. [Threshold settings](day_ahead/run_manifest.json)

## Forecast after the starting lineups are announced

The paired comparison used valid starting XIs for both teams, a confirmed XI in each team's preceding fixture, and at least three known XIs among its last six. All 350 numerically eligible holdout matches passed these checks. The numerical reference was refitted on the same 2,716 development fixtures as the lineup variants, so its accuracy here differs from the earlier forecast. Both starting XI and club histories come from the Opta archive. [Opta lineup archive](https://github.com/peteowen1/pannadata) · [Data coverage](lineup_coverage.csv)

| Inputs on the same fixtures | Accuracy | Log loss |
| --- | ---: | ---: |
| Same-fixture numerical reference | **53.7%** | 0.9862 |
| Add keeper continuity | 53.4% | **0.9859** |
| Add outfield continuity | 52.6% | 0.9902 |
| Add both lineup groups | 52.3% | 0.9903 |

Keeper inputs flag a starter change and count the starter's share of the club's recent starts. Outfield inputs measure changes from the preceding XI, regular players missing from the current XI, and overlap in center-back groups. A regular is a player who started at least two-thirds of the known recent lineups. Center-back positions were usable for all 700 team sides in the holdout. The lineup data show who started and leave the reasons for absences unknown. The full lineup model correctly called two of its nine draw predictions, covering only two of 80 actual draws. [Lineup metrics](post_lineup/metrics.csv) · [Paired intervals](post_lineup/run_manifest.json)

The goalkeeper inputs made the highest-probability draw shortlist a little better on this season (11 of 35 versus 9 of 35). With a small probability gain, worse overall accuracy, and worse development log loss, keep this finding exploratory. The result concerns keeper continuity and leaves individual save quality unmeasured.

## Milestone 1 connection and verification

The original Milestone 1 question and its 2017–2018 text-feature comparison are still documented in the original results. This extension tests football features using a separate 2021–2022 holdout. In a timing check, changing target-day goals, shots, corners, and cards affected only later numerical inputs. Changing an announced goalkeeper changed the current post-lineup forecast's features and left earlier fixtures intact. Both forecasts used identical holdout keys and outcomes, and the saved probability rows and decision counts matched the metrics. [Milestone 1 modeling results](../full/results.md)

## Reproduce

Run `python3 scripts/draw_lineup_study.py` from the repository root after preparing the precision and goalkeeper source files and downloading the two additional seasons in the source manifest. The study verifies source hashes and uses the existing Python dependencies plus PyArrow. The numerical and lineup predictions, validation folds, metrics, and run settings are saved under this report's directory. [Additional source files](../../data/draw_lineup_sources.json)
