# Improving EPL prediction with pre-match statistics

The expanded logistic model correctly predicted **202 of 350 matches (57.7%)** in a new 2018–2019 holdout. The three-match baseline correctly predicted **187 (53.4%)** on those same fixtures. Log loss fell from 0.9979 to 0.9121. The change in accuracy was +4.3 percentage points, with an exploratory 95% calendar-week bootstrap interval of +0.6 to +8.0 points. This interval describes one season.

| Signals added to the three-match baseline | Accuracy | Log loss |
| --- | ---: | ---: |
| None | 53.4% | 0.9979 |
| Elo strength and gap between teams | 56.3% | 0.9142 |
| Longer form, with goals scored and conceded separately | 54.9% | 0.9533 |
| Home and away scoring form | 55.4% | 0.9619 |
| Days since last league match and league fixture congestion | 54.0% | 1.0029 |
| Shots and shots on target, for and against | 53.7% | 0.9528 |
| All added signals | **57.7%** | **0.9121** |
| Bet365 pre-match odds benchmark | 58.3% | 0.8965 |

Elo was the strongest single addition. Prior-match shot numbers improved probability estimates with little improvement in the most likely outcome. League rest and congestion alone increased log loss. The odds benchmark was ahead on both metrics and used normalized inverse odds. [Metrics and per-class results](metrics.csv)

**Precision by outcome.** Home-win precision rose from 54.9% to 57.5% and away-win precision from 48.8% to 57.0%. The expanded model called three draws and got all three right, but caught only 3 of the 65 actual draws. The baseline called zero draws. Draw prediction remains a weakness.

**When we can skip a match.** The expanded model's confidence cutoff was 52.7%, set from the median top probability in the forward validation folds. It called **193 of 350 matches (55.1% coverage)** and got **134 right (69.4%)**. The baseline got 127 of those same 193 fixtures right (65.8%). This policy trades coverage for accuracy on the matches called. The confidence scores refer to model probabilities.

## How the experiment was run

The study used 1,750 EPL league fixtures from 2013–2014 through 2017–2018 for development and 350 from 2018–2019 as the holdout. Each included match requires three earlier league games for both clubs in that season. It uses tabular data from Football-Data.co.uk. Preview features and current-match betting odds are excluded from the fitted models. The odds serve as a separate benchmark and may reflect information available closer to kickoff. [Source season files](https://football-data.co.uk/englandm.php)

The model used only information from earlier calendar dates. The study uses up to eight prior games for longer form, six for shot profiles, and five same-venue games for venue form. Venue goal averages are shrunk toward the team's longer form using two equivalent games. Elo carries across seasons with a 25% regression toward 1500 and uses fixed parameters of K=20 and a 60-point home advantage. Rest is capped at 21 days and the congestion count covers league games in the previous 14 days. Recent match form resets each season. Each feature group was evaluated in four expanding-window development folds. The combination and its regularization were selected using pooled development log loss before scoring the holdout. The script saves per-match probabilities locally as `test_predictions.csv`. [Run settings](run_manifest.json) · [Forward validation](forward_validation.csv)

The new data preparation reproduced all four original baseline values on the original 1,293 matched fixtures. When we changed the targets and shot stats of one date in memory, features for that date and earlier dates stayed identical. Later dates changed as expected. The 2018–2019 holdout is larger than the original article-matched 2017–2018 holdout, so compare the models within this study.

## Additional football factors to test

- **Starting goalkeeper and lineup changes.** Test if the regular keeper is available, how many usual starters are missing, and whether the center-back pairing has changed. Use an archived team sheet if the forecast is made one hour before kickoff. A day-ahead model would need injury reports and expected lineups published by then. [Premier League 2018–2019 Handbook, Rule K.84](https://premierleague-static-files.s3.amazonaws.com/premierleague/document/2018/09/19/ef4c62c9-6b19-4189-a064-d212cbf68ce7/PL_Handbook_2018-19_Digital.pdf)
- **Shot quality and goalkeeper performance.** Prior expected goals for and against can help distinguish good chances from a high volume of weak shots. Prior post-shot expected goals and saves can help separate goalkeeper performance from defensive workload if historical coverage is sufficient.
- **Cup and European fixtures.** Add days since the last competitive match, minutes played, and rotation across competitions. The league-only rest feature misses that workload.

These extensions need matching historical data and a declared forecast time. Further model selection should use development seasons, with another fresh holdout for confirmation.

## Reproduce

After downloading the original project source data, add the official 2018–2019 EPL file and run the study from the repository root. The script checks all six source hashes.

```sh
curl -fL https://www.football-data.co.uk/mmz4281/1819/E0.csv -o data/raw/raw_results/E0_1819.csv
python3 scripts/precision_study.py
```
