# Starting goalkeepers and short-term team form

The Football-Data files have a dated match timeline for each club. Opta's starting lineups provide the missing goalkeeper identities. Matching by season, calendar date, and home and away teams yielded a common cohort of 2,076 development matches and 346 of 350 otherwise eligible 2019–2020 holdout matches. The existing numerical model was refitted to that same cohort. [Football-Data season files](https://football-data.co.uk/englandm.php) · [Opta lineup archive](https://github.com/peteowen1/pannadata) · [Join coverage](coverage.csv)

| Model | Correct | Accuracy | Log loss |
| --- | ---: | ---: | ---: |
| Existing Elo and match statistics | 184/346 | **53.18%** | **0.9818** |
| Existing model + RNN form | 182/346 | 52.60% | 0.9834 |
| Existing model + goalkeeper features | 179/346 | 51.73% | 0.9862 |
| Existing model + both | 178/346 | 51.45% | 0.9878 |

The existing model led in development and on the fresh holdout. The paired exploratory 95% intervals for log-loss changes included zero for all three additions. The archive offers proxy measures, so the effect of individual keeper quality remains unresolved. RNN home-win precision rose slightly from 54.1% to 54.4%, while away-win precision fell from 51.0% to 49.1%. [Metrics by outcome](metrics.csv) · [Validation folds](forward_validation.csv) · [Configuration and paired comparisons](run_manifest.json)

## What the team form mark means

A shared gated recurrent unit (GRU) reads the most recent three to eight league games for each side, oldest first. Its inputs include goals, points, shots, shots on target, venue, and how each previous result compared with the team's Elo expectation. The GRU estimates how much a team's next result score might exceed or fall short of Elo's expectation. Teams are marked **strong** at +0.05 or above, **weak** at −0.05 or below, and **steady** in between. These cutoffs were fixed before scoring the holdout. The labels describe estimated form relative to Elo. A strong club can temporarily receive a weak form mark.

For example, the GRU marked Man United weak before Southampton on August 31 (−0.095), steady before Leicester on September 14 (−0.007), and strong before Norwich on October 27 (+0.051). Each score was generated using only earlier matches as sequence inputs. The script saves `team_form_timeline.csv` and `predictions_and_team_status.csv` locally. The timeline gives one row per club and fixture, including its opponent, venue, starting goalkeeper, score, and mark. [Man United example](team_form_example.png)

## What the goalkeeper features measure

The model uses the announced starter, whether he changed since the previous league game, and his share of known starts in the club's last six league matches. It also uses his prior full-match experience and a shrunken proxy for save share from up to ten prior full matches. The proxy uses team shots on target faced minus goals conceded, floored at zero. It mixes individual goalkeeping with defense and shot quality. The archive identifies who started, while the injury or tactical reason for a change is unknown.

This is a **post-lineup, pre-kickoff** experiment. The 2019–2020 rules kept team sheets embargoed until 60 minutes before kickoff. Only starting identity from the current lineup enters prediction. We used minutes, shots, and goals from earlier dates to build the keeper histories. A day-ahead variant would need time-stamped availability reports or expected lineups. A stronger keeper study could use prior individual saves and post-shot expected goals, which would better distinguish keeping from the defense in front of the keeper. [Premier League 2019–2020 Handbook, Rule K.84](https://premierleague-static-files.s3.amazonaws.com/premierleague/document/2019/08/02/add6ee50-25c6-4474-8854-edf8a56c23c1/2019-20-PL-Handbook-020819.pdf)

We changed target-day scores, shots, and keeper minutes in a timing check. Features for the target day and all earlier dates stayed identical, and later features changed. Both saved GRU pipelines reproduced all exported holdout probabilities and status labels. The evaluation covers one held-out season.

## Reproduce

Run `python3 scripts/keeper_form_study.py` from the repository root with the previous precision-study files and the three additional files listed in the source manifest. In addition to the repository's existing packages, this run used PyTorch 2.14.0 and PyArrow 26.0.0. The script checks the source hashes. The Opta release tag is refreshed, so reuse the local archive snapshots if later downloads have different hashes. [Source URLs and hashes](../../data/goalkeeper_sources.json)
