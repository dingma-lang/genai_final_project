from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from xg_study import BASE_COLUMNS, PRESSURE_COLUMNS, XG_COLUMNS, build_features, choose_policy, prior_features
from goal_score_study import score_probabilities


def fixtures():
    rows = []
    for index in range(12):
        home, away = ('Alpha', 'Beta') if index % 2 == 0 else ('Beta', 'Alpha')
        rows.append({'Season': '2425', 'MatchDate': pd.Timestamp('2024-08-01') + pd.Timedelta(days=7 * index),
                     'HomeTeam': home, 'AwayTeam': away, 'FTR': 'H', 'FTHG': 2, 'FTAG': 1,
                     'HS': 14, 'AS': 9, 'HST': 5, 'AST': 3, 'B365A': 4, 'B365D': 3.5, 'B365H': 1.8,
                     'home_xg': 1.7, 'away_xg': 0.9, 'home_deep': 8, 'away_deep': 4,
                     'home_ppda': 10, 'away_ppda': 15})
    return pd.DataFrame(rows)


class ExpectedGoalsTests(unittest.TestCase):
    def test_current_and_future_statistics_do_not_change_earlier_features(self):
        source = fixtures()
        baseline = build_features(source)
        cutoff = source.MatchDate.iloc[7]
        changed = source.copy()
        changed.loc[changed.MatchDate.ge(cutoff), ['FTHG', 'HS', 'HST', 'home_xg', 'home_deep', 'home_ppda']] *= 2
        other = build_features(changed)
        columns = BASE_COLUMNS + XG_COLUMNS + PRESSURE_COLUMNS
        earlier = baseline.MatchDate.le(cutoff)
        pd.testing.assert_frame_equal(baseline.loc[earlier, columns], other.loc[earlier, columns])
        self.assertFalse(baseline.loc[~earlier, columns].equals(other.loc[~earlier, columns]))

    def test_missing_current_xg_does_not_remove_current_prediction(self):
        source = fixtures()
        target = source.MatchDate.iloc[7]
        source.loc[source.MatchDate.eq(target), ['home_xg', 'away_xg']] = np.nan
        result = build_features(source)
        self.assertIn(target, set(result.MatchDate))
        self.assertNotIn(source.MatchDate.iloc[8], set(result.MatchDate))
        self.assertIn(source.MatchDate.iloc[9], set(result.MatchDate))

    def test_insufficient_history_is_unavailable(self):
        self.assertIsNone(prior_features([], True))
        self.assertIsNone(prior_features([{'xg_for': 1.0}] * 2, True))

    def test_cutoff_has_no_policy_when_validation_misses_target(self):
        outcomes = np.array(['D'] * 20)
        seasons = np.array(['2223'] * 10 + ['2324'] * 10)
        forecasts = np.tile([0.1, 0.2, 0.7], (20, 1))
        policy, grid = choose_policy(outcomes, seasons, forecasts)
        self.assertIsNone(policy)
        self.assertTrue(grid)
        self.assertFalse(any(row['eligible'] for row in grid))

    def test_score_probabilities_use_away_draw_home_order(self):
        probabilities = score_probabilities([[3.0, 0.5], [0.5, 3.0], [1.0, 1.0]], 0.0)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_allclose(probabilities[0], probabilities[1, ::-1])
        self.assertGreater(probabilities[0, 2], probabilities[0, 0])
        self.assertAlmostEqual(probabilities[2, 0], probabilities[2, 2])

    def test_low_score_correction_preserves_mass_and_increases_draw_chance(self):
        rates = [[1.0, 1.0], [8.0, 0.05], [0.05, 8.0]]
        baseline = score_probabilities(rates, 0)
        corrected = score_probabilities(rates, -0.1)
        self.assertTrue((corrected >= 0).all())
        np.testing.assert_allclose(corrected.sum(axis=1), 1)
        self.assertTrue((corrected[:, 1] > baseline[:, 1]).all())


if __name__ == '__main__':
    unittest.main()
