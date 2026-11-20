from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shared_goal_study import COLUMNS, targets, team_features


class SharedGoalTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(np.ones((2, len(COLUMNS))), columns=COLUMNS)
        self.frame['elo_gap'] = [120.0, -80.0]
        self.frame['home_xg_for'] = [1.2, 0.7]
        self.frame['away_xg_for'] = [0.6, 1.1]
        self.frame['FTHG'] = [2, 0]
        self.frame['FTAG'] = [1, 3]
        self.frame['home_xg'] = [1.5, 0.2]
        self.frame['away_xg'] = [0.5, 2.0]

    def test_away_scoring_rows_reverse_teams_and_strength_gap(self):
        rows = team_features(self.frame)
        np.testing.assert_array_equal(rows.at_home, [1, 1, 0, 0])
        np.testing.assert_array_equal(rows.elo_gap, [120, -80, -120, 80])
        np.testing.assert_allclose(rows.own_xg_for, [1.2, 0.7, 0.6, 1.1])
        np.testing.assert_allclose(rows.opponent_xg_for, [0.6, 1.1, 1.2, 0.7])

    def test_current_goals_and_expected_goals_are_excluded_from_predictors(self):
        baseline = team_features(self.frame)
        changed = self.frame.copy()
        changed[['FTHG', 'FTAG', 'home_xg', 'away_xg']] = 99
        pd.testing.assert_frame_equal(baseline, team_features(changed))
        without_outcomes = self.frame.drop(columns=['FTHG', 'FTAG', 'home_xg', 'away_xg'])
        pd.testing.assert_frame_equal(baseline, team_features(without_outcomes))

    def test_training_targets_follow_home_then_away_order(self):
        np.testing.assert_array_equal(targets(self.frame, 'goals'), [2, 0, 1, 3])
        np.testing.assert_allclose(targets(self.frame, 'goals_and_xg'), [1.75, 0.1, 0.75, 2.5])
        with self.assertRaises(ValueError):
            targets(self.frame, 'unknown')


if __name__ == '__main__':
    unittest.main()
