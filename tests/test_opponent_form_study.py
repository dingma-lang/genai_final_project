from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from opponent_form_study import FORM_SIGNALS, opponent_form, scoring_features
from shared_goal_study import COLUMNS


class OpponentFormTests(unittest.TestCase):
    def test_current_and_future_results_leave_earlier_features_unchanged(self):
        dates = pd.date_range('2024-08-01', periods=8, freq='7D').repeat(2)
        fixtures = pd.DataFrame({'Season': '2425', 'MatchDate': dates,
                                 'HomeTeam': ['Alpha', 'Gamma'] * 8, 'AwayTeam': ['Beta', 'Delta'] * 8,
                                 'FTHG': 2, 'FTAG': 1, 'home_xg': 1.6, 'away_xg': 0.9})
        baseline = opponent_form(fixtures)
        cutoff = dates[8]
        changed = fixtures.copy()
        changed.loc[changed.MatchDate.ge(cutoff), ['FTHG', 'FTAG', 'home_xg', 'away_xg']] = 5.0
        other = opponent_form(changed)
        earlier = baseline.MatchDate.le(cutoff)
        pd.testing.assert_frame_equal(baseline.loc[earlier], other.loc[earlier])
        self.assertFalse(baseline.loc[~earlier].equals(other.loc[~earlier]))

    def test_missing_expected_goals_use_prior_scores_without_missing_features(self):
        fixtures = pd.DataFrame({'Season': ['2425', '2425'], 'MatchDate': pd.to_datetime(['2024-08-01', '2024-08-08']),
                                 'HomeTeam': ['Alpha', 'Alpha'], 'AwayTeam': ['Beta', 'Beta'],
                                 'FTHG': [3, 1], 'FTAG': [0, 1], 'home_xg': [np.nan, 1.0], 'away_xg': [np.nan, 1.0]})
        result = opponent_form(fixtures)
        columns = [side + '_' + signal for side in ['home', 'away'] for signal in FORM_SIGNALS]
        self.assertTrue(np.isfinite(result[columns]).all().all())
        self.assertGreater(result.home_attack_rating.iloc[1], result.home_attack_rating.iloc[0])
        self.assertGreater(result.away_defence_rating.iloc[1], result.away_defence_rating.iloc[0])

    def test_extended_team_rows_preserve_home_and_away_mapping(self):
        frame = pd.DataFrame(np.ones((1, len(COLUMNS))), columns=COLUMNS)
        for side, value in [('home', 2.0), ('away', 3.0)]:
            for signal in FORM_SIGNALS:
                frame[side + '_' + signal] = value
        result = scoring_features(frame, 'opponent_form')
        np.testing.assert_array_equal(result.own_attack_rating, [2, 3])
        np.testing.assert_array_equal(result.opponent_attack_rating, [3, 2])


if __name__ == '__main__':
    unittest.main()
