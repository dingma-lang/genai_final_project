from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from confidence_study import calibration_inputs, choose_policy, wilson_interval
from team_strength_study import fit_strength, predict_rates


class ConfidenceTests(unittest.TestCase):
    def test_wilson_interval_and_empty_policy(self):
        lower, upper = wilson_interval(44, 58)
        self.assertAlmostEqual(lower, 0.6347001975)
        self.assertAlmostEqual(upper, 0.8504112564)
        self.assertEqual(wilson_interval(0, 0), (None, None))
        seasons = np.array(['2223'] * 100 + ['2324'] * 100)
        probs = np.tile([0.2, 0.2, 0.6], (200, 1))
        outcomes = np.array(['D'] * 200)
        policy, candidates = choose_policy(outcomes, seasons, probs, probs.max(axis=1))
        self.assertIsNone(policy)
        self.assertTrue(candidates)

    def test_calibration_handles_extreme_probabilities(self):
        features = calibration_inputs(np.array([[0.0, 0.0, 1.0], [0.2, 0.5, 0.3]]))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[1], np.log([0.4, 0.6]))


class TeamStrengthTests(unittest.TestCase):
    def test_future_scores_do_not_affect_training_or_prediction(self):
        fixtures = pd.DataFrame({'HomeTeam': ['Alpha', 'Beta'] * 8, 'AwayTeam': ['Beta', 'Alpha'] * 8,
                                 'MatchDate': pd.date_range('2024-01-01', periods=16, freq='7D'),
                                 'FTHG': [2, 1] * 8, 'FTAG': [1, 0] * 8})
        cutoff = pd.Timestamp('2024-03-01')
        baseline = fit_strength(fixtures, cutoff, 0.01, 365)
        modified = fixtures.copy()
        modified.loc[modified.MatchDate.ge(cutoff), ['FTHG', 'FTAG']] = 99
        other = fit_strength(modified, cutoff, 0.01, 365)
        future = pd.DataFrame({'HomeTeam': ['Alpha', 'New club'], 'AwayTeam': ['Beta', 'Alpha']})
        predicted = predict_rates(baseline, future)
        np.testing.assert_allclose(predicted, predict_rates(other, future), rtol=1e-12)
        self.assertEqual(predicted.shape, (2, 2))
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue((predicted > 0).all())


if __name__ == '__main__':
    unittest.main()
