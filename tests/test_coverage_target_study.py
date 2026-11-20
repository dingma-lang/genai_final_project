from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from coverage_target_study import rank_selection, rolling_selection


class CoverageTargetTests(unittest.TestCase):
    def test_rank_selection_has_exact_count_and_stable_ties(self):
        values = np.array([0.8, 0.9, 0.8, 0.7])
        np.testing.assert_array_equal(rank_selection(values, 2), [True, True, False, False])
        self.assertFalse(rank_selection(values, 0).any())
        with self.assertRaises(ValueError):
            rank_selection(values, 5)

    def test_future_confidence_does_not_change_earlier_decisions(self):
        dates = pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-02', '2025-01-03'])
        values = np.array([0.7, 0.6, 0.9, 0.8])
        baseline, cutoffs = rolling_selection(dates, values, [0.5, 0.6, 0.7, 0.8], coverage=0.25, window=4)
        changed = values.copy()
        changed[-1] = 0.1
        other, other_cutoffs = rolling_selection(dates, changed, [0.5, 0.6, 0.7, 0.8], coverage=0.25, window=4)
        np.testing.assert_array_equal(baseline[:-1], other[:-1])
        np.testing.assert_array_equal(cutoffs, other_cutoffs)

    def test_same_day_matches_use_the_same_earlier_history(self):
        dates = pd.to_datetime(['2025-01-01', '2025-01-01', '2025-01-02'])
        _, cutoffs = rolling_selection(dates, [0.99, 0.9, 0.8], [0.5, 0.6], coverage=0.5, window=2)
        np.testing.assert_allclose(cutoffs, [0.6, 0.6, 0.99])

    def test_rolling_policy_requires_chronological_dates(self):
        with self.assertRaises(ValueError):
            rolling_selection(['2025-01-02', '2025-01-01'], [0.7, 0.8], [0.6])


if __name__ == '__main__':
    unittest.main()
