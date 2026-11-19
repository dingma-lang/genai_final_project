import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import FEATURES, digest, load_feature_values


class FeatureExportTests(unittest.TestCase):
    def test_csv_loader_preserves_sentiment_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            analysis = folder / 'analysis.csv'
            analysis.write_text('fixture analysis\n')
            path = folder / 'features.csv'
            frame = pd.DataFrame([{'MatchID': mid, **{key: 'unknown' for key in FEATURES}} for mid in [1, 2]])
            frame['home_sentiment'] = [-1, 'unknown']
            frame.to_csv(path, index=False)
            manifest = {'format': 'feature_values_v1', 'n_previews': 2,
                        'values_sha256': digest(path.read_text()), 'analysis_sha256': digest(analysis.read_text()),
                        'llm_provenance': [['fixture', 'fixture', 'prompt', 'cli', 'codex_cli', 'settings']]}
            path.with_suffix('.manifest.json').write_text(json.dumps(manifest))
            loaded, _ = load_feature_values(path, analysis)
            self.assertEqual(loaded.home_sentiment.iloc[0], -1)
            self.assertEqual(loaded.away_sentiment.iloc[0], 'unknown')
            self.assertEqual(loaded.home_sentiment.iloc[1], 'unknown')
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_feature_values(path, analysis)

    def test_csv_loader_rejects_wrong_categories_and_different_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            analysis = folder / 'analysis.csv'
            analysis.write_text('fixture analysis\n')
            path = folder / 'features.csv'
            frame = pd.DataFrame([{'MatchID': 1, **{key: 'unknown' for key in FEATURES}}])
            frame['home_form'] = 'major'
            frame.to_csv(path, index=False)
            manifest = {'format': 'feature_values_v1', 'n_previews': 1,
                        'values_sha256': digest(path.read_text()), 'analysis_sha256': digest(analysis.read_text()),
                        'llm_provenance': [['fixture']]}
            path.with_suffix('.manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'category'):
                load_feature_values(path, analysis)
            analysis.write_text('different analysis\n')
            with self.assertRaisesRegex(ValueError, 'different analysis'):
                load_feature_values(path, analysis)


if __name__ == '__main__':
    unittest.main()
