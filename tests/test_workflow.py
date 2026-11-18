import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import ALLOWED, BASE, ROOT, digest, load_analysis, load_records, mask_preview, read_inputs, validate_labels
from extract import parse_json, request_message
from model import CONTEXT, SENTIMENT, feature_sets, forward_folds, pipeline, probabilities
from quality import human_labels


def unknown_labels():
    return {side: {key: {'value': 'unknown', 'evidence': ''} for key in ALLOWED} for side in ['home', 'away']}


class ExtractionTests(unittest.TestCase):
    def test_masks_longest_alias_and_unicode_byline(self):
        text, change = mask_preview('Manchester United have struggled. Newcastle United look strong. Louise Taylor',
                                    {'Manchester United', 'United'}, {'Newcastle United', 'Newcastle'})
        self.assertEqual(text, '[HOME] have struggled. [AWAY] look strong.')
        self.assertIn('Louise Taylor', change)

    def test_quote_and_unknown_validation(self):
        labels = unknown_labels()
        labels['home']['form'] = {'value': 'poor', 'evidence': 'lost three games'}
        self.assertEqual(validate_labels(labels, '[HOME] lost three games.')['home_form'], 'poor')
        labels['home']['form']['evidence'] = 'won three games'
        with self.assertRaises(ValueError):
            validate_labels(labels, '[HOME] lost three games.')
        labels = unknown_labels()
        labels['home']['sentiment']['value'] = True
        with self.assertRaises(ValueError):
            validate_labels(labels, 'Any text')
        labels = unknown_labels()
        labels['home']['form']['evidence'] = 'Any'
        with self.assertRaises(ValueError):
            validate_labels(labels, 'Any text')

    def test_feature_provenance_and_duplicate_rejection(self):
        source = {'MatchID': 1, 'text': 'Some text', 'input_sha256': digest('Some text')}
        row = {'MatchID': 1, 'input_sha256': source['input_sha256'], 'model': 'fixture-model',
               'prompt_sha256': 'fixture-prompt', 'labels': unknown_labels()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.jsonl'
            path.write_text(json.dumps(row) + '\n')
            frame, _ = load_records(path, [source])
            self.assertEqual(len(frame), 1)
            path.write_text((json.dumps(row) + '\n') * 2)
            with self.assertRaises(ValueError):
                load_records(path, [source])
            row['input_sha256'] = 'different-text'
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError):
                load_records(path, [source])

    def test_fenced_json(self):
        self.assertEqual(parse_json('```json\n{"home": 1}\n```'), {'home': 1})

    def test_missing_key_never_calls_api(self):
        env = dict(os.environ)
        env.pop('ANTHROPIC_API_KEY', None)
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/extract.py'), '--model', 'fixture-model'],
                                env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ANTHROPIC_API_KEY is not set', result.stderr)

    def test_http_request_serializes_payload(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"content":[],"stop_reason":"end_turn"}'
        with patch('urllib.request.urlopen', return_value=response) as open_url:
            result = request_message({'model': 'fixture-model', 'temperature': 0}, 'not-a-real-key', '2023-06-01')
            request = open_url.call_args.args[0]
            self.assertEqual(json.loads(request.data)['temperature'], 0)
            self.assertEqual(result['stop_reason'], 'end_turn')


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analysis()
        cls.eligible = cls.analysis[cls.analysis[BASE].notna().all(axis=1)].copy()
        cls.train = cls.eligible[cls.eligible.Season.ne('2017-18')].reset_index(drop=True)
        cls.test = cls.eligible[cls.eligible.Season.eq('2017-18')].reset_index(drop=True)

    def test_fixed_cohort_and_pilot(self):
        self.assertEqual((len(self.analysis), len(self.train), len(self.test)), (1409, 1014, 279))
        pilot = set(json.loads((ROOT / 'data/pilot_ids.json').read_text()))
        self.assertEqual(len(pilot), 50)
        self.assertTrue(pilot <= set(self.train.MatchID))
        self.assertFalse(pilot & set(self.test.MatchID))
        inputs = read_inputs()
        self.assertEqual(len(inputs), 1409)
        self.assertTrue(all(set(row) == {'MatchID', 'text', 'input_sha256'} for row in inputs))
        self.assertTrue(all(row['input_sha256'] == digest(row['text']) for row in inputs))

    def test_validation_is_strictly_forward(self):
        folds = forward_folds(self.train)
        self.assertEqual(len(folds), 3)
        for fit, val, _ in folds:
            self.assertLess(self.train.iloc[fit].MatchDate.max(), self.train.iloc[val].MatchDate.min())

    def test_scaler_uses_training_only_and_drops_outcomes(self):
        fitted = pipeline(BASE, 'logistic', {'C': 0.01}).fit(self.train, self.train.FTR)
        scaler = fitted.named_steps['preprocess'].named_transformers_['numeric']
        np.testing.assert_allclose(scaler.mean_, self.train[BASE].mean())
        original = probabilities(fitted, self.test)
        altered = self.test.copy()
        altered['FTR'] = 'A'
        altered['FTHG'] = 999999
        altered['FTAG'] = -999999
        np.testing.assert_allclose(probabilities(fitted, altered), original)
        altered.loc[:, BASE] = 999999
        probabilities(fitted, altered)
        np.testing.assert_allclose(scaler.mean_, self.train[BASE].mean())

    def test_full_feature_pipelines_with_synthetic_labels(self):
        # Fixtures only: these labels are never saved as project data or reported as LLM output.
        fixture = self.train.head(80).copy()
        for side in ['home', 'away']:
            for key, values in ALLOWED.items():
                fixture[f'{side}_{key}'] = [values[i % len(values)] for i in range(len(fixture))]
        groups = feature_sets(fixture, True)
        self.assertEqual(set(groups), {'baseline', 'llm_only', 'combined', 'baseline_plus_sentiment', 'baseline_plus_context'})
        labels = np.resize(np.array(['A', 'D', 'H']), len(fixture))
        for columns in groups.values():
            for family, params in [('logistic', {'C': 0.1}), ('random_forest', {'max_depth': 3, 'min_samples_leaf': 10})]:
                fitted = pipeline(columns, family, params).fit(fixture, labels)
                pred = probabilities(fitted, fixture.tail(5))
                self.assertEqual(pred.shape, (5, 3))
                np.testing.assert_allclose(pred.sum(axis=1), 1)
                self.assertTrue(set(fitted.named_steps['preprocess'].get_feature_names_out()) <=
                                set(BASE + SENTIMENT + [f'{c}_{v}' for c in CONTEXT for v in ALLOWED[c.split('_', 1)[1]]]))

    def test_unlabeled_pilot_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'blank_pilot.csv'
            frame = pd.read_csv(ROOT / 'data/pilot_labels.csv', keep_default_na=False)
            frame['annotator'] = ''
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Annotator is blank'):
                human_labels(path)


if __name__ == '__main__':
    unittest.main()
