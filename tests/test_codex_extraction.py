import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import ALLOWED, FEATURES, digest, load_records
from extract_codex import output_schema, parse_events, resume_ids, validate_batch
import quality


def fixture_labels():
    return {side: {feature: {'value': 'unknown', 'evidence': ''} for feature in ALLOWED} for side in ['home', 'away']}


class CodexExtractionTests(unittest.TestCase):
    def test_exact_batch_coverage_and_quote_provenance(self):
        batch = [{'MatchID': 1, 'text': '[HOME] have lost three games.', 'input_sha256': digest('text')}]
        labels = fixture_labels()
        labels['home']['form'] = {'value': 'poor', 'evidence': 'lost three games'}
        response = {'records': [{'MatchID': 1, 'labels': labels}]}
        self.assertEqual(validate_batch(response, batch)[1]['home']['form']['value'], 'poor')
        with self.assertRaises(ValueError):
            validate_batch({'records': response['records'] * 2}, batch)
        labels['home']['form']['evidence'] = 'invented quote'
        with self.assertRaises(ValueError):
            validate_batch(response, batch)

    def test_rejects_tool_activity(self):
        events = [{'type': 'item.completed', 'item': {'type': 'command_execution'}}, {'type': 'turn.completed'}]
        with self.assertRaisesRegex(ValueError, 'tool activity'):
            parse_events('\n'.join(json.dumps(e) for e in events))
        events = [{'type': 'item.completed', 'item': {'type': 'agent_message'}}, {'type': 'turn.completed', 'usage': {'output_tokens': 12}}]
        self.assertEqual(parse_events('\n'.join(json.dumps(e) for e in events)), {'output_tokens': 12})

    def test_resume_checks_model_settings_and_input(self):
        source = {'MatchID': 1, 'text': 'Example', 'input_sha256': digest('Example')}
        metadata = {'model': 'fixture', 'prompt_sha256': 'prompt', 'api_version': 'cli-fixture', 'settings_sha256': 'fixed'}
        record = {'MatchID': 1, 'labels': fixture_labels(), 'input_sha256': source['input_sha256'], **metadata}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.jsonl'
            path.write_text(json.dumps(record) + '\n')
            self.assertEqual(resume_ids(path, [source], metadata), {1})
            with self.assertRaisesRegex(ValueError, 'settings_sha256'):
                resume_ids(path, [source], dict(metadata, settings_sha256='changed'))

    def test_schema_constrains_ids_and_required_fields(self):
        schema = output_schema([3, 7])
        item = schema['properties']['records']['items']
        self.assertEqual(item['properties']['MatchID']['enum'], [3, 7])
        self.assertEqual(set(item['properties']['labels']['properties']['home']['required']), set(ALLOWED))
        self.assertFalse(item['additionalProperties'])

    def test_loader_rejects_mixed_extraction_settings(self):
        sources = [{'MatchID': mid, 'text': 'Example', 'input_sha256': digest('Example')} for mid in [1, 2]]
        records = [{'MatchID': mid, 'labels': fixture_labels(), 'input_sha256': digest('Example'),
                    'model': 'fixture', 'prompt_sha256': 'prompt', 'provider': 'codex_cli',
                    'settings_sha256': str(mid)} for mid in [1, 2]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            with self.assertRaisesRegex(ValueError, 'extraction-settings'):
                load_records(path, sources)

    def test_summary_counts_batch_usage_once_and_does_not_claim_human_agreement(self):
        frame = pd.DataFrame([{'MatchID': mid, **{feature: 'unknown' for feature in FEATURES}} for mid in [1, 2]])
        records = [{'MatchID': mid, 'batch_id': 'same-batch', 'usage_at_batch_level': {'input_tokens': 100, 'output_tokens': 30}} for mid in [1, 2]]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / 'fixture.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            with patch.object(quality, 'ROOT', folder), patch.object(quality, 'load_records', return_value=(frame, {('fixture',)})), patch.object(sys, 'argv', ['quality.py', '--features', str(path), '--summary-only']):
                quality.main()
            summary = json.loads((folder / 'reports/quality/extraction_summary.json').read_text())
            self.assertEqual(summary['batch_usage_total']['input_tokens'], 100)
            self.assertEqual(summary['n_previews'], 2)
            self.assertFalse(summary['human_agreement_computed'])
            self.assertFalse((folder / 'reports/quality/pilot_agreement.json').exists())


if __name__ == '__main__':
    unittest.main()
