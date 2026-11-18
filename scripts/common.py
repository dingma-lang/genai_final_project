import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ['HomeLast3Pts', 'HomeLast3GD', 'AwayLast3Pts', 'AwayLast3GD']
CLASSES = ['A', 'D', 'H']
TRAIN_SEASONS = ['2013-14', '2014-15', '2015-16', '2016-17']
TEST_SEASON = '2017-18'
ALLOWED = {
    'sentiment': [-2, -1, 0, 1, 2, 'unknown'],
    'form': ['poor', 'mixed', 'good', 'unknown'],
    'availability': ['none_mentioned', 'minor', 'major', 'unknown'],
    'motivation': ['low', 'normal', 'high', 'unknown'],
    'managerial_uncertainty': ['yes', 'no', 'unknown'],
}
FEATURES = [f'{side}_{key}' for side in ['home', 'away'] for key in ALLOWED]


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).split())


def validate_labels(obj, text):
    if set(obj) != {'home', 'away'}:
        raise ValueError('Expected exactly home and away objects')
    flat = {}
    for side in ['home', 'away']:
        if not isinstance(obj[side], dict) or set(obj[side]) != set(ALLOWED):
            raise ValueError(f'{side}: missing or extra feature keys')
        for key, allowed in ALLOWED.items():
            item = obj[side][key]
            if not isinstance(item, dict) or set(item) != {'value', 'evidence'}:
                raise ValueError(f'{side}.{key}: expected value and evidence')
            value, evidence = item['value'], item['evidence']
            if isinstance(value, bool) or value not in allowed:
                raise ValueError(f'{side}.{key}: invalid value {value!r}')
            if key == 'sentiment' and value != 'unknown' and not isinstance(value, int):
                raise ValueError('Sentiment must be an integer or unknown')
            if not isinstance(evidence, str):
                raise ValueError('Evidence must be a string')
            if value == 'unknown' and evidence != '':
                raise ValueError('Unknown requires empty evidence')
            if value != 'unknown' and (not evidence.strip() or normalized(evidence) not in normalized(text)):
                raise ValueError(f'{side}.{key}: evidence is not a quote from the supplied preview')
            flat[f'{side}_{key}'] = value
    return flat


def load_analysis(path=None):
    frame = pd.read_csv(path or ROOT / 'data/processed/previews_with_results.csv')
    if frame.empty or frame.MatchID.duplicated().any() or frame.FixtureID.duplicated().any():
        raise ValueError('Analysis data are empty or contain duplicate IDs')
    for col in ['PubDate', 'MatchDate']:
        frame[col] = pd.to_datetime(frame[col], errors='raise')
    if not frame.FTR.isin(CLASSES).all() or not frame.PubDate.lt(frame.MatchDate).all():
        raise ValueError('Invalid result or preview not strictly before match date')
    if frame.Text.isna().any():
        raise ValueError('Missing preview text')
    return frame.sort_values(['MatchDate', 'MatchID']).reset_index(drop=True)


def load_records(path, inputs):
    records = []
    seen = set()
    input_map = {int(row['MatchID']): row for row in inputs}
    provenance = set()
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        mid = int(row['MatchID'])
        if mid in seen or mid not in input_map:
            raise ValueError(f'Duplicate or unrecognized feature MatchID: {mid}')
        seen.add(mid)
        source = input_map[mid]
        if row['input_sha256'] != source['input_sha256']:
            raise ValueError(f'Features are for different masked text: {mid}')
        if not row.get('model') or not row.get('prompt_sha256'):
            raise ValueError('Missing feature provenance')
        provenance.add((row['model'], row.get('returned_model') or row['model'], row['prompt_sha256'], row.get('api_version', 'unspecified')))
        records.append({'MatchID': mid, **validate_labels(row['labels'], source['text'])})
    if len(provenance) != 1:
        raise ValueError('Use one model and one frozen prompt for an analysis')
    return pd.DataFrame(records), provenance


def read_inputs(path=None):
    return [json.loads(line) for line in (Path(path) if path else ROOT / 'data/llm_inputs.jsonl').read_text().splitlines()]


def mask_preview(text, home_aliases, away_aliases):
    text = normalized(text)
    changes = []
    byline = re.search(r'(?<=[.!?])\s+([A-ZÀ-Ý][\wÀ-ÿ’\'-]+(?:\s+[A-ZÀ-Ý][\wÀ-ÿ’\'-]+){1,3})\s*$', text)
    if byline:
        changes.append('removed trailing byline: ' + byline.group(1))
        text = text[:byline.start()].strip()
    aliases = [(a, '[HOME]') for a in home_aliases] + [(a, '[AWAY]') for a in away_aliases]
    aliases.sort(key=lambda pair: len(pair[0]), reverse=True)
    replacements = {alias.casefold(): token for alias, token in aliases}
    pattern = r'(?<!\w)(?:' + '|'.join(re.escape(a) for a, _ in aliases) + r')(?!\w)'
    text = re.sub(pattern, lambda m: replacements[m.group().casefold()], text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:19|20)\d{2}\b', '[YEAR]', text)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b', '[DATE]', text)
    months = 'January February March April May June July August September October November December'.split()
    text = re.sub(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:' + '|'.join(months) + r')\b', '[DATE]', text, flags=re.I)
    return text, '; '.join(changes)
