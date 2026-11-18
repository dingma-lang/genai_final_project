import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, TEST_SEASON, digest, load_analysis, read_inputs, validate_labels

METADATA_PROMPT = 'Given only the home team, away team, and historical fixture date, return JSON with exactly one key "result" whose value is "H", "D", or "A". H means home win, D draw, A away win. Do not request or retrieve other information. This is a historical-knowledge diagnostic.'


def parse_json(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return json.loads(text)


def request_message(payload, key, api_version):
    request = urllib.request.Request('https://api.anthropic.com/v1/messages',
                                     data=json.dumps(payload).encode(), method='POST',
                                     headers={'x-api-key': key, 'anthropic-version': api_version,
                                              'content-type': 'application/json'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in [429, 500, 502, 503, 529] and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f'Anthropic returned HTTP {exc.code}; check access, model ID, API version, and temperature support. No API key was logged.') from None


def main():
    parser = argparse.ArgumentParser(description='One extraction request per preview, with local schema/quote validation and resumable records.')
    parser.add_argument('--model', required=True, help='Explicit Claude model identifier supporting temperature=0')
    parser.add_argument('--api-version', default='2023-06-01')
    parser.add_argument('--kind', choices=['features', 'metadata'], default='features')
    parser.add_argument('--scope', choices=['pilot', 'all', 'repeat'], default='pilot')
    parser.add_argument('--max-requests', type=int, default=50, help='Maximum new input records attempted; transient HTTP errors can retry a record up to three times')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.max_requests <= 0:
        parser.error('--max-requests must be positive')
    key = os.environ.get('ANTHROPIC_API_KEY')
    if not key:
        raise SystemExit('ANTHROPIC_API_KEY is not set. Configure it locally; do not put credentials in files or chat.')
    pilot_ids = json.loads((ROOT / 'data/pilot_ids.json').read_text())
    if args.kind == 'features':
        prompt = (ROOT / 'prompts/extraction.txt').read_text()
        inputs = read_inputs()
        if args.scope != 'all':
            wanted = set(pilot_ids if args.scope == 'pilot' else pilot_ids[:20])
            inputs = [r for r in inputs if r['MatchID'] in wanted]
        default_name = 'llm_features.jsonl' if args.scope != 'repeat' else 'llm_repeat.jsonl'
    else:
        if args.scope != 'all':
            parser.error('Metadata diagnostic requires --scope all; it uses the held-out cohort only')
        prompt = METADATA_PROMPT
        df = load_analysis()
        from common import BASE
        test = df[df.Season.eq(TEST_SEASON) & df[BASE].notna().all(axis=1)].sample(frac=1, random_state=244)
        inputs = []
        for row in test.itertuples():
            text = f'Home: {row.HomeStd}\nAway: {row.AwayStd}\nDate: {row.MatchDate.date()}'
            inputs.append({'MatchID': int(row.MatchID), 'text': text, 'input_sha256': digest(text)})
        default_name = 'metadata_diagnostic.jsonl'
    output = args.output or ROOT / 'data' / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt_hash = digest(prompt)
    seen = set()
    input_map = {r['MatchID']: r for r in inputs}
    all_input_map = {r['MatchID']: r for r in read_inputs()} if args.kind == 'features' else input_map
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            if row['model'] != args.model or row['prompt_sha256'] != prompt_hash or row.get('api_version') != args.api_version:
                raise SystemExit('Existing output has a different model, API version, or prompt. Use a new output path.')
            mid = row['MatchID']
            if mid in seen or mid not in all_input_map or row['input_sha256'] != all_input_map[mid]['input_sha256']:
                raise SystemExit('Existing output has duplicate IDs or changed inputs; review the file.')
            if args.kind == 'features':
                validate_labels(row['labels'], all_input_map[mid]['text'])
            elif row.get('result') not in ['H', 'D', 'A']:
                raise SystemExit('Invalid cached metadata result')
            seen.add(mid)
    calls = failures = 0
    raw_dir = output.parent / (output.stem + '_responses')
    raw_dir.mkdir(exist_ok=True)
    for row in inputs:
        if row['MatchID'] in seen or calls >= args.max_requests:
            continue
        calls += 1
        payload = {'model': args.model, 'max_tokens': 2000 if args.kind == 'features' else 100,
                   'temperature': 0, 'system': prompt,
                   'messages': [{'role': 'user', 'content': '<preview>\n' + row['text'] + '\n</preview>' if args.kind == 'features' else row['text']}]}
        try:
            response = request_message(payload, key, args.api_version)
        except (RuntimeError, urllib.error.URLError) as exc:
            raise SystemExit(str(exc)) from None
        raw = {'MatchID': row['MatchID'], 'input_sha256': row['input_sha256'], 'prompt_sha256': prompt_hash,
               'requested_model': args.model, 'api_version': args.api_version, 'response': response}
        (raw_dir / f'{row["MatchID"]}.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2) + '\n')
        try:
            if response.get('stop_reason') != 'end_turn':
                raise ValueError('Response incomplete or stopped unexpectedly')
            text = ''.join(item['text'] for item in response['content'] if item['type'] == 'text')
            labels = parse_json(text)
            if args.kind == 'features':
                validate_labels(labels, row['text'])
            elif set(labels) != {'result'} or labels['result'] not in ['H', 'D', 'A']:
                raise ValueError('Invalid metadata diagnostic output')
        except (ValueError, KeyError, TypeError) as exc:
            failures += 1
            with (output.parent / (output.stem + '_errors.jsonl')).open('a') as stream:
                stream.write(json.dumps({'MatchID': row['MatchID'], 'error': str(exc), 'raw_response': str(raw_dir / f'{row["MatchID"]}.json')}) + '\n')
            print(f'MatchID {row["MatchID"]}: local validation failed; raw response saved.', flush=True)
            continue
        record = {'MatchID': row['MatchID'], 'input_sha256': row['input_sha256'], 'prompt_sha256': prompt_hash,
                  'model': args.model, 'returned_model': response.get('model'), 'api_version': args.api_version,
                  'temperature': 0, 'extracted_utc': datetime.now(timezone.utc).isoformat(),
                  'usage': response.get('usage'), 'response_id': response.get('id')}
        record.update({'labels': labels} if args.kind == 'features' else labels)
        with output.open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        seen.add(row['MatchID'])
        print(f'Saved MatchID {row["MatchID"]}; new requests {calls}/{args.max_requests}', flush=True)
    remaining = sum(r['MatchID'] not in seen for r in inputs)
    print(f'New requests: {calls}; validation failures: {failures}; remaining selected inputs: {remaining}; output: {output}')
    if failures:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
