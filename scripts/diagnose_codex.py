import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import BASE, ROOT, TEST_SEASON, digest, load_analysis, write_json
from extract_codex import SETTINGS, run_batch

PROMPT = '''This is a historical-knowledge diagnostic, separate from feature extraction. Each input gives only a home team, away team, and historical match date. For each MatchID, predict the historical outcome using your existing knowledge or best estimate. Use no tools, files, search, or external retrieval. Return exactly {"records":[{"i":123,"r":"H"}]}, with one record per input ID. H means home win, D draw, A away win. Do not add explanations or extra fields.'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--reasoning-effort', default='low')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--max-previews', type=int, default=279)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/codex_metadata_diagnostic.jsonl')
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_previews < 1:
        parser.error('Batch size and max previews must be positive')
    executable = shutil.which('codex')
    if not executable:
        raise SystemExit('Codex CLI is not available')
    version = subprocess.run([executable, '--version'], capture_output=True, text=True, check=True).stdout.strip()
    df = load_analysis()
    test = df[df.Season.eq(TEST_SEASON) & df[BASE].notna().all(axis=1)]
    inputs = []
    for row in test.itertuples():
        text = f'Home: {row.HomeStd}\nAway: {row.AwayStd}\nDate: {row.MatchDate.date()}'
        inputs.append({'MatchID': int(row.MatchID), 'text': text, 'input_sha256': digest(text)})
    settings = {'provider': 'codex_cli', 'model': args.model, 'reasoning_effort': args.reasoning_effort,
                'mode': 'metadata_only', 'batch_size': args.batch_size, 'temperature': None, 'overrides': SETTINGS,
                'ignore_user_config': True, 'ephemeral': True, 'sandbox': 'read-only'}
    metadata = {'model': args.model, 'prompt_sha256': digest(PROMPT), 'api_version': version,
                'settings_sha256': digest(json.dumps(settings, sort_keys=True))}
    sources = {row['MatchID']: row for row in inputs}
    seen = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            mid = row['MatchID']
            if mid in seen or mid not in sources or row['result'] not in ['H', 'D', 'A']:
                raise ValueError('Duplicate, unknown, or invalid cached diagnostic record')
            if row['input_sha256'] != sources[mid]['input_sha256'] or any(row.get(key) != value for key, value in metadata.items()):
                raise ValueError('Diagnostic provenance changed; use a different output file')
            seen.add(mid)
    write_json(args.output.with_suffix('.manifest.json'), {
        **metadata, 'settings': settings, 'prompt': PROMPT, 'test_n': len(test),
        'interpretation': 'Team identity can encode real strength; accuracy cannot prove memorization or its absence.'})
    raw_dir = args.output.parent / (args.output.stem + '_responses')
    raw_dir.mkdir(exist_ok=True)
    pending = [row for row in inputs if row['MatchID'] not in seen][:args.max_previews]
    saved = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        ids = [row['MatchID'] for row in batch]
        batch_id = digest(json.dumps(ids) + metadata['prompt_sha256'])[:16]
        results, usage, request_hash, _ = run_batch(executable, args.model, args.reasoning_effort, PROMPT, batch,
                                                  raw_dir / (batch_id + '.json'), args.timeout, metadata_only=True)
        for source in batch:
            record = {'MatchID': source['MatchID'], 'input_sha256': source['input_sha256'], **metadata,
                      'provider': 'codex_cli', 'returned_model': None, 'temperature': None,
                      'batch_id': batch_id, 'batch_MatchIDs': ids, 'request_sha256': request_hash,
                      'usage_at_batch_level': usage, 'result': results[source['MatchID']],
                      'extracted_utc': datetime.now(timezone.utc).isoformat()}
            with args.output.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            saved += 1
        print(f'Saved {saved}/{len(pending)} diagnostic predictions', flush=True)
    print(f'Metadata diagnostic output: {args.output}', flush=True)


if __name__ == '__main__':
    main()
