"""Leakage probe: can the model identify each held-out match, and recall its result,
from a preview masked at a given level?

For every masking level, the 279 held-out (2017-18) previews are sent to Codex with the
same fixed prompt. For each preview the model returns its best guess of the home team,
away team and season, its confidence that it has identified the match, the result
(H/D/A) and a probability for each result. score_leakage.py scores the answers.

Design choices that keep levels comparable and avoid accidental hints:
  * Previews are sent under random four-digit IDs, not their MatchIDs (MatchIDs follow
    publication order and would hint at the date).
  * Batch membership and order are randomised once (fixed seed) and are identical at every
    level, so only the masking changes between levels.
  * Same Codex isolation as the feature extraction: fresh ephemeral session, empty
    folder, no tools, no web search, structured output.

Usage (from the repository root, after scripts/mask_previews.py has written data/masked/):
    python scripts/leakage_probe.py --model gpt-6.1-sol --levels L1 L2 L3 L4 L5
Try one batch first:
    python scripts/leakage_probe.py --model gpt-6.1-sol --levels L5 --max-batches 1

Outputs:
    reports/leakage/answers_<level>.jsonl     one line per match: guesses and provenance (no preview text)
    reports/leakage/probe.manifest.json       model, prompt, settings, ID scheme
    data/leakage/<level>_responses/           raw CLI responses (contain masked text; Git-ignored)
The script resumes: rerunning skips matches already answered with identical settings.
"""
import argparse
import json
import random
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from common import BASE, ROOT, TEST_SEASON, digest, load_analysis, write_json
from extract_codex import SETTINGS, parse_events

PROMPT = '''This is a research diagnostic about what a language model already knows. Each input is a short preview of an English football match, written by a journalist before kick-off. Some names, places and dates may have been replaced with placeholders such as [HOME], [AWAY], [PERSON_1], [CLUB_1], [VENUE], [YEAR] or [TIME].

For each input, answer from your own knowledge and judgement. Use no tools, files, search or external retrieval. Judge each input separately. Return one record per input ID with these fields:
- "home": your best guess of the real home team (full club name), or "unknown"
- "away": your best guess of the real away team (full club name), or "unknown"
- "season": your best guess of the season in which the match was played, written like "2016-17", or "unknown"
- "id_confidence": an integer from 0 to 100 for how confident you are that you know exactly which match this is
- "result": the result of the match: "H" home win, "D" draw or "A" away win. Always choose one, even if guessing.
- "p_home", "p_draw", "p_away": your probabilities for the three results, each between 0 and 1, summing to 1.
Return only the JSON object. Do not add explanations.'''

LEVELS = ['L1', 'L2', 'L3', 'L4', 'L5']
FIELDS = ['home', 'away', 'season', 'id_confidence', 'result', 'p_home', 'p_draw', 'p_away']


def schema(ids):
    record = {'type': 'object', 'properties': {
        'i': {'type': 'integer', 'enum': ids},
        'home': {'type': 'string'}, 'away': {'type': 'string'}, 'season': {'type': 'string'},
        'id_confidence': {'type': 'integer'},
        'result': {'type': 'string', 'enum': ['H', 'D', 'A']},
        'p_home': {'type': 'number'}, 'p_draw': {'type': 'number'}, 'p_away': {'type': 'number'}},
        'required': ['i'] + FIELDS, 'additionalProperties': False}
    return {'type': 'object', 'properties': {'records': {'type': 'array', 'items': record,
                                                      'minItems': len(ids), 'maxItems': len(ids)}},
            'required': ['records'], 'additionalProperties': False}


def validate(response, ids):
    if not isinstance(response, dict) or set(response) != {'records'} or not isinstance(response['records'], list):
        raise ValueError('Expected a records array')
    out = {}
    for row in response['records']:
        if not isinstance(row, dict) or set(row) != {'i'} | set(FIELDS):
            raise ValueError('Unexpected record fields')
        if type(row['i']) is not int or row['i'] not in ids or row['i'] in out:
            raise ValueError('Missing, duplicate or unexpected probe ID')
        if row['result'] not in ('H', 'D', 'A') or not isinstance(row['id_confidence'], int):
            raise ValueError('Invalid result or confidence')
        probs = [row['p_home'], row['p_draw'], row['p_away']]
        if any(not isinstance(p, (int, float)) or isinstance(p, bool) or p < 0 for p in probs) or sum(probs) <= 0:
            raise ValueError('Invalid probabilities')
        total = sum(probs)
        row = dict(row, p_home=probs[0] / total, p_draw=probs[1] / total, p_away=probs[2] / total,
                   probability_sum_returned=total, id_confidence=max(0, min(100, row['id_confidence'])))
        out[row['i']] = row
    if set(out) != set(ids):
        raise ValueError('Response does not cover every probe ID in the batch')
    return out


def call_codex(executable, model, effort, batch, raw_path, timeout):
    """batch: list of (probe_id, text). Same isolation as extract_codex.run_batch."""
    ids = [pid for pid, _ in batch]
    prompt = PROMPT + '\n\nINPUTS:\n' + json.dumps([{'i': pid, 'text': text} for pid, text in batch], ensure_ascii=False)
    with tempfile.TemporaryDirectory(prefix='epl-leakage-probe-') as folder:
        folder = Path(folder)
        schema_path, final = folder / 'schema.json', folder / 'response.json'
        schema_path.write_text(json.dumps(schema(ids)))
        command = [executable, 'exec', '--ignore-user-config', '--ephemeral', '--skip-git-repo-check',
                   '--sandbox', 'read-only', '--cd', str(folder), '--model', model,
                   '-c', 'model_reasoning_effort=' + json.dumps(effort),
                   '--output-schema', str(schema_path), '--output-last-message', str(final), '--json']
        for name, value in SETTINGS.items():
            command += ['-c', name + '=' + ('{}' if isinstance(value, dict) else json.dumps(value))]
        command += ['-']
        try:
            result = subprocess.run(command, input=prompt, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            write_json(raw_path, {'probe_ids': ids, 'error': 'timeout', 'command': command})
            raise RuntimeError(f'Codex batch timed out; diagnostics: {raw_path}') from None
        response_text = final.read_text() if final.exists() else None
        write_json(raw_path, {'probe_ids': ids, 'requested_model': model, 'command': command,
                             'request_sha256': digest(prompt), 'returncode': result.returncode,
                             'stdout': result.stdout, 'stderr': result.stderr, 'response_text': response_text})
        if result.returncode != 0 or response_text is None:
            raise RuntimeError(f'Codex failed with exit code {result.returncode}; diagnostics: {raw_path}')
        usage = parse_events(result.stdout)   # rejects any tool activity
        answers = validate(json.loads(response_text), ids)
    return answers, usage, digest(prompt)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', required=True)
    parser.add_argument('--reasoning-effort', default='low')
    parser.add_argument('--levels', nargs='+', default=LEVELS, choices=LEVELS)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--max-batches', type=int, help='Stop after this many new batches per level (for a trial run)')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--seed', type=int, default=244)
    parser.add_argument('--masked-dir', type=Path, default=ROOT / 'data/masked')
    args = parser.parse_args()
    executable = shutil.which('codex')
    if not executable:
        raise SystemExit('Codex CLI is not on PATH')
    version = subprocess.run([executable, '--version'], capture_output=True, text=True, check=True).stdout.strip()

    df = load_analysis()
    test = df[df.Season.eq(TEST_SEASON) & df[BASE].notna().all(axis=1)]
    test_ids = sorted(int(x) for x in test.MatchID)
    if len(test_ids) != 279:
        raise ValueError(f'Expected the 279 held-out matches, found {len(test_ids)}')
    rng = random.Random(args.seed)
    probe_ids = dict(zip(test_ids, rng.sample(range(1000, 10000), len(test_ids))))
    order = test_ids[:]
    rng.shuffle(order)
    batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]

    settings = {'provider': 'codex_cli', 'model': args.model, 'reasoning_effort': args.reasoning_effort,
                'batch_size': args.batch_size, 'seed': args.seed, 'temperature': None, 'overrides': SETTINGS,
                'ignore_user_config': True, 'ephemeral': True, 'sandbox': 'read-only',
                'id_scheme': 'random four-digit probe IDs, fixed by seed, identical across levels',
                'batching': 'one seeded random order, identical across levels'}
    meta = {'model': args.model, 'prompt_sha256': digest(PROMPT), 'api_version': version,
            'settings_sha256': digest(json.dumps(settings, sort_keys=True))}
    out_dir = ROOT / 'reports/leakage'
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / 'probe.manifest.json', {**meta, 'settings': settings, 'prompt': PROMPT, 'n_matches': 279,
                                                 'note': 'Answers are model guesses; raw responses stay in data/leakage/.'})

    for level in args.levels:
        source = args.masked_dir / f'llm_inputs_{level}.jsonl'
        if not source.exists():
            raise SystemExit(f'{source} not found. Run scripts/mask_previews.py first.')
        texts = {r['MatchID']: r for r in map(json.loads, source.read_text().splitlines())}
        missing = set(test_ids) - set(texts)
        if missing:
            raise ValueError(f'{level}: {len(missing)} held-out matches missing from {source}')
        answers_path = out_dir / f'answers_{level}.jsonl'
        done = set()
        if answers_path.exists():
            for row in map(json.loads, answers_path.read_text().splitlines()):
                if any(row.get(k) != v for k, v in meta.items()) or row['input_sha256'] != texts[row['MatchID']]['input_sha256']:
                    raise ValueError(f'{answers_path} was made with different settings or text; move it aside first')
                done.add(row['MatchID'])
        raw_dir = ROOT / 'data/leakage' / f'{level}_responses'
        raw_dir.mkdir(parents=True, exist_ok=True)
        new_batches = 0
        for batch in batches:
            if all(mid in done for mid in batch):
                continue
            if args.max_batches is not None and new_batches >= args.max_batches:
                break
            pairs = [(probe_ids[mid], texts[mid]['text']) for mid in batch]
            batch_id = digest(json.dumps([p for p, _ in pairs]) + meta['prompt_sha256'] + level)[:16]
            answers, usage, request_hash = call_codex(executable, args.model, args.reasoning_effort, pairs,
                                                      raw_dir / f'{batch_id}.json', args.timeout)
            stamp = datetime.now(timezone.utc).isoformat()
            with answers_path.open('a') as stream:
                for mid in batch:
                    a = answers[probe_ids[mid]]
                    stream.write(json.dumps({'MatchID': mid, 'probe_id': probe_ids[mid], 'level': level,
                                             'input_sha256': texts[mid]['input_sha256'], **meta,
                                             'batch_id': batch_id, 'request_sha256': request_hash,
                                             'usage_at_batch_level': usage, 'answered_utc': stamp,
                                             **{k: a[k] for k in FIELDS},
                                             'probability_sum_returned': a['probability_sum_returned']}) + '\n')
                    done.add(mid)
            new_batches += 1
            print(f'{level}: {len(done)}/279 answered', flush=True)
        print(f'{level}: finished with {len(done)}/279 answered -> {answers_path}', flush=True)


if __name__ == '__main__':
    main()
