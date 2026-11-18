import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import ALLOWED, FEATURES, ROOT, TEST_SEASON, digest, load_analysis, load_records, read_inputs, validate_labels, write_json


def agreement(left, right):
    joined = left.merge(right, on='MatchID', suffixes=('_left', '_right'), validate='one_to_one')
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError('Agreement requires the same IDs in both files')
    rows = []
    for feature in FEATURES:
        a, b = joined[feature + '_left'].astype(str), joined[feature + '_right'].astype(str)
        kappa = float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else None
        if kappa is not None and not np.isfinite(kappa):
            kappa = None
        rows.append({'feature': feature, 'n': len(a), 'exact_agreement': float(a.eq(b).mean()),
                     'cohen_kappa': kappa, 'left_unknown_share': float(a.eq('unknown').mean()),
                     'right_unknown_share': float(b.eq('unknown').mean())})
    return rows


def human_labels(path):
    df = pd.read_csv(path, keep_default_na=False)
    expected = set(json.loads((ROOT / 'data/pilot_ids.json').read_text()))
    if df.MatchID.duplicated().any() or set(df.MatchID) != expected:
        raise ValueError('Human labels must cover exactly the 50 fixed training-only pilot IDs')
    inputs = {r['MatchID']: r for r in read_inputs()}
    rows = []
    for row in df.to_dict('records'):
        if not str(row['annotator']).strip():
            raise ValueError('Annotator is blank. This sheet has not been independently labeled by a human.')
        labels = {'home': {}, 'away': {}}
        for feature in FEATURES:
            side, key = feature.split('_', 1)
            value = row[feature]
            if key == 'sentiment' and str(value) in ['-2', '-1', '0', '1', '2']:
                value = int(value)
            labels[side][key] = {'value': value, 'evidence': row[feature + '_evidence']}
        flat = validate_labels(labels, inputs[row['MatchID']]['text'])
        rows.append({'MatchID': row['MatchID'], **flat})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, default=ROOT / 'data/llm_features.jsonl')
    parser.add_argument('--human-labels', type=Path)
    parser.add_argument('--repeat', type=Path)
    parser.add_argument('--metadata', type=Path)
    args = parser.parse_args()
    out = ROOT / 'reports/quality'
    out.mkdir(parents=True, exist_ok=True)
    if args.human_labels or args.repeat:
        features, provenance = load_records(args.features, read_inputs())
        coverage = []
        for feature in FEATURES:
            coverage.append({'feature': feature, 'n': len(features), 'unknown_share': float(features[feature].eq('unknown').mean())})
        pd.DataFrame(coverage).to_csv(out / 'feature_coverage.csv', index=False)
        if args.human_labels:
            human = human_labels(args.human_labels)
            model = features[features.MatchID.isin(human.MatchID)]
            rows = agreement(human, model)
            write_json(out / 'pilot_agreement.json', {'n_previews': len(human), 'feature_results': rows,
                       'human_labels_sha256': digest(args.human_labels.read_text()),
                       'feature_file_sha256': digest(args.features.read_text()),
                       'llm_provenance': sorted([list(x) for x in provenance]),
                       'interpretation': 'Inspect disagreements and supporting quotes; no automatic pass threshold was specified by Milestone 1. Agreement can be inflated by unknown labels.'})
            joined = human.merge(model, on='MatchID', suffixes=('_human', '_llm'))
            disagreements = []
            for row in joined.to_dict('records'):
                for feature in FEATURES:
                    if str(row[feature + '_human']) != str(row[feature + '_llm']):
                        disagreements.append({'MatchID': row['MatchID'], 'feature': feature,
                                              'human': row[feature + '_human'], 'llm': row[feature + '_llm']})
            pd.DataFrame(disagreements, columns=['MatchID', 'feature', 'human', 'llm']).to_csv(out / 'pilot_disagreements.csv', index=False)
        if args.repeat:
            repeat, repeat_provenance = load_records(args.repeat, read_inputs())
            if repeat_provenance != provenance:
                raise ValueError('Repeat must use the same model and prompt')
            selected = features[features.MatchID.isin(repeat.MatchID)]
            write_json(out / 'repeat_agreement.json', {'n_previews': len(repeat), 'feature_results': agreement(selected, repeat)})
    if args.metadata:
        analysis = load_analysis()
        predicted = pd.read_csv(ROOT / 'reports/baseline/test_predictions.csv')
        records = [json.loads(line) for line in args.metadata.read_text().splitlines()]
        if not records or len({r['MatchID'] for r in records}) != len(records):
            raise ValueError('Empty or duplicate metadata predictions')
        for row in records:
            match = analysis.loc[analysis.MatchID.eq(row['MatchID'])]
            if len(match) != 1 or match.iloc[0].Season != TEST_SEASON or row['result'] not in ['H', 'D', 'A']:
                raise ValueError('Invalid metadata diagnostic row')
            match = match.iloc[0]
            text = f'Home: {match.HomeStd}\nAway: {match.AwayStd}\nDate: {match.MatchDate.date()}'
            if row['input_sha256'] != digest(text):
                raise ValueError('Metadata input hash mismatch')
        diagnostic = pd.DataFrame(records)[['MatchID', 'result']].merge(predicted, on='MatchID', validate='one_to_one')
        if len(diagnostic) != len(records):
            raise ValueError('Metadata diagnostic must use the fixed held-out cohort')
        write_json(out / 'metadata_diagnostic.json', {
            'n': len(diagnostic), 'metadata_only_accuracy': float(diagnostic.result.eq(diagnostic.FTR).mean()),
            'baseline_accuracy_same_subset': float(diagnostic.logistic__baseline_prediction.eq(diagnostic.FTR).mean()),
            'always_home_accuracy_same_subset': float(diagnostic.FTR.eq('H').mean()),
            'limitation': 'Metadata-only accuracy cannot prove memorization: team identities encode legitimate strength and home advantage. This diagnostic also cannot establish the absence of outcome contamination in extraction.'})
    if not (args.human_labels or args.repeat or args.metadata):
        parser.error('Provide --human-labels, --repeat, or --metadata')
    print(f'Quality outputs: {out}')


if __name__ == '__main__':
    main()
