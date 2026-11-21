import argparse
from pathlib import Path

from common import FEATURES, ROOT, digest, inputs_path, load_records, read_inputs, write_json


def main():
    parser = argparse.ArgumentParser(description='Export validated feature values without article text or supporting quotes.')
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/llm_feature_values.csv')
    args = parser.parse_args()
    frame, provenance = load_records(args.features, read_inputs())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame[['MatchID'] + FEATURES].to_csv(args.output, index=False)
    write_json(args.output.with_suffix('.manifest.json'), {
        'format': 'feature_values_v1', 'n_previews': len(frame),
        'values_sha256': digest(args.output.read_text()),
        'source_extraction_sha256': digest(args.features.read_text()),
        'analysis_sha256': digest((ROOT / 'data/processed/previews_with_results.csv').read_text()),
        'llm_provenance': sorted([list(row) for row in provenance]),
        'inputs_file': str(inputs_path().relative_to(ROOT)) if inputs_path().is_relative_to(ROOT) else str(inputs_path()),
        'inputs_sha256': digest(inputs_path().read_text()),
        'human_reference_validation': 'omitted',
        'note': 'AI-derived feature values only. Supporting quotes and raw previews remain local. This export supports model reproduction, not independent semantic validation.'})
    print(f'Exported {len(frame)} feature rows: {args.output}')


if __name__ == '__main__':
    main()
