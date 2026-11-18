import json

import pandas as pd

from common import BASE, FEATURES, ROOT, TRAIN_SEASONS, TEST_SEASON, digest, load_analysis, mask_preview, write_json


def main():
    df = load_analysis()
    eligible = df.loc[df[BASE].notna().all(axis=1)].copy()
    if (len(df), len(eligible)) != (1409, 1293):
        raise ValueError('Data differ from the Milestone 1 cohort; review before modeling')
    train = eligible[eligible.Season.isin(TRAIN_SEASONS)]
    test = eligible[eligible.Season.eq(TEST_SEASON)]
    if (len(train), len(test)) != (1014, 279) or train.MatchDate.max() >= test.MatchDate.min():
        raise ValueError('Unexpected chronological split')

    fixtures = pd.read_csv(ROOT / 'data/processed/epl_results_2013_2018.csv')
    fixtures.Date = pd.to_datetime(fixtures.Date)
    for row in eligible.itertuples():
        for team, prefix in [(row.HomeStd, 'Home'), (row.AwayStd, 'Away')]:
            prior = fixtures[(fixtures.Season.eq(row.Season)) & (fixtures.Date.lt(row.MatchDate)) &
                             (fixtures.HomeTeam.eq(team) | fixtures.AwayTeam.eq(team))].sort_values('Date').tail(3)
            if len(prior) != 3:
                raise ValueError('Baseline lacks three strictly earlier matches')
            at_home = prior.HomeTeam.eq(team)
            pts = ((prior.FTR.eq('H') & at_home) | (prior.FTR.eq('A') & ~at_home)).sum() * 3 + prior.FTR.eq('D').sum()
            gd = ((prior.FTHG - prior.FTAG).where(at_home, prior.FTAG - prior.FTHG)).sum()
            if pts != getattr(row, prefix + 'Last3Pts') or gd != getattr(row, prefix + 'Last3GD'):
                raise ValueError(f'Baseline mismatch for MatchID {row.MatchID}')

    mappings = pd.read_csv(ROOT / 'data/processed/team_name_map.csv')
    alias_map = mappings.groupby('StandardName').GuardianSpelling.apply(list).to_dict()
    extras = {'Man United': ['United', 'Man Utd', 'Manchester Utd'], 'Man City': ['City'],
              'West Brom': ['Albion', 'Baggies'], 'Tottenham': ['Spurs'],
              'Newcastle': ['Magpies'], 'West Ham': ['Hammers'], 'Crystal Palace': ['Palace'],
              'Southampton': ['Saints'], 'Bournemouth': ['Cherries'], 'Swansea': ['Swans'],
              'Leicester': ['Foxes'], 'Stoke': ['Potters'], 'Sunderland': ['Black Cats'],
              'Huddersfield': ['Terriers'], 'Brighton': ['Albion', 'Seagulls']}
    records, masking_log = [], []
    for row in df.itertuples():
        home = set(alias_map[row.HomeStd] + [row.HomeStd] + extras.get(row.HomeStd, []))
        away = set(alias_map[row.AwayStd] + [row.AwayStd] + extras.get(row.AwayStd, []))
        ambiguous = home & away
        home -= ambiguous
        away -= ambiguous
        text, change = mask_preview(row.Text, home, away)
        records.append({'MatchID': int(row.MatchID), 'text': text, 'input_sha256': digest(text)})
        masking_log.append({'MatchID': row.MatchID, 'change': change, 'ambiguous_aliases': ', '.join(sorted(ambiguous)),
                            'has_home_token': '[HOME]' in text, 'has_away_token': '[AWAY]' in text})
    (ROOT / 'data/llm_inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    pd.DataFrame(masking_log).to_csv(ROOT / 'data/masking_log.csv', index=False)

    pilot = train.sample(n=50, random_state=244).sort_values(['MatchDate', 'MatchID'])
    texts = {r['MatchID']: r['text'] for r in records}
    labels = pd.DataFrame({'MatchID': pilot.MatchID, 'Text': pilot.MatchID.map(texts)})
    for feature in FEATURES:
        labels[feature] = ''
        labels[feature + '_evidence'] = ''
    labels['annotator'] = ''
    labels['notes'] = ''
    pilot_path = ROOT / 'data/pilot_labels.csv'
    if not pilot_path.exists():
        labels.to_csv(pilot_path, index=False)
    elif pd.read_csv(pilot_path).MatchID.tolist() != labels.MatchID.tolist():
        raise ValueError('Existing pilot differs; not overwriting annotations')
    write_json(ROOT / 'data/pilot_ids.json', [int(x) for x in pilot.MatchID])
    split = eligible[['MatchID', 'FixtureID', 'Season', 'MatchDate']].copy()
    split['split'] = split.Season.map(lambda x: 'test' if x == TEST_SEASON else 'train')
    split.to_csv(ROOT / 'data/split_manifest.csv', index=False)
    write_json(ROOT / 'reports/data_validation.json', {
        'matched_previews': len(df), 'eligible': len(eligible), 'train': len(train), 'test': len(test),
        'pilot': len(pilot), 'pilot_from_training_only': True, 'baseline_values_recomputed_from_strictly_prior_fixtures': True,
        'no_duplicate_match_or_fixture_ids': True, 'all_previews_before_match_date': True,
        'analysis_sha256': digest((ROOT / 'data/processed/previews_with_results.csv').read_text()),
        'test_policy': 'Final season held out. Hyperparameters selected by earlier-season forward validation.',
        'masking_limit': 'Heuristic alias/byline masking is partial. Players, managers, venues, and indirect clues remain; masking is not proof against memorization.'})
    print(f'Prepared {len(records)} masked inputs; pilot 50; train {len(train)}; test {len(test)}.')


if __name__ == '__main__':
    main()
