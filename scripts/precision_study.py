import hashlib
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from common import BASE, CLASSES, ROOT, write_json
from model import pipeline, probabilities, tune, weekly_bootstrap

SEASON_CODES = ['1314', '1415', '1516', '1617', '1718', '1819']
HOLDOUT = SEASON_CODES[-1]
ODDS = ['B365A', 'B365D', 'B365H']
RAW = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
       'HS', 'AS', 'HST', 'AST', *ODDS]
HISTORY_SIGNALS = ['points_per_game', 'goals_for', 'goals_against', 'shots_for', 'shots_against',
                   'on_target_for', 'on_target_against', 'at_home', 'opposition_adjusted_score']
TEAM_GROUPS = {
    'longer_form': ['points_per_game', 'goals_for', 'goals_against'],
    'venue_form': ['venue_goals_for', 'venue_goals_against'],
    'rest': ['rest_days', 'matches_in_14_days'],
    'shot_profile': ['shots_for', 'shots_against', 'on_target_for', 'on_target_against'],
}
GROUPS = {
    'elo_strength': ['elo_gap', 'absolute_elo_gap'],
    **{name: [f'{team}_{col}' for team in ['home', 'away'] for col in cols]
       for name, cols in TEAM_GROUPS.items()},
}
FEATURE_SETS = {'baseline': BASE, **{name: BASE + cols for name, cols in GROUPS.items()},
                'all_signals': BASE + [col for cols in GROUPS.values() for col in cols]}


def read_fixtures(season_codes=SEASON_CODES, extra_sources=(), extra_columns=()):
    frames = []
    sources = json.loads((ROOT / 'data/source_manifest.json').read_text())
    sources.append(json.loads((ROOT / 'data/precision_source.json').read_text()))
    sources.extend(extra_sources)
    hashes = {row['path']: row['sha256'] for row in sources}
    for season in season_codes:
        relative = f'data/raw/raw_results/E0_{season}.csv'
        path = ROOT / relative
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != hashes[relative]:
            raise ValueError(f'Unexpected source file contents: {relative}')
        columns = [*RAW, *extra_columns]
        encoding = 'utf-8-sig' if content.startswith(b'\xef\xbb\xbf') else 'latin-1'
        raw = pd.read_csv(path, encoding=encoding, usecols=columns)
        df = raw.loc[raw.Div.eq('E0')].copy()
        dates = pd.to_datetime(df.Date, format='%d/%m/%Y', errors='coerce')
        df['MatchDate'] = dates.fillna(pd.to_datetime(df.Date, format='%d/%m/%y', errors='coerce'))
        first_year = 2000 + int(season[:2])
        if len(df) != 380 or df[columns + ['MatchDate']].isna().any().any():
            raise ValueError(f'Expected 380 complete EPL matches in {season}')
        end_month = 8 if season == '1920' else 7
        if df.MatchDate.min() < pd.Timestamp(first_year, 7, 1) or df.MatchDate.max() >= pd.Timestamp(first_year + 1, end_month, 1):
            raise ValueError(f'Unexpected match dates in {season}')
        expected = np.where(df.FTHG > df.FTAG, 'H', np.where(df.FTHG < df.FTAG, 'A', 'D'))
        if not df.FTR.eq(expected).all() or not df[ODDS].gt(1).all().all():
            raise ValueError(f'Invalid outcomes or odds in {season}')
        if df.duplicated(['HomeTeam', 'AwayTeam']).any():
            raise ValueError(f'Duplicate fixture in {season}')
        df['Season'] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(['MatchDate', 'HomeTeam'])


def team_features(history, date, at_home):
    longer = history[-8:]
    shots = history[-6:]
    venue = [r for r in history if r['at_home'] == at_home][-5:]
    features = {}
    for name in ['points_per_game', 'goals_for', 'goals_against']:
        features[name] = np.mean([r[name] for r in longer])
    for name in ['goals_for', 'goals_against']:
        features['venue_' + name] = (sum(r[name] for r in venue) + 2 * features[name]) / (len(venue) + 2)
    for name in TEAM_GROUPS['shot_profile']:
        features[name] = np.mean([r[name] for r in shots])
    features['rest_days'] = min((date - history[-1]['date']).days, 21)
    features['matches_in_14_days'] = sum((date - r['date']).days <= 14 for r in history)
    return features


def match_rows(fixtures, sequence_length=0):
    rows = []
    ratings = {}
    for season, season_rows in fixtures.groupby('Season', sort=True):
        clubs = set(season_rows.HomeTeam) | set(season_rows.AwayTeam)
        ratings = {club: 1500 + 0.75 * (ratings.get(club, 1500) - 1500) for club in clubs}
        history = defaultdict(list)
        for date, day in season_rows.groupby('MatchDate', sort=True):
            played = []
            for match in day.itertuples(index=False):
                home, away = match.HomeTeam, match.AwayTeam
                gap = ratings[home] - ratings[away]
                expected_home = 1 / (1 + 10 ** (-(gap + 60) / 400))
                home_points, away_points, score = {'H': (3, 0, 1), 'D': (1, 1, 0.5), 'A': (0, 3, 0)}[match.FTR]
                if len(history[home]) >= 3 and len(history[away]) >= 3:
                    row = {'Season': season, 'MatchDate': date, 'HomeTeam': home, 'AwayTeam': away,
                           'FTR': match.FTR, 'elo_gap': gap, 'absolute_elo_gap': abs(gap),
                           **{col: getattr(match, col) for col in ODDS}}
                    for team, prefix, at_home in [(home, 'Home', True), (away, 'Away', False)]:
                        last_three = history[team][-3:]
                        row[prefix + 'Last3Pts'] = sum(r['points_per_game'] for r in last_three)
                        row[prefix + 'Last3GD'] = sum(r['goals_for'] - r['goals_against'] for r in last_three)
                        row.update({f'{prefix.lower()}_{k}': v for k, v in team_features(history[team], date, at_home).items()})
                        if sequence_length:
                            row[prefix.lower() + '_sequence'] = [
                                [previous[key] for key in HISTORY_SIGNALS] for previous in history[team][-sequence_length:]]
                    if sequence_length:
                        row['elo_expected_home'] = expected_home
                    rows.append(row)
                change = 20 * (score - expected_home)
                played.append((home, change, {'date': date, 'at_home': True, 'points_per_game': home_points,
                                               'goals_for': match.FTHG, 'goals_against': match.FTAG,
                                               'shots_for': match.HS, 'shots_against': match.AS,
                                               'on_target_for': match.HST, 'on_target_against': match.AST,
                                               'opposition_adjusted_score': score - expected_home}))
                played.append((away, -change, {'date': date, 'at_home': False, 'points_per_game': away_points,
                                                'goals_for': match.FTAG, 'goals_against': match.FTHG,
                                                'shots_for': match.AS, 'shots_against': match.HS,
                                                'on_target_for': match.AST, 'on_target_against': match.HST,
                                                'opposition_adjusted_score': expected_home - score}))
            if len(played) != len({team for team, _, _ in played}):
                raise ValueError(f'A team appeared in multiple fixtures on {date.date()}')
            for team, change, record in played:
                ratings[team] += change
                history[team].append(record)
    return pd.DataFrame(rows)


def forward_folds(history):
    folds = []
    for season in SEASON_CODES[1:-1]:
        prior = np.flatnonzero(history.Season.lt(season))
        validation = np.flatnonzero(history.Season.eq(season))
        if history.iloc[prior].MatchDate.max() >= history.iloc[validation].MatchDate.min():
            raise ValueError('Validation is not chronological')
        folds.append((prior, validation, season))
    return folds


def confidence_cutoff(history, folds, columns, params):
    maxima = []
    for prior, validation, _ in folds:
        model = pipeline(columns, 'logistic', params).fit(history.iloc[prior], history.iloc[prior].FTR)
        maxima.extend(probabilities(model, history.iloc[validation]).max(axis=1))
    return float(np.median(maxima))


def precision_row(name, policy, outcomes, probs, selected, predicted=None):
    if predicted is None:
        predicted = np.array(CLASSES)[probs.argmax(axis=1)]
    good = outcomes == predicted
    row = {'model': name, 'policy': policy, 'covered': int(selected.sum()),
           'coverage': float(selected.mean()), 'correct': int(good[selected].sum()),
           'accuracy': float(good[selected].mean()) if selected.any() else None,
           'log_loss': log_loss(outcomes[selected], probs[selected], labels=CLASSES) if selected.any() else None}
    for outcome in CLASSES:
        calls = selected & (predicted == outcome)
        actual = selected & (outcomes == outcome)
        relevant = int((calls & actual).sum())
        row[f'called_{outcome}'] = int(calls.sum())
        row[f'precision_{outcome}'] = relevant / int(calls.sum()) if calls.any() else None
        row[f'recall_{outcome}'] = relevant / int(actual.sum()) if actual.any() else None
    return row


def main():
    fixtures = read_fixtures()
    all_rows = match_rows(fixtures)
    history = all_rows.loc[all_rows.Season.lt(HOLDOUT)].reset_index(drop=True)
    test = all_rows.loc[all_rows.Season.eq(HOLDOUT)].reset_index(drop=True)
    folds = forward_folds(history)
    outcomes = test.FTR.to_numpy()
    selected_all = np.ones(len(test), dtype=bool)
    candidates = []
    configurations = {}
    for name, columns in FEATURE_SETS.items():
        params, cv_loss, details = tune(history, columns, 'logistic', folds)
        configurations[name] = {'params': params, 'cv_log_loss': cv_loss, 'columns': columns}
        candidates.extend({**row, 'feature_set': name} for row in details)
    winner = min(configurations, key=lambda name: configurations[name]['cv_log_loss'])
    cutoffs = {name: confidence_cutoff(history, folds, FEATURE_SETS[name], configurations[name]['params'])
               for name in dict.fromkeys(['baseline', winner])}

    scores = []
    predictions = test[['MatchDate', 'HomeTeam', 'AwayTeam', 'FTR']].copy()
    forecasts = {}
    for name, config in configurations.items():
        model = pipeline(config['columns'], 'logistic', config['params']).fit(history, history.FTR)
        forecasts[name] = probabilities(model, test)
    market = 1 / test[ODDS].to_numpy(dtype=float)
    forecasts['market_benchmark'] = market / market.sum(axis=1, keepdims=True)
    for name, probs in forecasts.items():
        scores.append(precision_row(name, 'all_matches', outcomes, probs, selected_all))
        for i, outcome in enumerate(CLASSES):
            predictions[f'{name}_p_{outcome}'] = probs[:, i]
        if name in cutoffs:
            selected = probs.max(axis=1) >= cutoffs[name]
            scores.append(precision_row(name, 'confidence_cutoff', outcomes, probs, selected))
            predictions[f'{name}_confidence_pick'] = selected
    contrast = weekly_bootstrap(test, forecasts['baseline'], forecasts[winner]) if winner != 'baseline' else None
    output = ROOT / 'reports/precision'
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(candidates).to_csv(output / 'forward_validation.csv', index=False)
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    write_json(output / 'run_manifest.json', {
        'train_n': len(history), 'holdout_n': len(test), 'season_counts': all_rows.Season.value_counts().sort_index().to_dict(),
        'selected_by_forward_log_loss': winner, 'models': configurations, 'confidence_cutoffs': cutoffs,
        'confidence_policy': 'Median winning probability in the forward validation folds, fixed before holdout scoring',
        'holdout_season': HOLDOUT, 'selected_minus_baseline': contrast,
        'market_probabilities': 'Normalize reciprocal decimal Bet365 odds. Market benchmark only.',
        'feature_timing': 'Features and Elo use earlier calendar dates only. Rolling team stats reset each season.',
    })
    print(f'Training {len(history)}, holdout {len(test)}. Selected by forward log loss: {winner}.')
    print(pd.DataFrame(scores)[['model', 'policy', 'covered', 'correct', 'accuracy', 'log_loss', 'called_D']].to_string(index=False))


if __name__ == '__main__':
    main()
