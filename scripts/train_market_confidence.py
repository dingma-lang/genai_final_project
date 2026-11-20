import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from common import CLASSES, ROOT, write_json
from model import pipeline, probabilities
from precision_study import RAW, match_rows, read_fixtures
from train_precision65 import COLUMNS, SEASONS, measure

CLOSING = ['AvgCA', 'AvgCD', 'AvgCH']
ODDS_COLUMNS = ['close_log_away_vs_draw', 'close_log_home_vs_draw']
FEATURES = [*COLUMNS, *ODDS_COLUMNS]
VALIDATION_SEASONS = ['2122', '2223', '2324', '2425', '2526']
RECENT = ['2425', '2526']


def build_features(fixtures):
    keys = ['Season', 'MatchDate', 'HomeTeam', 'AwayTeam']
    rows = match_rows(fixtures).merge(fixtures[keys + CLOSING], on=keys, validate='one_to_one')
    rows = rows.loc[rows[CLOSING].gt(1).all(axis=1)].copy()
    rows[ODDS_COLUMNS[0]] = np.log(rows.AvgCD / rows.AvgCA)
    rows[ODDS_COLUMNS[1]] = np.log(rows.AvgCD / rows.AvgCH)
    return rows


def market_probability(rows):
    inverse = 1 / rows[CLOSING].to_numpy(dtype=float)
    return inverse / inverse.sum(axis=1, keepdims=True)


def train(frame, penalty):
    return pipeline(FEATURES, 'logistic', {'C': penalty}).fit(frame[FEATURES], frame.FTR)


def choose_policy(outcomes, seasons, probs):
    correct = np.array(CLASSES)[probs.argmax(axis=1)] == outcomes
    grid = []
    for cutoff in np.linspace(0.45, 0.90, 91):
        selected = probs.max(axis=1) >= cutoff
        if not selected.any():
            continue
        overall = float(correct[selected].mean())
        coverage = float(selected.mean())
        row = {'cutoff': float(cutoff), 'precision': overall, 'coverage': coverage}
        recent_pass = True
        for season in RECENT:
            cohort = seasons == season
            picked = selected & cohort
            row[f'{season}_coverage'] = float(picked.sum() / cohort.sum())
            row[f'{season}_precision'] = float(correct[picked].mean()) if picked.any() else None
            use = row[f'{season}_precision']
            recent_pass &= use is not None and use >= 0.65 and row[f'{season}_coverage'] >= 0.10
        row['eligible'] = recent_pass and overall >= 0.72 and coverage >= 0.12
        grid.append(row)
    feasible = [row for row in grid if row['eligible']]
    return (max(feasible, key=lambda row: (row['coverage'], row['precision'])) if feasible else None), grid


def load_snapshot():
    source = json.loads((ROOT / 'data/market_snapshot_source.json').read_text())
    path = ROOT / source['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('Current-season source changed')
    snapshot = pd.read_csv(path, encoding='utf-8-sig', usecols=[*RAW, *CLOSING])
    snapshot['MatchDate'] = pd.to_datetime(snapshot.Date, format='%d/%m/%Y', errors='raise')
    snapshot = snapshot.loc[snapshot.Div.eq('E0') & snapshot.FTR.isin(CLASSES)
                            & snapshot.MatchDate.le(source['cutoff_date'])].copy()
    if snapshot.empty or snapshot[RAW].isna().any().any():
        raise ValueError('Incomplete snapshot results')
    if snapshot.MatchDate.min() < pd.Timestamp('2026-07-01') or snapshot.duplicated(['HomeTeam', 'AwayTeam']).any():
        raise ValueError('Invalid current-season fixture identities')
    records = np.where(snapshot.FTHG > snapshot.FTAG, 'H', np.where(snapshot.FTHG < snapshot.FTAG, 'A', 'D'))
    if not snapshot.FTR.eq(records).all():
        raise ValueError('Snapshot score and outcome disagree')
    snapshot['Season'] = '2627'
    return source, snapshot


def main():
    sources = []
    for name in ['goalkeeper_sources', 'draw_lineup_sources', 'confidence_sources']:
        sources.extend(json.loads((ROOT / 'data' / (name + '.json')).read_text()))
    warmup = read_fixtures(SEASONS[:6], sources)
    recent = read_fixtures(SEASONS[6:], sources, extra_columns=CLOSING)
    fixtures = pd.concat([warmup, recent], ignore_index=True)
    development = build_features(fixtures)
    folds = [(development.loc[development.Season.lt(season)], development.loc[development.Season.eq(season)])
             for season in VALIDATION_SEASONS]
    if any(fit.empty or val.empty or fit.MatchDate.max() >= val.MatchDate.min() for fit, val in folds):
        raise ValueError('Invalid validation chronology')
    outcomes = pd.concat([val.FTR for _, val in folds]).to_numpy()
    seasons = pd.concat([val.Season for _, val in folds]).to_numpy()
    candidates = []
    for penalty in [0.01, 0.1, 1.0, 10.0]:
        by_fold = []
        for fit, val in folds:
            by_fold.append(probabilities(train(fit, penalty), val[FEATURES]))
        probs = np.vstack(by_fold)
        candidates.append((float(log_loss(outcomes, probs, labels=CLASSES)), penalty, probs))
    score, penalty, probs = min(candidates, key=lambda candidate: candidate[0])
    development_forecasts = {'football_plus_closing_odds': probs,
                             'closing_market': np.vstack([market_probability(val) for _, val in folds])}
    policies = {}
    candidates_table = []
    for name, forecast in development_forecasts.items():
        policy, grid = choose_policy(outcomes, seasons, forecast)
        policies[name] = policy
        candidates_table.extend({'model': name, **entry} for entry in grid)

    output = ROOT / 'reports/precision65/market_followup'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'frozen_policy.json', {
        'forecast_time': 'After the archived closing quote and before kickoff.',
        'odds_columns_in_A_D_H_order': CLOSING, 'development_ends': '2526',
        'development_matches': len(development), 'validation_seasons': VALIDATION_SEASONS,
        'validation_matches': len(outcomes), 'selection_log_loss': score, 'selected_C': penalty, 'model_features': FEATURES,
        'requirements': {'pooled_validation_precision': 0.72, 'pooled_validation_coverage': 0.12,
                         'each_recent_season_precision': 0.65, 'each_recent_season_coverage': 0.10, 'recent_seasons': RECENT},
        'policy': policies, 'snapshot_date': '2026-11-19'})
    pd.DataFrame(candidates_table).to_csv(output / 'validation_cutoffs.csv', index=False)
    print('Frozen market-aware policies:', policies, flush=True)

    source, fresh = load_snapshot()
    if development.MatchDate.max() >= fresh.MatchDate.min():
        raise ValueError('Snapshot is not later than development')
    test = build_features(pd.concat([fixtures, fresh], ignore_index=True))
    test = test.loc[test.Season.eq('2627')].reset_index(drop=True)
    fitted = train(development, penalty)
    model_path = ROOT / 'models/precision65/football_plus_closing_odds.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, model_path)
    forecasts = {'football_plus_closing_odds': probabilities(fitted, test[FEATURES]), 'closing_market': market_probability(test)}
    predictions = test[['MatchDate', 'HomeTeam', 'AwayTeam', 'FTR']].copy()
    metrics = []
    for name, values in forecasts.items():
        labels = test.FTR.to_numpy()
        metrics.append(measure(name, '2627_snapshot', labels, values, 'all_fixtures', np.ones(len(test), dtype=bool)))
        policy = policies[name]
        if policy is not None:
            chosen = values.max(axis=1) >= policy['cutoff']
            metrics.append(measure(name, '2627_snapshot', labels, values, 'confidence_cutoff', chosen))
            predictions[name + '_covered'] = chosen
        for index, outcome in enumerate(CLASSES):
            predictions[name + '_p_' + outcome] = values[:, index]
    results = pd.DataFrame(metrics)
    results.to_csv(output / 'metrics.csv', index=False)
    predictions.to_csv(output / 'predictions.csv', index=False)
    write_json(output / 'snapshot_coverage.json', {'cutoff_date': source['cutoff_date'],
               'completed_in_source': len(fresh), 'eligible': len(test),
               'earliest_eligible_match': str(test.MatchDate.min().date()), 'latest_eligible_match': str(test.MatchDate.max().date())})
    print(results[['model', 'policy', 'covered', 'correct', 'accuracy', 'coverage', 'log_loss', 'called_D', 'hit_target']].to_string(index=False))


if __name__ == '__main__':
    main()
