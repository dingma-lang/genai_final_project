import hashlib
import json
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

from common import CLASSES, ROOT, write_json
from model import pipeline, probabilities, weekly_bootstrap
from precision_study import FEATURE_SETS, match_rows, precision_row, read_fixtures
from train_precision65 import HOLDOUTS, SEASONS

KEYS = ['Season', 'MatchDate', 'HomeTeam', 'AwayTeam']
BASE_COLUMNS = FEATURE_SETS['all_signals']
XG_COLUMNS = [f'{side}_{signal}' for side in ['home', 'away']
              for signal in ['xg_for', 'xg_against', 'finishing', 'conceding', 'venue_xg_for', 'venue_xg_against']]
PRESSURE_COLUMNS = [f'{side}_{signal}' for side in ['home', 'away']
                    for signal in ['deep_for', 'deep_against', 'ppda']]
VALIDATION = ['1920', '2021', '2122', '2223', '2324']
ALIASES = {'Manchester City': 'Man City', 'Manchester United': 'Man United',
           'Newcastle United': 'Newcastle', 'Nottingham Forest': "Nott'm Forest",
           'Queens Park Rangers': 'QPR', 'West Bromwich Albion': 'West Brom',
           'Wolverhampton Wanderers': 'Wolves'}
OUTPUT = ROOT / 'reports/xg'
MODEL_DIR = ROOT / 'models/xg'


def load_matches():
    sources = []
    for name in ['goalkeeper_sources', 'draw_lineup_sources', 'confidence_sources']:
        sources.extend(json.loads((ROOT / 'data' / (name + '.json')).read_text()))
    fixtures = read_fixtures(SEASONS, sources)
    manifest = json.loads((ROOT / 'data/xg_source.json').read_text())
    path = ROOT / manifest['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Expected-goals archive changed')
    archive = pd.read_parquet(path)
    archive = archive.loc[archive.league.eq('EPL')].copy()
    fields = [column for column in archive if column != 'season_input']
    unique = archive.drop_duplicates(fields)
    if unique.understat_id.duplicated().any():
        raise ValueError('Conflicting records for the same Understat match')
    unique = unique.copy()
    unique['HomeTeam'] = unique.home_team.replace(ALIASES)
    unique['AwayTeam'] = unique.away_team.replace(ALIASES)
    unique['MatchDate'] = pd.to_datetime(unique.match_date).dt.normalize()
    join_keys = KEYS[1:]
    joined = fixtures.merge(unique, on=join_keys, how='left', validate='one_to_one')
    present = joined.understat_id.notna()
    agreement = joined.FTHG.eq(joined.home_score) & joined.FTAG.eq(joined.away_score)
    if not agreement[present].all():
        raise ValueError('The two sources disagree on a matched score')
    xg = joined.loc[present, ['home_xg', 'away_xg']]
    if xg.isna().any().any() or not np.isfinite(xg).all().all() or xg.lt(0).any().any():
        raise ValueError('Invalid expected goals')
    coverage = joined.groupby('Season').agg(fixtures=('FTR', 'size'), matched_xg=('understat_id', 'count'))
    audit = {'archive_rows': len(archive), 'duplicate_copies_removed': len(archive) - len(unique),
             'join': 'Calendar date and both team names; one-to-one; matched scores must agree.',
             'last_archive_match': str(unique.MatchDate.max().date())}
    return joined, coverage, audit


def prior_features(history, at_home):
    recent = history[-8:]
    known = [record for record in recent if np.isfinite(record['xg_for'])]
    if len(known) < 3 or not np.isfinite(history[-1]['xg_for']):
        return None
    weights = 0.5 ** (np.arange(len(known) - 1, -1, -1) / 4)
    output = {signal: float(np.average([row[signal] for row in known], weights=weights))
              for signal in ['xg_for', 'xg_against', 'finishing', 'conceding', 'deep_for', 'deep_against', 'ppda']}
    venue = [row for row in known if row['at_home'] == at_home]
    for signal in ['xg_for', 'xg_against']:
        output['venue_' + signal] = (sum(row[signal] for row in venue) + 2 * output[signal]) / (len(venue) + 2)
    return output


def build_features(fixtures):
    history = defaultdict(list)
    rows = []
    for date, day in fixtures.groupby('MatchDate', sort=True):
        for match in day.itertuples(index=False):
            home = prior_features(history[match.HomeTeam], True)
            away = prior_features(history[match.AwayTeam], False)
            if home is not None and away is not None:
                row = {key: getattr(match, key) for key in KEYS}
                row.update({'home_' + key: value for key, value in home.items()})
                row.update({'away_' + key: value for key, value in away.items()})
                rows.append(row)
        for match in day.itertuples(index=False):
            for side, opponent, club, scored, conceded in [
                    ('home', 'away', match.HomeTeam, match.FTHG, match.FTAG),
                    ('away', 'home', match.AwayTeam, match.FTAG, match.FTHG)]:
                xg_for = getattr(match, side + '_xg')
                xg_against = getattr(match, opponent + '_xg')
                history[club].append({'xg_for': xg_for, 'xg_against': xg_against,
                                      'finishing': scored - xg_for, 'conceding': conceded - xg_against,
                                      'deep_for': getattr(match, side + '_deep'),
                                      'deep_against': getattr(match, opponent + '_deep'),
                                      'ppda': getattr(match, side + '_ppda'), 'at_home': side == 'home'})
    return match_rows(fixtures).merge(pd.DataFrame(rows), on=KEYS, validate='one_to_one')


def fit_candidate(frame, family, columns, settings):
    settings = settings.copy()
    half_life = settings.pop('half_life_days')
    age = (frame.MatchDate.max() - frame.MatchDate).dt.days.to_numpy()
    weights = np.ones(len(frame)) if half_life is None else 0.5 ** (age / half_life)
    weights /= weights.mean()
    if family == 'logistic':
        fitted = pipeline(columns, family, settings)
        return fitted.fit(frame, frame.FTR, model__sample_weight=weights)
    return CatBoostClassifier(**settings, loss_function='MultiClass', learning_rate=0.03,
                              l2_leaf_reg=5, random_seed=244, thread_count=2,
                              allow_writing_files=False, verbose=False).fit(frame[columns], frame.FTR, sample_weight=weights)


def choose_policy(outcomes, seasons, forecast):
    correct = np.array(CLASSES)[forecast.argmax(axis=1)] == outcomes
    candidates = []
    for cutoff in np.linspace(0.45, 0.85, 81):
        calls = forecast.max(axis=1) >= cutoff
        if not calls.any():
            continue
        row = {'cutoff': float(cutoff), 'precision': float(correct[calls].mean()),
               'coverage': float(calls.mean()), 'calls': int(calls.sum())}
        eligible = row['precision'] >= 0.69 and row['coverage'] >= 0.15
        for season in VALIDATION[-2:]:
            cohort = seasons == season
            selected = calls & cohort
            rate = float(correct[selected].mean()) if selected.any() else 0.0
            coverage = float(selected.sum() / cohort.sum())
            row[season + '_precision'] = rate
            row[season + '_coverage'] = coverage
            eligible &= rate >= 0.65 and coverage >= 0.10
        row['eligible'] = bool(eligible)
        candidates.append(row)
    eligible = [row for row in candidates if row['eligible']]
    policy = max(eligible, key=lambda row: (row['coverage'], row['precision'])) if eligible else None
    return policy, candidates


def candidates():
    for family in ['logistic', 'catboost']:
        for group, extra in [('history', []), ('xg', XG_COLUMNS), ('xg_pressure', XG_COLUMNS + PRESSURE_COLUMNS)]:
            configs = [{'C': c, 'half_life_days': half} for c in [0.01, 0.1, 1.0] for half in [None, 730]]
            if family == 'catboost':
                configs = [{'depth': depth, 'iterations': count, 'half_life_days': half}
                           for depth, count in [(3, 200), (4, 400)] for half in [None, 730]]
            yield family + '_' + group, family, BASE_COLUMNS + extra, configs


def main():
    fixtures, coverage, audit = load_matches()
    frame = build_features(fixtures)
    columns = BASE_COLUMNS + XG_COLUMNS + PRESSURE_COLUMNS
    if not np.isfinite(frame[columns]).all().all():
        raise ValueError('Incomplete modeling features')
    development = frame.loc[frame.Season.lt(HOLDOUTS[0])]
    test = frame.loc[frame.Season.isin(HOLDOUTS)].reset_index(drop=True)
    folds = [(development.loc[development.Season.lt(season)], development.loc[development.Season.eq(season)], season)
             for season in VALIDATION]
    if any(fit.empty or val.empty or fit.MatchDate.max() >= val.MatchDate.min() for fit, val, _ in folds):
        raise ValueError('Invalid forward validation dates')
    outcomes = np.concatenate([val.FTR.to_numpy() for _, val, _ in folds])
    seasons = np.concatenate([val.Season.to_numpy() for _, val, _ in folds])
    details, threshold_rows, selected = [], [], {}
    for name, family, columns, configs in candidates():
        trials = []
        for config in configs:
            forecasts = []
            for train, validation, season in folds:
                fitted = fit_candidate(train, family, columns, config)
                probs = probabilities(fitted, validation[columns])
                forecasts.append(probs)
                details.append({'model': name, 'settings': json.dumps(config), 'season': season,
                                'train_n': len(train), 'validation_n': len(validation),
                                'log_loss': log_loss(validation.FTR, probs, labels=CLASSES)})
            forecast = np.vstack(forecasts)
            trials.append((float(log_loss(outcomes, forecast, labels=CLASSES)), config, forecast))
        loss, config, forecast = min(trials, key=lambda trial: trial[0])
        policy, grid = choose_policy(outcomes, seasons, forecast)
        threshold_rows.extend({'model': name, **row} for row in grid)
        selected[name] = {'family': family, 'columns': columns, 'settings': config,
                          'validation_log_loss': loss, 'policy': policy}
        print(name, round(loss, 4), config, 'cutoff', policy['cutoff'] if policy else None, flush=True)
    winner = min(selected, key=lambda name: selected[name]['validation_log_loss'])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {'development_matches': len(development), 'test_matches': len(test), 'validation_seasons': VALIDATION,
                'test_seasons': HOLDOUTS, 'selected_by_validation_log_loss': winner, 'models': selected,
                'audit': audit, 'eligibility': 'Each team has at least three known xG games among its previous eight league matches and xG for its most recent league match. Existing three prior in-season games also required.',
                'timing': 'Every rolling feature uses strictly earlier calendar dates. xG history carries across seasons.',
                'evaluation_status': 'Retrospective follow-up on previously examined seasons; results need a future independent evaluation.',
                'policy_requirements': '69% pooled validation hit rate at 15% coverage, plus 65% hit rate at 10% coverage in each of the last two validation seasons.'}
    write_json(OUTPUT / 'frozen_policy.json', manifest)
    pd.DataFrame(details).to_csv(OUTPUT / 'forward_validation.csv', index=False)
    pd.DataFrame(threshold_rows).to_csv(OUTPUT / 'validation_cutoffs.csv', index=False)
    coverage['feature_eligible'] = frame.groupby('Season').size()
    coverage.fillna(0).to_csv(OUTPUT / 'coverage.csv')
    predictions = test[KEYS + ['FTR']].copy()
    metrics, forecasts = [], {}
    for name, config in selected.items():
        fitted = fit_candidate(development, config['family'], config['columns'], config['settings'])
        model_path = MODEL_DIR / (name + '.joblib')
        joblib.dump(fitted, model_path)
        probs = probabilities(fitted, test[config['columns']])
        restored = probabilities(joblib.load(model_path), test[config['columns']])
        np.testing.assert_allclose(restored, probs, rtol=1e-12, atol=1e-12)
        forecasts[name] = probs
    old = pd.read_csv(ROOT / 'reports/precision65/predictions.csv', parse_dates=['MatchDate'], dtype={'Season': str})
    old = test[KEYS].merge(old, on=KEYS, validate='one_to_one', how='left')
    for name in ['catboost', 'elo_logistic', 'market_benchmark']:
        forecasts['previous_' + name] = old[[name + '_p_' + label for label in CLASSES]].to_numpy()
    for name, probs in forecasts.items():
        if not np.isfinite(probs).all():
            raise ValueError('Missing paired benchmark forecasts')
        for index, label in enumerate(CLASSES):
            predictions[name + '_p_' + label] = probs[:, index]
        config = selected.get(name)
        policy = config['policy'] if config else None
        for season in [*HOLDOUTS, 'pooled']:
            cohort = test.Season.eq(season).to_numpy() if season != 'pooled' else np.ones(len(test), dtype=bool)
            cohort_probs = probs[cohort]
            policies = {'all_fixtures': np.ones(cohort.sum(), dtype=bool)}
            top = np.zeros(cohort.sum(), dtype=bool)
            top[np.argsort(-cohort_probs.max(axis=1), kind='stable')[:int(np.ceil(cohort.sum() / 3))]] = True
            policies['top_third_retrospective'] = top
            if policy:
                policies['confidence_cutoff'] = cohort_probs.max(axis=1) >= policy['cutoff']
            for policy_name, calls in policies.items():
                row = precision_row(name, policy_name, test.FTR.to_numpy()[cohort], cohort_probs, calls)
                row.update({'season': season, 'eligible_matches': int(cohort.sum())})
                for label in CLASSES:
                    actual = test.FTR.to_numpy()[cohort] == label
                    pred = np.array(CLASSES)[cohort_probs.argmax(axis=1)]
                    row['recall_' + label] = float((calls & actual & (pred == label)).sum() / actual.sum())
                metrics.append(row)
    for baseline in ['previous_catboost', 'catboost_history']:
        comparison = weekly_bootstrap(test, forecasts[baseline], forecasts[winner])
        comparison['limitation'] = 'Exploratory weekly block interval across two previously examined seasons; serial dependence and model selection uncertainty remain.'
        manifest[baseline + '_comparison'] = comparison
    write_json(OUTPUT / 'frozen_policy.json', manifest)
    predictions.to_csv(OUTPUT / 'predictions.csv', index=False)
    result = pd.DataFrame(metrics)
    result.to_csv(OUTPUT / 'metrics.csv', index=False)
    print('Selected', winner, flush=True)
    print(result.loc[result.season.eq('pooled'), ['model', 'policy', 'covered', 'correct', 'accuracy', 'log_loss']].to_string(index=False))


if __name__ == '__main__':
    main()
