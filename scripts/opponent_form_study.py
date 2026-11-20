import json
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLASSES, ROOT, write_json
from confidence_study import wilson_interval
from coverage_target_study import rank_selection, rolling_selection
from goal_score_study import score_probabilities
from precision_study import precision_row
from shared_goal_study import targets, team_features as shared_team_features
from train_precision65 import HOLDOUTS
from xg_study import KEYS, VALIDATION, build_features, load_matches

OUTPUT = ROOT / 'reports/precision150'
MODEL_DIR = ROOT / 'models/precision150'
TARGET_CALLS = 150
REFERENCE_MATCHES = 619
COVERAGE = TARGET_CALLS / REFERENCE_MATCHES
FORM_SIGNALS = ['attack_rating', 'defence_rating', 'adjusted_xg_for', 'adjusted_xg_against',
                'schedule_attack', 'schedule_defence', 'expected_xg']


def opponent_form(fixtures, learning_rate=0.05):
    attack, defence = defaultdict(float), defaultdict(float)
    history = defaultdict(list)
    home_intercept, away_intercept = np.log(1.5), np.log(1.2)
    current_season = None
    rows = []
    for date, day in fixtures.sort_values(['MatchDate', 'HomeTeam']).groupby('MatchDate', sort=True):
        season = day.Season.iloc[0]
        if season != current_season:
            attack = defaultdict(float, {team: 0.8 * value for team, value in attack.items()})
            defence = defaultdict(float, {team: 0.8 * value for team, value in defence.items()})
            current_season = season
        pending = []
        for match in day.itertuples(index=False):
            home, away = match.HomeTeam, match.AwayTeam
            home_rate = np.exp(np.clip(home_intercept + attack[home] + defence[away], np.log(0.1), np.log(6)))
            away_rate = np.exp(np.clip(away_intercept + attack[away] + defence[home], np.log(0.1), np.log(6)))
            row = {key: getattr(match, key) for key in KEYS}
            for side, team, opponent, expected in [('home', home, away, home_rate), ('away', away, home, away_rate)]:
                earlier = history[team][-8:]
                row[side + '_attack_rating'] = np.exp(attack[team])
                row[side + '_defence_rating'] = np.exp(defence[team])
                row[side + '_expected_xg'] = expected
                for signal in ['adjusted_xg_for', 'adjusted_xg_against', 'schedule_attack', 'schedule_defence']:
                    row[side + '_' + signal] = float(np.mean([record[signal] for record in earlier])) if earlier else 1.0
            rows.append(row)
            observed_home = match.home_xg if np.isfinite(match.home_xg) else float(match.FTHG)
            observed_away = match.away_xg if np.isfinite(match.away_xg) else float(match.FTAG)
            home_error = np.clip(observed_home - home_rate, -3, 3)
            away_error = np.clip(observed_away - away_rate, -3, 3)
            home_record = {'adjusted_xg_for': observed_home / np.exp(home_intercept + defence[away]),
                           'adjusted_xg_against': observed_away / np.exp(away_intercept + attack[away]),
                           'schedule_attack': np.exp(attack[away]), 'schedule_defence': np.exp(defence[away])}
            away_record = {'adjusted_xg_for': observed_away / np.exp(away_intercept + defence[home]),
                           'adjusted_xg_against': observed_home / np.exp(home_intercept + attack[home]),
                           'schedule_attack': np.exp(attack[home]), 'schedule_defence': np.exp(defence[home])}
            pending.append((home, away, home_error, away_error, home_record, away_record))
        for home, away, home_error, away_error, home_record, away_record in pending:
            attack[home] = float(np.clip(attack[home] + learning_rate * home_error, -1.2, 1.2))
            defence[away] = float(np.clip(defence[away] + learning_rate * home_error, -1.2, 1.2))
            attack[away] = float(np.clip(attack[away] + learning_rate * away_error, -1.2, 1.2))
            defence[home] = float(np.clip(defence[home] + learning_rate * away_error, -1.2, 1.2))
            home_intercept = float(np.clip(home_intercept + 0.002 * home_error, np.log(0.8), np.log(2.5)))
            away_intercept = float(np.clip(away_intercept + 0.002 * away_error, np.log(0.8), np.log(2.5)))
            history[home].append(home_record)
            history[away].append(away_record)
    return pd.DataFrame(rows)


def scoring_features(frame, variant):
    rows = shared_team_features(frame)
    if variant == 'base':
        return rows
    extras = {}
    signals = FORM_SIGNALS if variant == 'opponent_form' else FORM_SIGNALS + ['deep_for', 'deep_against', 'ppda']
    for prefix, sides in [('own', ['home', 'away']), ('opponent', ['away', 'home'])]:
        for signal in signals:
            extras[prefix + '_' + signal] = np.concatenate([frame[side + '_' + signal].to_numpy() for side in sides])
    return pd.concat([rows, pd.DataFrame(extras)], axis=1)


def fit_scoring(frame, config):
    if config['family'] == 'linear':
        fitted = make_pipeline(StandardScaler(), PoissonRegressor(alpha=config['alpha'], max_iter=1500))
    else:
        fitted = CatBoostRegressor(depth=config['depth'], iterations=config['iterations'], learning_rate=0.03,
                                   loss_function='Poisson', l2_leaf_reg=8, random_seed=244,
                                   thread_count=2, allow_writing_files=False, verbose=False)
    x = scoring_features(frame, config['features'])
    y = targets(frame, config['target'])
    ages = (frame.MatchDate.max() - frame.MatchDate).dt.days.to_numpy()
    weights = np.ones(len(frame)) if config['half_life'] is None else 0.5 ** (ages / config['half_life'])
    weights = np.tile(weights / weights.mean(), 2)
    if config['family'] == 'linear':
        return fitted.fit(x, y, poissonregressor__sample_weight=weights)
    return fitted.fit(x, y, sample_weight=weights)


def predict_scoring(fitted, frame, config):
    x = scoring_features(frame, config['features'])
    rates = fitted.predict(x, prediction_type='Exponent', thread_count=2) if config['family'] == 'catboost' else fitted.predict(x)
    rates = np.column_stack([rates[:len(frame)], rates[len(frame):]])
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError('Invalid expected goal rates')
    return score_probabilities(rates, 0)


def configurations():
    for features in ['base', 'opponent_form', 'opponent_form_pressure']:
        for target in ['goals', 'goals_and_xg']:
            for alpha in [0.03, 0.1, 0.3]:
                yield {'family': 'linear', 'features': features, 'target': target, 'alpha': alpha, 'half_life': None}
            for depth, iterations in [(2, 300), (3, 300), (4, 600)]:
                for half_life in [None, 730]:
                    yield {'family': 'catboost', 'features': features, 'target': target, 'depth': depth,
                           'iterations': iterations, 'half_life': half_life}


def validation_score(frame, forecast):
    correct = np.array(CLASSES)[forecast.argmax(axis=1)] == frame.FTR.to_numpy()
    selected = np.zeros(len(frame), dtype=bool)
    recent = []
    for season in VALIDATION:
        indices = np.flatnonzero(frame.Season.eq(season).to_numpy())
        picked = rank_selection(forecast[indices].max(axis=1), int(round(len(indices) * COVERAGE)))
        selected[indices] = picked
        if season in VALIDATION[-2:]:
            recent.append(float(correct[indices][picked].mean()))
    return {'precision': float(correct[selected].mean()), 'calls': int(selected.sum()),
            'correct': int(correct[selected].sum()), 'minimum_recent_precision': min(recent),
            'log_loss': float(log_loss(frame.FTR, forecast, labels=CLASSES))}


def evaluate(test, name, forecast, selections):
    rows = []
    predicted = np.array(CLASSES)[forecast.argmax(axis=1)]
    for season in [*HOLDOUTS, 'pooled']:
        cohort = test.Season.eq(season).to_numpy() if season != 'pooled' else np.ones(len(test), dtype=bool)
        for policy, selected in selections.items():
            row = precision_row(name, policy, test.FTR.to_numpy()[cohort], forecast[cohort], selected[cohort])
            row.update({'season': season, 'eligible_matches': int(cohort.sum())})
            row['wilson_lower'], row['wilson_upper'] = wilson_interval(row['correct'], row['covered'])
            for label in CLASSES:
                actual = test.FTR.to_numpy() == label
                count = int((actual & cohort).sum())
                row['recall_' + label] = float((actual & cohort & selected & (predicted == label)).sum() / count) if count else None
            rows.append(row)
    return rows


def main():
    fixtures, _, _ = load_matches()
    form = opponent_form(fixtures)
    frame = build_features(fixtures).merge(form, on=KEYS, validate='one_to_one')
    frame = frame.merge(fixtures[KEYS + ['FTHG', 'FTAG', 'home_xg', 'away_xg']], on=KEYS, validate='one_to_one')
    development = frame.loc[frame.Season.lt(HOLDOUTS[0])]
    training = development.dropna(subset=['home_xg', 'away_xg'])
    test = frame.loc[frame.Season.isin(HOLDOUTS)].reset_index(drop=True)
    if len(test) != REFERENCE_MATCHES:
        raise ValueError('The reference evaluation sample changed')
    folds = [(training.loc[training.Season.lt(season)], development.loc[development.Season.eq(season)]) for season in VALIDATION]
    if any(train.MatchDate.max() >= val.MatchDate.min() for train, val in folds):
        raise ValueError('Invalid chronological validation fold')
    validation = pd.concat([val for _, val in folds]).reset_index(drop=True)
    trials = []
    forecast_cache = {}
    for number, config in enumerate(configurations()):
        forecasts = []
        for train, val in folds:
            fitted = fit_scoring(train, config)
            forecasts.append(predict_scoring(fitted, val, config))
        forecast = np.vstack(forecasts)
        score = validation_score(validation, forecast)
        name = f'candidate_{number:02d}'
        trials.append({'name': name, 'config': config, **score})
        forecast_cache[name] = forecast
        print(name, config['family'], config['features'], config['target'], 'precision', round(score['precision'], 4), flush=True)
    winner = max(trials, key=lambda row: (row['precision'], row['minimum_recent_precision'], -row['log_loss']))
    chosen = winner['name']
    scores = forecast_cache[chosen].max(axis=1)
    cutoff = float(np.quantile(scores, 1 - COVERAGE, method='higher'))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {'target_precision': 0.70, 'target_calls': TARGET_CALLS, 'reference_matches': len(test),
                'target_coverage': COVERAGE, 'validation_seasons': VALIDATION, 'test_seasons': HOLDOUTS,
                'selected_candidate': winner, 'fixed_cutoff': cutoff,
                'opponent_form_settings': {'learning_rate': 0.05, 'season_retention': 0.8, 'window': 8},
                'selection': 'Greatest validation precision at 150/619 coverage in each season, then minimum recent-season precision, then lowest log loss.',
                'timing': 'All team ratings and adjusted form use earlier calendar dates. Missing xG updates use the earlier actual goals.',
                'evaluation_status': 'Retrospective follow-up on previously examined seasons. Independent future validation is required.'}
    write_json(OUTPUT / 'frozen_policy.json', manifest)
    pd.DataFrame([{**row, 'config': json.dumps(row['config'], sort_keys=True)} for row in trials]).to_csv(OUTPUT / 'validation_search.csv', index=False)
    print('Frozen selected model', winner, flush=True)
    final = fit_scoring(training, winner['config'])
    bundle = {'model': final, 'config': winner['config'], 'fixed_cutoff': cutoff,
              'initial_confidence_history': scores[-350:].tolist(), 'target_coverage': COVERAGE}
    path = MODEL_DIR / 'scoring_forecaster.joblib'
    joblib.dump(bundle, path)
    forecast = predict_scoring(final, test, winner['config'])
    restored = joblib.load(path)
    np.testing.assert_allclose(predict_scoring(restored['model'], test, restored['config']), forecast, rtol=1e-12, atol=1e-12)
    confidence = forecast.max(axis=1)
    rolling, thresholds = rolling_selection(test.MatchDate, confidence, scores[-350:], coverage=COVERAGE)
    selections = {'all_fixtures': np.ones(len(test), dtype=bool), 'top_150_batch': rank_selection(confidence, TARGET_CALLS),
                  'fixed_cutoff': confidence >= cutoff, 'rolling_cutoff': rolling}
    rows = evaluate(test, 'selected_scoring_model', forecast, selections)
    predictions = test[KEYS + ['FTR']].copy()
    for index, label in enumerate(CLASSES):
        predictions['p_' + label] = forecast[:, index]
    for name, selected in selections.items():
        predictions[name] = selected
    predictions['rolling_threshold'] = thresholds
    predictions.to_csv(OUTPUT / 'predictions.csv', index=False)
    previous = pd.read_csv(ROOT / 'reports/shared_goals/predictions.csv', parse_dates=['MatchDate'], dtype={'Season': str})
    previous = test[KEYS].merge(previous, on=KEYS, how='left', validate='one_to_one')
    for name in ['linear_goals', 'catboost_goals']:
        probs = previous[[name + '_p_' + label for label in CLASSES]].to_numpy()
        rows.extend(evaluate(test, 'previous_' + name, probs, {'top_150_batch': rank_selection(probs.max(axis=1), TARGET_CALLS)}))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / 'metrics.csv', index=False)
    print(metrics.loc[metrics.season.eq('pooled'), ['model', 'policy', 'covered', 'correct', 'accuracy', 'coverage', 'log_loss']].to_string(index=False))


if __name__ == '__main__':
    main()
